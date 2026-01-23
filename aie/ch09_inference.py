"""Chapitre 9 : optimisation de l'inférence.

Fonctions réutilisables des notebooks du chapitre 9 :

1. matériel : fiches d'accélérateurs, intensité arithmétique, modèle roofline ;
2. métriques : TTFT, TPOT, percentiles, goodput, coût par million de tokens, MFU et MBU ;
3. mesures GPU robustes (médiane de répétitions, relance automatique après un OOM) ;
4. compression : quantification simulée (par tenseur, par canal, par groupe), élagage ;
5. décodage : génération gloutonne avec ou sans cache KV, décodage spéculatif, inférence
   avec référence (*prompt lookup*), décodage de Jacobi ; formules théoriques associées ;
6. attention : taille du cache KV (formule du livre), allocation contiguë contre paginée ;
7. service : simulation à événements discrets du batching statique, dynamique et continu,
   prefill et decode colocalisés ou dissociés, cache de prompt, parallélisme.

Les fonctions « modèle » (sections 3 et 5) importent torch à la demande ; tout le reste est
du numpy pur, testé sans GPU.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# =====================================================================================
# 1. Matériel et modèle roofline
# =====================================================================================

#: Fiches simplifiées (valeurs **denses**, sans parcimonie, arrondies, d'après les fiches
#: constructeur). flops : FLOP/s de pointe par précision ; bw : bande passante mémoire (octets/s) ;
#: mem : capacité (octets) ; tdp : puissance (W). Pour la RTX 5070 Laptop, la puissance de
#: calcul en bf16 est une **estimation** (36 SM x 512 FLOP/cycle x ~3,1 GHz), la puissance
#: dépendant fortement du plafond fixé par le constructeur du portable.
ACCELERATORS = {
    "CPU portable (DDR5)": dict(flops={"fp32": 1.0e12, "bf16": 1.0e12}, bw=70e9, mem=32e9, tdp=55,
                                kind="CPU"),
    "RTX 5070 Laptop": dict(flops={"fp32": 28e12, "tf32": 28e12, "bf16": 57e12, "fp16": 57e12, "fp8": 114e12,
                                   "int8": 114e12, "fp4": 228e12},
                            bw=384e9, mem=8e9, tdp=100, kind="GPU grand public"),
    "A100 SXM 80 Go": dict(flops={"fp32": 19.5e12, "tf32": 156e12, "bf16": 312e12, "fp16": 312e12,
                                  "int8": 624e12},
                           bw=2.04e12, mem=80e9, tdp=400, kind="GPU datacenter"),
    "H100 SXM": dict(flops={"fp32": 67e12, "tf32": 494.5e12, "bf16": 989e12, "fp16": 989e12, "fp8": 1979e12,
                            "int8": 1979e12},
                     bw=3.35e12, mem=80e9, tdp=700, kind="GPU datacenter"),
    "B200": dict(flops={"tf32": 1100e12, "bf16": 2250e12, "fp16": 2250e12, "fp8": 4500e12, "fp4": 9000e12},
                 bw=8e12, mem=192e9, tdp=1000, kind="GPU datacenter"),
    "MI300X": dict(flops={"fp32": 163e12, "tf32": 654e12, "bf16": 1307e12, "fp16": 1307e12, "fp8": 2615e12},
                   bw=5.3e12, mem=192e9, tdp=750, kind="GPU datacenter"),
    "TPU v5e": dict(flops={"bf16": 197e12, "int8": 394e12}, bw=819e9, mem=16e9, tdp=200, kind="TPU"),
}

BYTES = {"fp32": 4, "tf32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5, "fp4": 0.5, "nf4": 0.5}


def matmul_flops(m: int, k: int, n: int) -> float:
    """FLOPs d'un produit (m, k) x (k, n) : une multiplication et une addition par terme."""
    return 2.0 * m * k * n


def matmul_bytes(m: int, k: int, n: int, bytes_per: float = 2) -> float:
    """Octets déplacés au minimum : lire A (m k) et B (k n), écrire C (m n)."""
    return float(bytes_per) * (m * k + k * n + m * n)


def arithmetic_intensity(flops: float, nbytes: float) -> float:
    """Intensité arithmétique : opérations par octet lu ou écrit en mémoire."""
    return flops / nbytes


def matmul_intensity(m: int, k: int, n: int, bytes_per: float = 2) -> float:
    return arithmetic_intensity(matmul_flops(m, k, n), matmul_bytes(m, k, n, bytes_per))


def ridge_point(peak_flops: float, bandwidth: float) -> float:
    """Intensité (FLOP/octet) à partir de laquelle une opération devient limitée par le calcul."""
    return peak_flops / bandwidth


def roofline(intensity, peak_flops: float, bandwidth: float):
    """Débit atteignable (FLOP/s) : min(pic de calcul, intensité x bande passante)."""
    return np.minimum(peak_flops, np.asarray(intensity, dtype=float) * bandwidth)


def bound_kind(intensity: float, peak_flops: float, bandwidth: float) -> str:
    return "calcul" if intensity >= ridge_point(peak_flops, bandwidth) else "mémoire"


def decode_intensity(n_params: float, batch: int, bytes_per: float = 2, kv_bytes_per_seq: float = 0.0) -> float:
    """Intensité d'un pas de decode : 2 N B FLOPs pour lire N b octets de poids (+ B caches KV)."""
    return 2.0 * n_params * batch / (n_params * bytes_per + batch * kv_bytes_per_seq)


def prefill_intensity(d_model: int, d_out: int, n_tokens: int, bytes_per: float = 2) -> float:
    """Intensité d'une couche linéaire (d_model -> d_out) appliquée à n_tokens tokens à la fois."""
    return matmul_intensity(n_tokens, d_model, d_out, bytes_per)


def time_lower_bound(flops: float, nbytes: float, peak_flops: float, bandwidth: float) -> float:
    """Borne inférieure de durée : max(temps de calcul, temps de transfert)."""
    return max(flops / peak_flops, nbytes / bandwidth)


def decode_tokens_per_s_bound(n_params: float, bandwidth: float, bytes_per: float = 2, batch: int = 1,
                              kv_bytes_per_seq: float = 0.0) -> float:
    """Débit maximal (tokens/s, tous utilisateurs) d'un decode limité par la bande passante."""
    t_step = (n_params * bytes_per + batch * kv_bytes_per_seq) / bandwidth
    return batch / t_step


def energy_per_token_j(power_w: float, tokens_per_s: float) -> float:
    """Énergie par token (joules) à puissance et débit donnés."""
    return power_w / tokens_per_s


def accelerators_table() -> pd.DataFrame:
    rows = []
    for name, a in ACCELERATORS.items():
        bf = a["flops"].get("bf16", np.nan)
        rows.append({"accélérateur": name, "type": a["kind"], "bf16 (TFLOP/s)": bf / 1e12,
                     "bande passante (To/s)": a["bw"] / 1e12, "mémoire (Go)": a["mem"] / 1e9,
                     "TDP (W)": a["tdp"], "point d'équilibre (FLOP/octet)": bf / a["bw"],
                     "TFLOP/s par W": bf / 1e12 / a["tdp"]})
    return pd.DataFrame(rows)


# =====================================================================================
# 2. Métriques
# =====================================================================================


def total_latency(ttft: float, tpot: float, n_output: int) -> float:
    """Latence totale = TTFT + TPOT x (nombre de tokens de sortie - 1).

    Le livre écrit TTFT + TPOT x n ; nous comptons le premier token dans le TTFT, d'où n - 1.
    """
    return ttft + tpot * max(n_output - 1, 0)


def percentiles(values, qs=(50, 90, 95, 99)) -> dict:
    v = np.asarray(values, dtype=float)
    return {f"p{q}": float(np.percentile(v, q)) for q in qs} | {"moyenne": float(v.mean())}


def goodput(ttft, tpot, duration_s: float, slo_ttft: float, slo_tpot: float) -> tuple[float, float]:
    """(débit en requêtes/s, goodput en requêtes/s respectant les deux SLO)."""
    ttft, tpot = np.asarray(ttft, float), np.asarray(tpot, float)
    ok = (ttft <= slo_ttft) & (tpot <= slo_tpot)
    return len(ttft) / duration_s, float(ok.sum()) / duration_s


def cost_per_million_tokens(price_per_hour: float, tokens_per_s: float) -> float:
    """Coût (dollars) d'un million de tokens pour une machine louée price_per_hour dollars/heure."""
    return price_per_hour / (tokens_per_s * 3600.0) * 1e6


def mbu(n_params: float, bytes_per: float, tokens_per_s: float, bandwidth: float,
        kv_bytes_per_token_step: float = 0.0) -> float:
    """Model Bandwidth Utilization : (N b x tokens/s [+ lecture du cache]) / bande passante."""
    return (n_params * bytes_per + kv_bytes_per_token_step) * tokens_per_s / bandwidth


def mfu(flops_per_token: float, tokens_per_s: float, peak_flops: float) -> float:
    """Model FLOP/s Utilization : FLOP/s utiles observés / FLOP/s de pointe."""
    return flops_per_token * tokens_per_s / peak_flops


def flops_per_token(n_params: float, n_layers: int = 0, d_attn: int = 0, context: int = 0) -> float:
    """FLOPs d'une passe avant par token : 2 N (couches linéaires) + 4 L T d (scores et somme pondérée)."""
    return 2.0 * n_params + 4.0 * n_layers * context * d_attn


# =====================================================================================
# 3. Mesures sur GPU
# =====================================================================================


def sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_fn(fn, repeats: int = 7, warmup: int = 2) -> dict:
    """Chronomètre fn (synchronisée avec le GPU) : médiane, meilleure valeur et toutes les mesures.

    Le GPU est partagé : la médiane résiste aux mesures perturbées, la meilleure valeur
    approche ce que le matériel permet sans concurrence.
    """
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return {"median": statistics.median(ts), "best": min(ts), "all": ts}


def retry_oom(fn, *args, wait: float = 45.0, tries: int = 12, **kw):
    """Exécute fn ; en cas d'erreur CUDA out-of-memory, libère le cache, attend `wait` s et relance."""
    import gc

    import torch

    for i in range(tries):
        try:
            return fn(*args, **kw)
        except torch.cuda.OutOfMemoryError:
            if i == tries - 1:
                raise
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(wait)


def load_model(model_id: str, quant: str = "bf16"):
    """Charge un modèle causal en bf16, en int8 (LLM.int8 de bitsandbytes) ou en nf4 (4 bits).

    Contrairement à llm.load, ce chargement n'est pas mis en cache : on peut comparer
    plusieurs variantes en les libérant entre deux (del model ; llm.free_gpu()).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.utils import logging as hf_logging

    import logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    logging.getLogger("bitsandbytes").setLevel(logging.ERROR)  # avertissements répétés de LLM.int8
    tok = AutoTokenizer.from_pretrained(model_id)
    kw = dict(dtype=torch.bfloat16)
    if quant == "int8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    if quant == "bf16":
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0}, **kw)
    model.eval()
    return tok, model


def random_ids(batch: int, length: int, seed: int = 0, device: str = "cuda"):
    """Identifiants de tokens aléatoires (pour chronométrer sans dépendre d'un texte)."""
    import torch

    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, 20000, (batch, length), generator=g).to(device)


def prefill_time(model, n_tokens: int, batch: int = 1, repeats: int = 5) -> dict:
    """Durée d'un prefill de n_tokens tokens (par séquence) pour un lot de `batch` séquences."""
    import torch

    x = random_ids(batch, n_tokens, device=model.device)
    with torch.no_grad():
        return time_fn(lambda: model(x, use_cache=True, logits_to_keep=1), repeats=repeats, warmup=1)


def decode_step_time(model, batch: int = 1, context: int = 128, steps: int = 24, warmup: int = 4) -> dict:
    """Durée d'un pas de decode (un token par séquence) pour `batch` séquences de `context` tokens.

    Renvoie médiane et meilleure valeur sur `steps` pas (après `warmup` pas ignorés)."""
    import torch

    x = random_ids(batch, context, device=model.device)
    ts = []
    with torch.no_grad():
        o = model(x, use_cache=True, logits_to_keep=1)
        c, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
        for i in range(steps + warmup):
            sync()
            t0 = time.perf_counter()
            o = model(nxt, cache=c, use_cache=True, logits_to_keep=1)
            c, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
            sync()
            if i >= warmup:
                ts.append(time.perf_counter() - t0)
    del c, o
    return {"median": statistics.median(ts), "best": min(ts), "all": ts}


def perplexity_of(model, tok, text: str) -> float:
    """Perplexité d'un modèle déjà chargé sur un texte (une seule passe avant)."""
    import torch

    ids = tok(text, return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        out = model(ids, labels=ids)
    return float(torch.exp(out.loss.float()))


def _ids_tensor(ids, device):
    import torch

    return torch.tensor([list(ids)], device=device)


def _crop_to(cache, length: int) -> None:
    """Ramène un cache KV Hugging Face à `length` positions."""
    extra = cache.get_seq_length() - length
    if extra > 0:
        cache.crop(-extra)  # transformers >= 5 : valeur négative = nombre de positions à retirer


def eos_ids(tok) -> set:
    ids = {tok.eos_token_id}
    for s in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(s)
        if isinstance(i, int) and i >= 0:
            ids.add(i)
    return {i for i in ids if i is not None}


@dataclass
class GenResult:
    tokens: list
    seconds: float
    forwards: int = 0
    rounds: list = field(default_factory=list)  # nombre de propositions acceptées par tour
    proposed: list = field(default_factory=list)  # nombre de propositions par tour
    times: list = field(default_factory=list)  # instant (s, depuis le début) où chaque token est prêt

    @property
    def ttft(self) -> float:
        return self.times[0] if self.times else float("nan")

    @property
    def tpot(self) -> float:
        return float(np.mean(np.diff(self.times))) if len(self.times) > 1 else float("nan")

    @property
    def tokens_per_s(self) -> float:
        return len(self.tokens) / self.seconds

    @property
    def acceptance(self) -> float:
        """Fraction des tokens proposés qui ont été acceptés."""
        p = sum(self.proposed)
        return sum(self.rounds) / p if p else float("nan")


def greedy_generate(model, input_ids, max_new_tokens: int = 64, use_cache: bool = True, stop=None) -> GenResult:
    """Décodage glouton écrit à la main, avec ou sans cache KV (sans cache : on recalcule tout le
    préfixe à chaque token, ce qui rend le coût d'un pas proportionnel à la longueur)."""
    import torch

    dev = model.device
    seq = list(input_ids)
    stop = stop or set()
    out, past, times = [], None, []
    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        cur = _ids_tensor(seq, dev)
        for _ in range(max_new_tokens):
            if use_cache:
                o = model(cur, cache=past, use_cache=True, logits_to_keep=1)
                past = o.past_key_values
            else:
                o = model(_ids_tensor(seq, dev), use_cache=False, logits_to_keep=1)
            t = int(o.logits[0, -1].argmax())  # .item() implicite : attend la fin du calcul GPU
            times.append(time.perf_counter() - t0)
            if t in stop:
                break
            out.append(t)
            seq.append(t)
            cur = _ids_tensor([t], dev)
    sync()
    return GenResult(out, time.perf_counter() - t0, forwards=len(out), times=times[:len(out)])


class DraftModelProposer:
    """Propose k tokens en décodant glouton avec un petit modèle (qui garde son propre cache KV)."""

    def __init__(self, model):
        self.model, self.cache, self.n_cached = model, None, 0

    def reset(self):
        self.cache, self.n_cached = None, 0

    def __call__(self, seq, k: int):
        import torch

        if self.cache is not None:
            _crop_to(self.cache, min(self.n_cached, len(seq) - 1))
            self.n_cached = self.cache.get_seq_length()
        feed = seq[self.n_cached:]
        props = []
        with torch.no_grad():
            for _ in range(k):
                o = self.model(_ids_tensor(feed, self.model.device), cache=self.cache, use_cache=True, logits_to_keep=1)
                self.cache = o.past_key_values
                t = int(o.logits[0, -1].argmax())
                props.append(t)
                feed = [t]
        self.n_cached = self.cache.get_seq_length()
        return props


def prompt_lookup_proposer(ngram: int = 3, min_ngram: int = 1):
    """Inférence avec référence : cherche dans le contexte la dernière occurrence des derniers
    tokens générés (n-gramme, du plus long au plus court) et propose les tokens qui la suivaient."""

    def propose(seq, k: int):
        n_seq = len(seq)
        for n in range(ngram, min_ngram - 1, -1):
            if n_seq <= n:
                continue
            pat = seq[-n:]
            for start in range(n_seq - n - 1, -1, -1):
                if seq[start:start + n] == pat:
                    cont = seq[start + n:start + n + k]
                    if cont:
                        return list(cont)
        return []

    return propose


def assisted_generate(target, input_ids, propose, k: int = 4, max_new_tokens: int = 64, stop=None) -> GenResult:
    """Décodage glouton « assisté » : à chaque tour, `propose(seq, k)` suggère jusqu'à k tokens
    (petit modèle de brouillon, copie du contexte...) ; la cible les vérifie en **une seule**
    passe avant, garde le plus long préfixe sur lequel elle est d'accord, puis ajoute son propre
    token suivant. Le résultat est identique au glouton de la cible (aux arrondis bf16 près).
    """
    import torch

    dev = target.device
    seq = list(input_ids)
    stop = stop or set()
    out, rounds, proposed = [], [], []
    cache, forwards = None, 0
    if hasattr(propose, "reset"):
        propose.reset()
    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        # prefill de la cible sur tout le contexte sauf le dernier token
        if len(seq) > 1:
            cache = target(_ids_tensor(seq[:-1], dev), use_cache=True, logits_to_keep=1).past_key_values
        n_cached = len(seq) - 1
        done = False
        while len(out) < max_new_tokens and not done:
            budget = max_new_tokens - len(out)
            props = list(propose(seq, min(k, budget - 1))) if budget > 1 else []
            feed = seq[n_cached:] + props
            o = target(_ids_tensor(feed, dev), cache=cache, use_cache=True, logits_to_keep=len(props) + 1)
            forwards += 1
            cache = o.past_key_values
            pred = o.logits[0, -(len(props) + 1):].argmax(-1).tolist()
            j = 0
            while j < len(props) and props[j] == pred[j]:
                j += 1
            new = props[:j] + [pred[j]]
            rounds.append(j)
            proposed.append(len(props))
            for t in new:
                if t in stop or len(out) >= max_new_tokens:
                    done = True
                    break
                out.append(t)
                seq.append(t)
            _crop_to(cache, len(seq) - 1)
            n_cached = cache.get_seq_length()
    sync()
    return GenResult(out, time.perf_counter() - t0, forwards=forwards, rounds=rounds, proposed=proposed)


def jacobi_decode(model, input_ids, block: int = 8, max_iter: int | None = None, init_token: int | None = None):
    """Décodage de Jacobi (glouton) d'un bloc de `block` tokens.

    On part d'une supposition arbitraire pour les `block` tokens futurs ; à chaque itération,
    une passe avant **parallèle** recalcule tous les tokens du bloc à partir de la supposition
    courante. Point fixe atteint en au plus `block` itérations : c'est exactement la suite
    gloutonne. Renvoie (tokens du bloc, nombre d'itérations, historique des suppositions).
    """
    import torch

    dev = model.device
    prefix = list(input_ids)
    max_iter = max_iter or block
    guess = [init_token if init_token is not None else prefix[-1]] * block
    history = [list(guess)]
    with torch.no_grad():
        for it in range(1, max_iter + 1):
            logits = model(_ids_tensor(prefix + guess, dev)).logits[0]
            new = logits[len(prefix) - 1:len(prefix) - 1 + block].argmax(-1).tolist()
            history.append(new)
            if new == guess:
                return new, it - 1 if it > 1 else 1, history[:-1]
            guess = new
    return guess, max_iter, history


def expected_tokens_per_round(alpha, k: int):
    """Nombre moyen de tokens produits par tour de décodage spéculatif :
    E = 1 + alpha + ... + alpha^k = (1 - alpha^(k+1)) / (1 - alpha) (acceptations indépendantes)."""
    a = np.asarray(alpha, dtype=float)
    return np.where(np.isclose(a, 1.0), k + 1.0, (1 - a ** (k + 1)) / (1 - np.where(np.isclose(a, 1.0), 0.5, a)))


def speculative_speedup(alpha, k: int, c: float, verify_cost: float = 1.0):
    """Accélération attendue (Leviathan et al., 2023) : E / (k c + verify_cost), où c est le coût
    d'un pas du brouillon relatif à un pas de la cible et verify_cost le coût de la vérification
    de k + 1 tokens en une passe (proche de 1 tant que le decode est limité par la mémoire)."""
    return expected_tokens_per_round(alpha, k) / (k * c + verify_cost)


def optimal_k(alpha: float, c: float, k_max: int = 16, verify_cost: float = 1.0) -> int:
    ks = np.arange(1, k_max + 1)
    return int(ks[np.argmax([speculative_speedup(alpha, int(k), c, verify_cost) for k in ks])])


def estimate_alpha(accepted, proposed) -> float:
    """Estimateur du maximum de vraisemblance du taux d'acceptation par token (modèle géométrique
    tronqué) : acceptations / (acceptations + rejets), un rejet étant un tour où j < proposés."""
    acc = np.asarray(accepted)
    prop = np.asarray(proposed)
    rejects = int((acc < prop).sum())
    return float(acc.sum() / (acc.sum() + rejects)) if (acc.sum() + rejects) else float("nan")


# =====================================================================================
# 4. Compression : quantification et élagage
# =====================================================================================


def quantize_symmetric(x, bits: int = 8, axis=None, group_size: int | None = None):
    """Quantification symétrique « absmax » (arrondi au plus proche), puis déquantification.

    axis=None : une échelle pour tout le tenseur ; axis=k : une échelle par tranche le long de
    l'axe k (par canal) ; group_size : une échelle par groupe de group_size valeurs consécutives
    de la dernière dimension. Renvoie (x déquantifié, nombre d'échelles).
    """
    x = np.asarray(x, dtype=np.float64)
    qmax = 2 ** (bits - 1) - 1
    if group_size:
        shp = x.shape
        g = x.reshape(-1, group_size)
        s = np.abs(g).max(axis=1, keepdims=True) / qmax
        s[s == 0] = 1
        return (np.clip(np.round(g / s), -qmax - 1, qmax) * s).reshape(shp), s.size
    if axis is None:
        s = np.abs(x).max() / qmax or 1.0
        return np.clip(np.round(x / s), -qmax - 1, qmax) * s, 1
    s = np.abs(x).max(axis=axis, keepdims=True) / qmax
    s[s == 0] = 1
    return np.clip(np.round(x / s), -qmax - 1, qmax) * s, s.size


def quant_bits_per_weight(bits: int, group_size: int | None, scale_bits: int = 16) -> float:
    """Bits effectifs par poids, échelles comprises (une échelle de scale_bits par groupe)."""
    return bits + (scale_bits / group_size if group_size else 0.0)


def fake_quantize_model_(model, bits: int = 4, group_size: int | None = 128, skip=("lm_head", "embed")) -> int:
    """Remplace en place les poids des couches linéaires par leur version quantifiée-déquantifiée
    (simulation de la quantification poids seuls). Renvoie le nombre de matrices modifiées."""
    import torch

    n = 0
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and not any(s in name for s in skip):
                w = mod.weight.data.float()
                if group_size:
                    g = w.reshape(-1, group_size)
                    s = g.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
                    wq = (torch.clamp(torch.round(g / s), -qmax - 1, qmax) * s).reshape(w.shape)
                else:  # une échelle par canal de sortie
                    s = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
                    wq = torch.clamp(torch.round(w / s), -qmax - 1, qmax) * s
                mod.weight.data.copy_(wq.to(mod.weight.dtype))
                n += 1
    return n


def linear_input_norms(model, tok, text: str, scope: str = "mlp") -> dict:
    """Norme L2, sur les tokens d'un texte de calibration, de chaque composante d'entrée des couches
    linéaires visées (nom -> vecteur). Sert aux critères d'élagage tenant compte des activations."""
    import torch

    norms, hooks = {}, []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and scope in name:
            def hook(m, inp, out, name=name):
                norms[name] = inp[0].float().reshape(-1, inp[0].shape[-1]).norm(dim=0)
            hooks.append(mod.register_forward_hook(hook))
    ids = tok(text, return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()
    return norms


def magnitude_prune_(model, sparsity: float, scope: str = "mlp", method: str = "magnitude", seed: int = 0,
                     input_norms: dict | None = None) -> float:
    """Élagage non structuré : met à zéro la fraction `sparsity` des poids les moins importants de
    chaque matrice des couches visées. method : "magnitude" (|w|), "wanda" (|w| x norme de
    l'entrée correspondante, Sun et al. 2023, nécessite input_norms) ou "random".
    Renvoie la fraction de poids nuls parmi les matrices visées."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    zeros = total = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and scope in name:
                w = mod.weight.data
                k = int(sparsity * w.numel())
                if k > 0:
                    if method == "random":
                        mask = (torch.rand(w.shape, generator=g) >= sparsity).to(w.device)
                    else:
                        score = w.abs().float()
                        if method == "wanda":
                            score = score * input_norms[name][None, :]
                        thr = score.flatten().kthvalue(k).values
                        mask = score > thr
                    w.mul_(mask.to(w.dtype))
                zeros += int((w == 0).sum())
                total += w.numel()
    return zeros / max(total, 1)


def structured_prune_mlp_(model, fraction: float, method: str = "activation", input_norms: dict | None = None,
                          seed: int = 0) -> int:
    """Élagage structuré : supprime (met à zéro) une fraction des neurones de chaque MLP SwiGLU.

    method="weights" : neurones dont les poids (ligne de up_proj x colonne de down_proj) ont la
    plus petite norme ; "activation" : plus petite norme de l'activation (mesurée sur un texte de
    calibration, input_norms des down_proj) x norme de la colonne de down_proj ; "random".
    Renvoie le nombre de neurones ôtés.
    """
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    removed = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if hasattr(mod, "down_proj") and hasattr(mod, "up_proj"):
                col = mod.down_proj.weight.float().norm(dim=0)
                if method == "weights":
                    score = col * mod.up_proj.weight.float().norm(dim=1)
                elif method == "activation":
                    score = col * input_norms[name + ".down_proj"]
                else:
                    score = torch.rand(col.shape, generator=g).to(col.device)
                k = int(fraction * score.numel())
                if k:
                    idx = score.argsort()[:k]
                    mod.up_proj.weight[idx] = 0
                    mod.gate_proj.weight[idx] = 0
                    mod.down_proj.weight[:, idx] = 0
                    removed += k
    return removed


def two_four_mask(w):
    """Masque de parcimonie semi-structurée 2:4 : dans chaque groupe de 4 poids consécutifs,
    on garde les 2 de plus grande valeur absolue (format accéléré par les tensor cores NVIDIA)."""
    w = np.asarray(w)
    g = np.abs(w).reshape(-1, 4)
    keep = np.argsort(-g, axis=1)[:, :2]
    mask = np.zeros_like(g, dtype=bool)
    np.put_along_axis(mask, keep, True, axis=1)
    return mask.reshape(w.shape)


# =====================================================================================
# 5. Cache KV et allocation mémoire
# =====================================================================================


def kv_cache_bytes_book(batch: int, seq_len: int, n_layers: int, d_model: int, bytes_per: float = 2) -> float:
    """Formule du livre (attention multi-têtes, sans optimisation) : 2 x B x S x L x H x M."""
    return 2.0 * batch * seq_len * n_layers * d_model * bytes_per


def kv_cache_bytes_gqa(batch: int, seq_len: int, n_layers: int, n_kv_heads: int, head_dim: int,
                       bytes_per: float = 2, window: int | None = None, layer_share: int = 1) -> float:
    """Cache KV avec GQA/MQA (n_kv_heads têtes KV), fenêtre locale éventuelle et partage entre couches."""
    s = min(seq_len, window) if window else seq_len
    return 2.0 * batch * s * math.ceil(n_layers / layer_share) * n_kv_heads * head_dim * bytes_per


def kv_allocation_stats(lengths, max_len: int, block_size: int) -> dict:
    """Mémoire (en positions de token) réservée par trois stratégies d'allocation du cache KV :
    réservation contiguë de max_len par requête, allocation exacte (idéale) et blocs de taille
    block_size (PagedAttention). Renvoie la mémoire réservée et la part gaspillée."""
    L = np.asarray(lengths, dtype=int)
    used = int(L.sum())
    contiguous = len(L) * max_len
    paged = int((np.ceil(L / block_size) * block_size).sum())
    return {"utile": used, "contigu (max_len)": contiguous, "paginé": paged,
            "gaspillage contigu": 1 - used / contiguous, "gaspillage paginé": 1 - used / paged}


def max_concurrent_requests(lengths, pool_tokens: int, max_len: int | None = None, block_size: int | None = None) -> int:
    """Nombre de requêtes (prises dans l'ordre) qui tiennent dans un réservoir de pool_tokens
    positions : réservation de max_len par requête, ou blocs de block_size (paginé)."""
    total = 0
    for i, n in enumerate(lengths):
        need = max_len if max_len else int(math.ceil(n / block_size) * block_size)
        if total + need > pool_tokens:
            return i
        total += need
    return len(lengths)


def paged_layout(lengths, block_size: int, n_blocks: int, seed: int = 0):
    """Disposition de blocs physiques façon PagedAttention : chaque requête reçoit des blocs libres
    quelconques (non contigus). Renvoie (owner[n_blocks], fill[n_blocks], table des blocs)."""
    rng = np.random.default_rng(seed)
    free = list(rng.permutation(n_blocks))
    owner = -np.ones(n_blocks, dtype=int)
    fill = np.zeros(n_blocks)
    tables = {}
    for r, n in enumerate(lengths):
        nb = int(math.ceil(n / block_size))
        blocks = [free.pop() for _ in range(nb)]
        tables[r] = blocks
        for i, b in enumerate(blocks):
            owner[b] = r
            fill[b] = min(block_size, n - i * block_size) / block_size
    return owner, fill, tables


# =====================================================================================
# 6. Service : simulation du batching
# =====================================================================================


@dataclass
class StepCost:
    """Modèle de coût d'un serveur (secondes) : un pas de decode pour b requêtes coûte
    dec0 + dec1 b, le prefill de n tokens coûte pre0 + pre1 n."""

    dec0: float = 0.020
    dec1: float = 0.0004
    pre0: float = 0.005
    pre1: float = 0.00005

    def decode(self, b: int) -> float:
        return self.dec0 + self.dec1 * b if b > 0 else 0.0

    def prefill(self, n_tokens: int) -> float:
        return self.pre0 + self.pre1 * n_tokens if n_tokens > 0 else 0.0


def make_workload(n: int, rate: float, seed: int = 0, prompt_mean: int = 200, out_median: int = 60,
                  out_sigma: float = 0.9, out_max: int = 600) -> pd.DataFrame:
    """Charge synthétique : arrivées de Poisson (rate requêtes/s), prompts ~ géométriques,
    sorties ~ log-normales (beaucoup de réponses courtes, quelques très longues)."""
    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(1 / rate, n))
    prompts = rng.geometric(1 / prompt_mean, n).clip(8, 4 * prompt_mean)
    outs = np.exp(rng.normal(np.log(out_median), out_sigma, n)).astype(int).clip(4, out_max)
    return pd.DataFrame({"arrivee": arrivals, "prompt": prompts, "sortie": outs})


def simulate_batching(work: pd.DataFrame, policy: str = "continuous", max_batch: int = 8,
                      window: float = 0.1, cost: StepCost | None = None) -> pd.DataFrame:
    """Simulation à événements discrets d'un serveur (un GPU) servant `work`.

    policy :
      - "none" : une requête à la fois ;
      - "static" : on attend max_batch requêtes, puis prefill + decode jusqu'à la fin de la plus
        longue ; les réponses sont rendues à la fin du lot ;
      - "dynamic" : idem, mais le lot part quand il est plein **ou** quand la plus ancienne
        requête attend depuis `window` secondes ;
      - "continuous" : batching au niveau de l'itération (Orca) : à chaque pas, les requêtes
        terminées sortent et les requêtes en attente entrent (prefill fusionné au pas courant).

    Renvoie une ligne par requête : début, premier token, fin, TTFT, latence, TPOT.
    """
    cost = cost or StepCost()
    arr = work["arrivee"].to_numpy()
    P = work["prompt"].to_numpy()
    O = work["sortie"].to_numpy()
    n = len(arr)
    start = np.full(n, np.nan)
    first = np.full(n, np.nan)
    finish = np.full(n, np.nan)
    t = 0.0
    nxt = 0  # prochaine requête non encore admise
    if policy in ("none", "static", "dynamic"):
        B = 1 if policy == "none" else max_batch
        while nxt < n:
            if policy == "static":
                last = min(nxt + B, n) - 1
                t = max(t, arr[last])
            elif policy == "dynamic":
                t = max(t, arr[nxt])
                full_at = arr[min(nxt + B, n) - 1]
                t = max(t, min(full_at, arr[nxt] + window)) if nxt + B <= n else max(t, arr[nxt] + window)
            else:
                t = max(t, arr[nxt])
            batch = [i for i in range(nxt, min(nxt + B, n)) if arr[i] <= t + 1e-12]
            nxt = batch[-1] + 1
            start[batch] = t
            t += cost.prefill(int(P[batch].sum()))
            first[batch] = t  # le prefill produit le premier token
            steps = int(O[batch].max()) - 1
            # lot naïf : toutes les requêtes restent dans le lot jusqu'à la fin
            t += steps * cost.decode(len(batch))
            finish[batch] = t
    elif policy == "continuous":
        active = {}  # id -> tokens restant à générer
        while nxt < n or active:
            if not active and nxt < n:
                t = max(t, arr[nxt])
            new = []
            while nxt < n and arr[nxt] <= t + 1e-12 and len(active) + len(new) < max_batch:
                new.append(nxt)
                nxt += 1
            dt = cost.decode(len(active)) + cost.prefill(int(P[new].sum()) if new else 0)
            t += dt
            for i in list(active):
                active[i] -= 1
                if active[i] <= 0:
                    finish[i] = t
                    del active[i]
            for i in new:
                start[i] = t - dt
                first[i] = t
                if O[i] <= 1:
                    finish[i] = t
                else:
                    active[i] = O[i] - 1
    else:
        raise ValueError(policy)
    df = work.copy()
    df["debut"], df["premier_token"], df["fin"] = start, first, finish
    df["ttft"] = first - arr
    df["latence"] = finish - arr
    df["tpot"] = np.where(O > 1, (finish - first) / np.maximum(O - 1, 1), 0.0)
    return df


def batching_summary(df: pd.DataFrame) -> dict:
    makespan = df["fin"].max() - df["arrivee"].min()
    return {"débit (tokens/s)": df["sortie"].sum() / makespan,
            "requêtes/s": len(df) / makespan,
            "latence p50 (s)": df["latence"].median(), "latence p90 (s)": df["latence"].quantile(0.9),
            "TTFT p50 (s)": df["ttft"].median(), "TTFT p90 (s)": df["ttft"].quantile(0.9),
            "TPOT p50 (ms)": 1e3 * df["tpot"].median()}


def simulate_colocated_vs_disaggregated(work: pd.DataFrame, cost: StepCost, max_batch: int = 32,
                                        transfer_s: float = 0.002) -> dict:
    """Compare, à 2 GPU, deux répliques colocalisées (prefill et decode sur la même machine, en
    batching continu) et une architecture dissociée (1 GPU de prefill + 1 GPU de decode).

    Renvoie pour chaque architecture la liste des TTFT et le plus grand écart entre deux tokens
    (max ITL) de chaque requête : en colocalisé, un prefill inséré bloque les decodes en cours.
    """
    out = {}
    # --- colocalisé : 2 répliques, requêtes réparties en alternance
    ttft, itl = [], []
    for r in range(2):
        sub = work.iloc[r::2].reset_index(drop=True)
        ttft_r, itl_r = _continuous_with_itl(sub, cost, max_batch)
        ttft += ttft_r
        itl += itl_r
    out["colocalisé (2 répliques)"] = (np.array(ttft), np.array(itl))
    # --- dissocié : prefill FIFO sur le GPU 1, decode continu sur le GPU 2
    arr = work["arrivee"].to_numpy()
    P = work["prompt"].to_numpy()
    O = work["sortie"].to_numpy()
    t = 0.0
    ready = np.zeros(len(arr))
    for i in range(len(arr)):  # le GPU de prefill traite les requêtes une par une
        t = max(t, arr[i]) + cost.prefill(int(P[i]))
        ready[i] = t + transfer_s
    ttft_d = ready - arr
    dec_work = pd.DataFrame({"arrivee": ready, "prompt": np.zeros(len(arr), int), "sortie": O})
    order = np.argsort(ready, kind="stable")
    dec_work = dec_work.iloc[order].reset_index(drop=True)
    _, itl_d = _continuous_with_itl(dec_work, StepCost(cost.dec0, cost.dec1, 0.0, 0.0), max_batch)
    out["dissocié (1 prefill + 1 decode)"] = (ttft_d, np.array(itl_d))
    return out


def _continuous_with_itl(work, cost: StepCost, max_batch: int):
    arr = work["arrivee"].to_numpy()
    P = work["prompt"].to_numpy()
    O = work["sortie"].to_numpy()
    n = len(arr)
    t, nxt = 0.0, 0
    active, last_tok, max_gap, ttft = {}, {}, {}, {}
    while nxt < n or active:
        if not active and nxt < n:
            t = max(t, arr[nxt])
        new = []
        while nxt < n and arr[nxt] <= t + 1e-12 and len(active) + len(new) < max_batch:
            new.append(nxt)
            nxt += 1
        t += cost.decode(len(active)) + cost.prefill(int(P[new].sum()) if new else 0)
        for i in list(active):
            max_gap[i] = max(max_gap.get(i, 0.0), t - last_tok[i])
            last_tok[i] = t
            active[i] -= 1
            if active[i] <= 0:
                del active[i]
        for i in new:
            ttft[i] = t - arr[i]
            last_tok[i] = t
            if O[i] > 1:
                active[i] = O[i] - 1
    return [ttft[i] for i in range(n)], [max_gap.get(i, 0.0) for i in range(n)]


# =====================================================================================
# 7. Cache de prompt et parallélisme
# =====================================================================================


def prompt_cache_savings(n_requests: int, cached_tokens: int, fresh_tokens: int, output_tokens: int,
                         price_in: float, price_out: float, cache_read_factor: float = 0.1,
                         cache_write_factor: float = 1.25, n_writes: int = 1) -> dict:
    """Coût (dollars, prix par million de tokens) avec et sans cache de prompt.

    Sans cache : chaque requête paie cached + fresh tokens d'entrée au prix normal.
    Avec cache : n_writes écritures du préfixe (majorées de cache_write_factor), puis lectures
    à cache_read_factor du prix normal (valeurs proches de la tarification d'Anthropic).
    """
    per_m = 1e-6
    no_cache = n_requests * ((cached_tokens + fresh_tokens) * price_in + output_tokens * price_out) * per_m
    with_cache = (n_writes * cached_tokens * price_in * cache_write_factor
                  + (n_requests - n_writes) * cached_tokens * price_in * cache_read_factor
                  + n_requests * (fresh_tokens * price_in + output_tokens * price_out)) * per_m
    return {"sans cache ($)": no_cache, "avec cache ($)": with_cache, "économie": 1 - with_cache / no_cache}


def tensor_parallel_matmul(x, W, n_devices: int, mode: str = "column"):
    """Produit x @ W découpé sur n_devices « appareils » (simulés).

    mode="column" : chaque appareil a un bloc de colonnes de W et calcule un bloc de colonnes de
    la sortie (concaténation, all-gather) ; mode="row" : chaque appareil a un bloc de lignes de W
    et la tranche correspondante de x ; les sorties partielles sont additionnées (all-reduce).
    """
    x, W = np.asarray(x), np.asarray(W)
    if mode == "column":
        parts = [x @ Wi for Wi in np.array_split(W, n_devices, axis=1)]
        return np.concatenate(parts, axis=-1), parts
    xs = np.array_split(x, n_devices, axis=-1)
    Ws = np.array_split(W, n_devices, axis=0)
    parts = [xi @ Wi for xi, Wi in zip(xs, Ws)]
    return sum(parts), parts


def pipeline_bubble_fraction(n_stages: int, n_microbatches: int) -> float:
    """Fraction de temps où les étages d'un pipeline (ordonnancement GPipe) sont inactifs :
    (p - 1) / (m + p - 1)."""
    return (n_stages - 1) / (n_microbatches + n_stages - 1)


def pipeline_schedule(n_stages: int, n_microbatches: int) -> np.ndarray:
    """Grille (étages x pas de temps) : numéro du micro-lot traité, -1 si l'étage est inactif."""
    T = n_microbatches + n_stages - 1
    g = -np.ones((n_stages, T), dtype=int)
    for s in range(n_stages):
        for mb in range(n_microbatches):
            g[s, s + mb] = mb
    return g


def first_fit_decreasing(items: dict, capacities: list) -> tuple[dict, list]:
    """Rangement glouton (bin packing) : place les éléments (nom -> taille), du plus grand au plus
    petit, sur le premier GPU (capacités) où ils tiennent. Renvoie (placement, mémoire libre)."""
    free = list(capacities)
    placement = {}
    for name, size in sorted(items.items(), key=lambda kv: -kv[1]):
        for g, f in enumerate(free):
            if size <= f + 1e-9:
                placement[name] = g
                free[g] -= size
                break
        else:
            placement[name] = None
    return placement, free


def serving_memory_gb(n_params: float, bytes_per: float, kv_bytes_per_token: float, context: int,
                      concurrent: int, overhead: float = 0.1) -> float:
    """Mémoire d'un service : poids + caches KV des requêtes simultanées, + marge (activations, etc.)."""
    return (n_params * bytes_per + kv_bytes_per_token * context * concurrent) * (1 + overhead) / 1e9


#: Texte d'évaluation (rédigé pour ce dépôt) pour mesurer la perplexité des variantes d'un modèle.
EVAL_TEXT = (
    "Inference optimization is the art of making a trained model cheaper and faster to run without "
    "changing what it computes, or while changing it as little as possible. A language model generates "
    "text one token at a time: for every new token, the accelerator must read all the weights of the "
    "network from its memory, multiply them by a single vector, and write a few numbers back. Because so "
    "little arithmetic is done for every byte that is read, the speed of generation is set by the memory "
    "bandwidth rather than by the raw computing power of the chip. This simple observation explains most "
    "of the techniques used in production. Quantization stores each weight with fewer bits, so that fewer "
    "bytes have to travel for each token. Batching lets many users share the same reading of the weights. "
    "Speculative decoding asks a small model to guess several tokens ahead, and lets the large model check "
    "all the guesses in one pass. Caching avoids computing the same keys and values twice, whether they "
    "belong to the previous tokens of a conversation or to a system prompt shared by every request. "
    "Finally, when one chip is not enough, the model can be split across several devices, at the price of "
    "extra communication between them."
)

#: Texte de calibration (distinct du texte d'évaluation) pour les critères d'élagage.
CALIB_TEXT = (
    "Large language models are trained on trillions of tokens and then deployed to answer questions, "
    "write code, translate documents and summarize long reports. Serving them is expensive: a single "
    "graphics card can cost several dollars per hour, and a popular application may receive millions of "
    "requests every day. Engineers therefore measure the time to the first token, the time between two "
    "tokens and the number of tokens produced per second, and they look for the cheapest hardware that "
    "still satisfies their latency objectives. The weather was mild in Paris yesterday, and the museums "
    "were crowded with tourists who had come to see the new exhibition about the history of computing, "
    "from mechanical calculators to modern data centers filled with accelerators."
)

#: Petit morceau de code utilisé pour les démonstrations d'inférence avec référence.
CODE_SNIPPET = '''def moving_average(values, window):
    """Return the moving average of a list of numbers."""
    if window <= 0:
        raise ValueError("window must be positive")
    result = []
    total = 0.0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
        if i >= window - 1:
            result.append(total / window)
    return result
'''
