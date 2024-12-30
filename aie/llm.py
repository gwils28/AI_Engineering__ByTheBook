"""Accès unifié aux petits modèles de langage locaux (Hugging Face Transformers).

Tous les notebooks utilisent des modèles ouverts qui tiennent sur un GPU grand public
(ou, plus lentement, sur CPU) : aucune clé d'API n'est nécessaire.

Modèles par défaut
------------------
- ``SMALL`` : Qwen2.5-0.5B-Instruct (≈ 0,5 milliard de paramètres, rapide) ;
- ``MEDIUM`` : Qwen2.5-1.5B-Instruct (meilleure qualité, ≈ 3 Go en bf16) ;
- ``BASE`` : GPT-2 (124 M paramètres, modèle « de base » non aligné, utile pour comparer) ;
- ``EMBED`` : all-MiniLM-L6-v2 (embeddings de phrases, 384 dimensions) ;
- ``EMBED_MULTI`` : paraphrase-multilingual-MiniLM-L12-v2 (embeddings multilingues, dont le français).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

SMALL = "Qwen/Qwen2.5-0.5B-Instruct"
MEDIUM = "Qwen/Qwen2.5-1.5B-Instruct"
BASE = "gpt2"
EMBED = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MULTI = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def softmax(z, temperature: float = 1.0) -> np.ndarray:
    """Softmax numériquement stable de logits `z` à la température donnée."""
    z = np.asarray(z, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int = 0) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@functools.lru_cache(maxsize=4)
def load(model_id: str = SMALL, dtype: str = "bfloat16"):
    """Charge (une seule fois) un modèle causal et son tokenizer. Renvoie (tokenizer, modèle)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch_dtype = getattr(torch, dtype) if device() == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype).to(device())
    model.eval()
    return tok, model


@functools.lru_cache(maxsize=2)
def load_embedder(model_id: str = EMBED):
    """Charge un modèle d'embeddings sentence-transformers (mis en cache)."""
    from sentence_transformers import SentenceTransformer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    return SentenceTransformer(model_id, device=device())


def embed(texts, model_id: str = EMBED, normalize: bool = True) -> np.ndarray:
    """Embeddings (une ligne par texte), normalisés en norme 2 par défaut."""
    if isinstance(texts, str):
        texts = [texts]
    return load_embedder(model_id).encode(list(texts), normalize_embeddings=normalize,
                                          show_progress_bar=False)


def _chat_inputs(tok, messages, add_generation_prompt: bool = True):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    return tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt,
                                   return_tensors="pt", return_dict=True)


def chat(messages, model_id: str = SMALL, max_new_tokens: int = 200, temperature: float = 0.0,
         top_p: float = 1.0, system: str | None = None, seed: int | None = 0, **gen_kw) -> str:
    """Génère une réponse à une conversation.

    messages : une chaîne (message utilisateur) ou une liste de dicts {"role", "content"}.
    temperature = 0 : décodage glouton (déterministe).
    """
    import torch

    tok, model = load(model_id)
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if system is not None:
        messages = [{"role": "system", "content": system}] + list(messages)
    inputs = _chat_inputs(tok, messages).to(model.device)
    if seed is not None:
        seed_everything(seed)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if temperature and temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    kw.update(gen_kw)
    with torch.no_grad():
        out = model.generate(**inputs, **kw)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def complete(prompt: str, model_id: str = BASE, max_new_tokens: int = 50, temperature: float = 0.0,
             seed: int | None = 0, **gen_kw) -> str:
    """Complétion brute d'un texte (sans gabarit de conversation), typique d'un modèle de base."""
    import torch

    tok, model = load(model_id)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    if seed is not None:
        seed_everything(seed)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if temperature and temperature > 0:
        kw.update(do_sample=True, temperature=temperature)
    else:
        kw.update(do_sample=False)
    kw.update(gen_kw)
    with torch.no_grad():
        out = model.generate(**inputs, **kw)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


@dataclass
class NextToken:
    tokens: list[str]
    probs: np.ndarray
    logits: np.ndarray


def next_token_distribution(text: str, model_id: str = BASE, k: int = 10, chat_format: bool = False) -> NextToken:
    """Les k tokens les plus probables après `text`, avec probabilités et logits."""
    import torch

    tok, model = load(model_id)
    if chat_format:
        inputs = _chat_inputs(tok, text).to(model.device)
    else:
        inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1].float()
    probs = torch.softmax(logits, -1)
    top = torch.topk(probs, k)
    ids = top.indices.tolist()
    return NextToken([tok.decode([i]) for i in ids], top.values.cpu().numpy(), logits[ids].cpu().numpy())


def full_logits(text: str, model_id: str = BASE) -> np.ndarray:
    """Logits complets (taille du vocabulaire) du prochain token après `text`."""
    import torch

    tok, model = load(model_id)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        return model(**inputs).logits[0, -1].float().cpu().numpy()


def token_logprobs(text: str, model_id: str = BASE) -> tuple[list[str], np.ndarray]:
    """Log-probabilité (en nats) de chaque token de `text` sachant les précédents.

    Le premier token n'a pas de contexte : il est omis. Renvoie (tokens, logprobs).
    """
    import torch

    tok, model = load(model_id)
    ids = tok(text, return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, :-1].float()
    lp = torch.log_softmax(logits, -1)
    target = ids[0, 1:]
    out = lp[torch.arange(len(target)), target].cpu().numpy()
    return [tok.decode([i]) for i in target.tolist()], out


def perplexity(text: str, model_id: str = BASE) -> float:
    """Perplexité du modèle sur `text` : exp(- moyenne des log-probabilités)."""
    _, lp = token_logprobs(text, model_id)
    return float(math.exp(-lp.mean()))


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def free_gpu() -> None:
    """Vide les caches de modèles et la mémoire GPU."""
    import gc

    import torch

    load.cache_clear()
    load_embedder.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
