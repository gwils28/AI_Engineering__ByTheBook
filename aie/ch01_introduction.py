"""Chapitre 1 : introduction à l'ingénierie des applications de modèles de fondation.

Fonctions réutilisables des notebooks du chapitre 1 :

- **modèles de langage statistiques** : fréquences de lettres, entropie, modèle n-gramme de
  caractères (dans l'esprit de Shannon, 1951) ;
- **tokenisation** : interface unifiée sur plusieurs tokenizers (tiktoken et Hugging Face),
  comparaison du coût en tokens selon la langue, mini-BPE pédagogique, affichage coloré ;
- **auto-supervision** : transformation d'un texte en paires (contexte, cible), version
  autorégressive et version masquée ;
- **échelle** : tailles de modèles publiques, nombre de paramètres d'un transformer, règle
  « Chinchilla », coût d'entraînement en FLOP ;
- **planification d'une application** : coût en tokens, latence (TTFT, TPOT), grille de
  décision, courbe du « dernier kilomètre », intervalle de Wilson, seuil de rentabilité
  construire / acheter.

Toutes les fonctions sans modèle sont pures et testées dans ``tests/test_ch01_introduction.py``.
"""

from __future__ import annotations

import functools
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

# =====================================================================================
# 1. Statistiques du langage
# =====================================================================================

FRENCH_ALPHABET = "abcdefghijklmnopqrstuvwxyzàâçéèêëîïôùûüœ"

#: petit corpus français rédigé pour le dépôt (libre de droits), utilisé pour les n-grammes
CORPUS_FR = """\
Le matin, la ville se réveille lentement. Les boulangers ont allumé leurs fours bien avant
l'aube et l'odeur du pain chaud remonte la rue principale. Les premiers cyclistes passent
devant le marché, où les maraîchers installent leurs étals de tomates, de salades et de
fraises. Une petite fille tient la main de son père et lui demande pourquoi le ciel est bleu.
Il réfléchit un instant, puis lui explique que la lumière du soleil se disperse dans l'air,
et que le bleu se disperse davantage que le rouge. Elle hoche la tête, pas tout à fait
convaincue, et réclame une chocolatine.

Dans le bureau d'à côté, une équipe d'ingénieurs prépare une démonstration. Ils ont branché
un modèle de langage à la base de connaissances de l'entreprise et veulent montrer qu'il
répond correctement aux questions des clients. La première réponse est parfaite, la deuxième
aussi. La troisième invente une politique de remboursement qui n'existe pas. Le chef de
projet sourit nerveusement : il faudra évaluer le système bien plus sérieusement avant de le
mettre entre les mains du public. Construire une démonstration prend un week-end ; construire
un produit fiable prend des mois.

Le modèle, lui, ne sait rien du monde au sens où nous l'entendons. Il a appris à prédire le
mot suivant dans des milliards de phrases, et cette tâche simple, répétée à une échelle
immense, l'a forcé à capturer la grammaire, les faits, les styles et même un peu de
raisonnement. Quand on lui donne le début d'une phrase, il calcule une probabilité pour
chaque suite possible, choisit un morceau de mot, l'ajoute au texte et recommence. C'est
une machine à compléter, rien de plus, et pourtant presque toutes les tâches peuvent être
formulées comme une complétion : traduire, résumer, classer, écrire du code ou répondre à
une question.

Le soir venu, la petite fille revient du marché avec son père. Elle a mangé sa chocolatine
et pose maintenant une nouvelle question : comment l'ordinateur peut-il parler s'il n'a pas
de bouche ? Le père rit. Il lui dit que l'ordinateur ne parle pas vraiment, qu'il assemble
des morceaux de mots comme on assemble des briques, en choisissant à chaque fois la brique
la plus probable. Elle trouve l'idée amusante et décide qu'elle construira, elle aussi, une
machine qui parle. Elle commencera demain, après l'école, avec des briques de toutes les
couleurs.

Pendant ce temps, les ingénieurs ont repris leur travail. Ils écrivent des tests, collectent
des exemples de questions difficiles, mesurent la latence et le coût de chaque requête. Ils
découvrent que passer de soixante à cent pour cent de réussite est beaucoup plus difficile
que passer de zéro à soixante. Chaque point gagné demande davantage d'efforts que le
précédent. Mais ils savent désormais ce qu'ils mesurent, et c'est déjà une grande victoire.
"""


def normalize_text(text: str, alphabet: str = FRENCH_ALPHABET) -> str:
    """Minuscules, espaces simplifiés, caractères hors `alphabet` (et hors espace) supprimés."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    keep = set(alphabet) | {" "}
    return "".join(c for c in text if c in keep).strip()


def letter_frequencies(text: str, alphabet: str = FRENCH_ALPHABET) -> dict[str, float]:
    """Fréquence relative de chaque lettre de `alphabet` dans `text` (espaces exclus)."""
    letters = [c for c in text.lower() if c in alphabet]
    n = len(letters)
    if n == 0:
        return {}
    counts = Counter(letters)
    return {c: counts[c] / n for c in sorted(counts, key=lambda c: -counts[c])}


def entropy(p, base: float = 2.0) -> float:
    """Entropie de Shannon H(p) = - somme p_i log p_i (termes nuls ignorés), en bits par défaut."""
    p = np.asarray(list(p.values()) if isinstance(p, dict) else p, dtype=float)
    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / math.log(base))


class CharNGram:
    """Modèle de langage n-gramme sur les caractères (lissage additif de Laplace).

    ``order = n`` : la probabilité d'un caractère dépend des ``n - 1`` caractères précédents.
    ``order = 1`` correspond aux simples fréquences de lettres (aucun contexte).
    """

    def __init__(self, order: int = 3, alpha: float = 0.01):
        if order < 1:
            raise ValueError("order >= 1")
        self.order, self.alpha = order, alpha
        self.counts: dict[str, Counter] = defaultdict(Counter)
        self.vocab: list[str] = []

    def fit(self, text: str) -> "CharNGram":
        self.vocab = sorted(set(text))
        k = self.order - 1
        padded = "\x02" * k + text
        for i in range(k, len(padded)):
            self.counts[padded[i - k:i]][padded[i]] += 1
        return self

    def distribution(self, context: str) -> dict[str, float]:
        """P(caractère | contexte) sur tout le vocabulaire (contexte tronqué à order - 1).

        Si le contexte n'a jamais été vu, on se replie sur ses suffixes plus courts.
        """
        k = self.order - 1
        ctx = ("\x02" * k + context)[-k:] if k else ""
        c = self.counts.get(ctx, Counter())
        while not c and ctx:  # repli (« backoff ») vers un contexte plus court si inconnu
            ctx = ctx[1:]
            c = Counter()
            for full, cnt in self.counts.items():
                if full.endswith(ctx):
                    c.update(cnt)
        total = sum(c.values()) + self.alpha * len(self.vocab)
        return {ch: (c[ch] + self.alpha) / total for ch in self.vocab}

    def prob(self, char: str, context: str) -> float:
        return self.distribution(context).get(char, 0.0)

    def generate(self, n: int, prompt: str = "", seed: int = 0, temperature: float = 1.0) -> str:
        rng = np.random.default_rng(seed)
        out = prompt
        for _ in range(n):
            d = self.distribution(out)
            chars = list(d)
            p = np.array([d[c] for c in chars]) ** (1.0 / temperature)
            out += chars[rng.choice(len(chars), p=p / p.sum())]
        return out[len(prompt):]

    def cross_entropy(self, text: str) -> float:
        """Entropie croisée moyenne (bits par caractère) du modèle sur `text`."""
        k = self.order - 1
        padded = "\x02" * k + text
        lp = [math.log2(max(self.prob(padded[i], padded[max(0, i - k):i]), 1e-12))
              for i in range(k, len(padded))]
        return -float(np.mean(lp))


def conditional_entropy(text: str, order: int) -> float:
    """Entropie conditionnelle empirique H(X_n | X_{n-k..n-1}) en bits, k = order - 1.

    Estimation « plug-in » sur le texte lui-même (biaisée vers le bas quand l'ordre grandit,
    car le modèle finit par mémoriser le corpus).
    """
    k = order - 1
    ctx_counts: dict[str, Counter] = defaultdict(Counter)
    for i in range(k, len(text)):
        ctx_counts[text[i - k:i]][text[i]] += 1
    n = sum(sum(c.values()) for c in ctx_counts.values())
    h = 0.0
    for c in ctx_counts.values():
        tot = sum(c.values())
        h += tot / n * entropy(list(c.values()))
    return h


def setup_style() -> None:
    """`set_style()` du socle + repli de police (flèches, symboles) vers DejaVu Sans."""
    import matplotlib as mpl

    from .plotting import set_style

    set_style()
    mpl.rcParams["font.family"] = ["Noto Sans", "DejaVu Sans"]


def plain_log(ax, which: str = "y") -> None:
    """Graduations lisibles (0,001 ; 10 ; 1 000...) sur un axe logarithmique, sans mathtext."""
    from matplotlib.ticker import FuncFormatter, LogLocator

    def fmt(v, _):
        if v <= 0:
            return ""
        if v >= 1:
            return f"{v:,.0f}".replace(",", " ")
        return f"{v:.10f}".rstrip("0").replace(".", ",")

    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_locator(LogLocator(base=10))
    axis.set_major_formatter(FuncFormatter(fmt))
    axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def human_number(v: float) -> str:
    """1 234 -> « 1 k », 2e9 -> « 2 Md », 3e12 -> « 3 000 Md » (unités françaises : k, M, Md)."""
    for div, unit in [(1e9, " Md"), (1e6, " M"), (1e3, " k")]:
        if abs(v) >= div:
            x = v / div
            if x >= 10 or x == int(x):
                return f"{x:,.0f}".replace(",", " ") + unit
            return f"{x:.1f}".replace(".", ",") + unit
    return f"{v:g}".replace(".", ",")


def human_log(ax, which: str = "y") -> None:
    """Axe logarithmique gradué en unités lisibles (k, M, Md) au lieu de longues suites de zéros."""
    from matplotlib.ticker import FuncFormatter, LogLocator

    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_locator(LogLocator(base=10))
    axis.set_major_formatter(FuncFormatter(lambda v, _: human_number(v) if v > 0 else ""))
    axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def title_sub(ax, title: str, sub: str | None = None, pad: float = 20) -> None:
    """Titre d'axe + sous-titre gris sans chevauchement (le titre est remonté de `pad` points)."""
    from .plotting import GRAY

    ax.set_title(title, pad=pad if sub else 6)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9, color=GRAY, va="bottom", ha="left")


# =====================================================================================
# 2. Tokenisation
# =====================================================================================

@dataclass
class Tokenizer:
    """Interface commune à un tokenizer tiktoken ou Hugging Face."""

    name: str
    backend: str  # "tiktoken" ou "hf"
    ident: str
    _tok: object = field(default=None, repr=False)

    def _load(self):
        if self._tok is None:
            if self.backend == "tiktoken":
                import tiktoken

                self._tok = tiktoken.get_encoding(self.ident)
            else:
                from transformers import AutoTokenizer
                from transformers.utils import logging as hf_logging

                hf_logging.set_verbosity_error()
                self._tok = AutoTokenizer.from_pretrained(self.ident)
        return self._tok

    @property
    def vocab_size(self) -> int:
        t = self._load()
        return t.n_vocab if self.backend == "tiktoken" else len(t)

    def encode(self, text: str) -> list[int]:
        t = self._load()
        if self.backend == "tiktoken":
            return t.encode(text, disallowed_special=())
        return t.encode(text, add_special_tokens=False)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def token_bytes(self, text: str) -> list[bytes]:
        """Octets UTF-8 couverts par chaque token (un token peut couvrir un bout de caractère)."""
        t = self._load()
        ids = self.encode(text)
        if self.backend == "tiktoken":
            return [t.decode_single_token_bytes(i) for i in ids]
        pieces = t.convert_ids_to_tokens(ids)
        if any("▁" in p for p in pieces):  # SentencePiece : « ▁ » marque une espace
            return [p.replace("▁", " ").encode("utf-8") for p in pieces]
        bd = _gpt2_byte_decoder()  # BPE au niveau des octets (GPT-2, Qwen...)
        return [bytes(bd[ch] for ch in p) if all(ch in bd for ch in p) else p.encode("utf-8")
                for p in pieces]

    def pieces(self, text: str) -> list[str]:
        """Texte lisible de chaque token (un fragment de caractère multi-octets s'affiche « � »)."""
        return [b.decode("utf-8", errors="replace") for b in self.token_bytes(text)]


@functools.lru_cache(maxsize=1)
def _gpt2_byte_decoder() -> dict[str, int]:
    """Table inverse de l'encodage « octets -> caractères imprimables » des BPE de type GPT-2."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


#: tokenizers comparés dans le chapitre (nom affiché -> (backend, identifiant))
TOKENIZERS = {
    "GPT-2": ("tiktoken", "gpt2"),
    "GPT-4 (cl100k)": ("tiktoken", "cl100k_base"),
    "GPT-4o (o200k)": ("tiktoken", "o200k_base"),
    "Qwen2.5": ("hf", "Qwen/Qwen2.5-0.5B-Instruct"),
    "XLM-R": ("hf", "FacebookAI/xlm-roberta-base"),
}


@functools.lru_cache(maxsize=16)
def get_tokenizer(name: str) -> Tokenizer:
    """Tokenizer du registre `TOKENIZERS` (mis en cache)."""
    backend, ident = TOKENIZERS[name]
    return Tokenizer(name, backend, ident)


#: une même phrase dans plusieurs langues (traductions rédigées pour le dépôt)
PARALLEL = {
    "anglais": "The weather is nice today, so let's take a walk in the park after lunch.",
    "français": "Il fait beau aujourd'hui, alors allons nous promener au parc après le déjeuner.",
    "espagnol": "Hoy hace buen tiempo, así que vamos a pasear por el parque después de comer.",
    "allemand": "Heute ist schönes Wetter, also lass uns nach dem Mittagessen im Park spazieren gehen.",
    "chinois": "今天天气很好，所以午饭后我们去公园散步吧。",
    "arabe": "الطقس جميل اليوم، فلنتمشى في الحديقة بعد الغداء.",
    "hindi": "आज मौसम अच्छा है, तो चलो दोपहर के खाने के बाद पार्क में टहलने चलते हैं।",
}

#: petit corpus parallèle anglais / français (phrases rédigées pour le dépôt)
PARALLEL_EN_FR = [
    ("I can't wait to build AI applications.", "J'ai hâte de construire des applications d'IA."),
    ("The model predicts the next token in the sequence.", "Le modèle prédit le prochain token de la séquence."),
    ("Please summarize this meeting in three bullet points.", "Merci de résumer cette réunion en trois points."),
    ("Our customer support chatbot answers questions about refunds.",
     "Notre chatbot de support client répond aux questions sur les remboursements."),
    ("Evaluation is harder when outputs are open-ended.",
     "L'évaluation est plus difficile lorsque les sorties sont ouvertes."),
    ("The invoice was sent yesterday and should arrive tomorrow.",
     "La facture a été envoyée hier et devrait arriver demain."),
    ("Foundation models can process text, images and audio.",
     "Les modèles de fondation peuvent traiter du texte, des images et de l'audio."),
    ("We need to reduce latency below one second.", "Nous devons réduire la latence sous une seconde."),
]


def words(text: str) -> list[str]:
    """Découpage naïf en mots (suites de lettres, chiffres ou apostrophes)."""
    return re.findall(r"[\w']+", text)


def tokens_per_word(text: str, tok: Tokenizer) -> float:
    """Nombre moyen de tokens par mot (la « fertilité » du tokenizer sur ce texte)."""
    return tok.count(text) / max(1, len(words(text)))


def compare_tokenizers(texts: dict[str, str], names=None):
    """Tableau (pandas) : nombre de tokens de chaque texte pour chaque tokenizer."""
    import pandas as pd

    names = list(names or TOKENIZERS)
    rows = {k: {n: get_tokenizer(n).count(t) for n in names} for k, t in texts.items()}
    return pd.DataFrame(rows).T[names]


# ---- mini-BPE pédagogique -----------------------------------------------------------

def _pair_counts(word_freqs: dict[tuple[str, ...], int]) -> Counter:
    pairs: Counter = Counter()
    for symbols, f in word_freqs.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += f
    return pairs


def _merge(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    out, i = [], 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train_bpe(corpus, n_merges: int, end: str = "_") -> list[tuple[str, str]]:
    """Apprend `n_merges` fusions BPE (Sennrich et al., 2016) sur une liste de mots.

    Chaque mot est vu comme une suite de caractères terminée par `end` (marque de fin de
    mot). À chaque étape, la paire de symboles adjacents la plus fréquente est fusionnée
    (en cas d'égalité : ordre lexicographique, pour un résultat déterministe).
    """
    if isinstance(corpus, str):
        corpus = words(corpus.lower())
    word_freqs: dict[tuple[str, ...], int] = Counter(tuple(w) + (end,) for w in corpus)
    merges = []
    for _ in range(n_merges):
        pairs = _pair_counts(word_freqs)
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        if pairs[best] < 2:
            break
        merges.append(best)
        word_freqs = Counter({_merge(s, best): f for s, f in word_freqs.items()})
    return merges


def bpe_tokenize(word: str, merges, end: str = "_") -> list[str]:
    """Applique les fusions apprises, dans l'ordre, à un mot (éventuellement inconnu)."""
    symbols = tuple(word) + (end,)
    for pair in merges:
        symbols = _merge(symbols, pair)
    return list(symbols)


def bpe_vocab(corpus, merges, end: str = "_") -> set[str]:
    """Vocabulaire = caractères de base + un symbole par fusion."""
    if isinstance(corpus, str):
        corpus = words(corpus.lower())
    base = {c for w in corpus for c in w} | {end}
    return base | {a + b for a, b in merges}


# ---- affichage coloré des tokens ----------------------------------------------------

def draw_tokens(ax, pieces, y: float = 0.0, x0: float = 0.0, height: float = 0.7,
                fontsize: float = 10, char_w: float = 0.13, colors=None, label: str | None = None,
                max_x: float | None = None, widths=None) -> float:
    """Dessine les tokens comme des pastilles colorées alternées (à la manière de tiktokenizer).

    Les espaces sont rendues par « · » pour les rendre visibles. `widths` impose la largeur
    de chaque pastille (utile pour aligner deux rangées). Renvoie l'abscisse finale.
    """
    from matplotlib.patches import FancyBboxPatch

    from .plotting import AMBER, INDIGO, LIGHT, ROSE, SKY, TEAL, VIOLET

    palette = colors or [INDIGO, TEAL, AMBER, ROSE, VIOLET, SKY]
    x = x0
    if label:
        ax.text(x0 - 0.2, y, label, ha="right", va="center", fontsize=fontsize, fontweight="bold")
    for i, p in enumerate(pieces):
        shown = p.replace(" ", "·").replace("\n", "↵")
        w = widths[i] if widths is not None else max(0.3, char_w * len(shown) + 0.16)
        if max_x is not None and x + w > max_x:
            ax.text(x + 0.05, y, "…", ha="left", va="center", fontsize=fontsize)
            return x
        c = palette[i % len(palette)]
        ax.add_patch(FancyBboxPatch((x, y - height / 2), w, height, boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor=LIGHT.get(c, "#EEE"), edgecolor=c, lw=1.0))
        ax.text(x + w / 2, y, shown, ha="center", va="center", fontsize=fontsize, family="DejaVu Sans Mono")
        x += w + 0.05
    return x


# =====================================================================================
# 3. Auto-supervision
# =====================================================================================

BOS, EOS, MASK = "<BOS>", "<EOS>", "<MASK>"


def autoregressive_pairs(tokens, bos: str | None = BOS, eos: str | None = EOS,
                         max_context: int | None = None) -> list[tuple[list[str], str]]:
    """Paires (contexte, prochain token) d'un modèle autorégressif.

    Une séquence de n tokens donne n paires (n + 1 si l'on ajoute <EOS> en cible) :
    pour chaque position, le contexte est l'ensemble des tokens précédents, tronqué aux
    `max_context` derniers si une fenêtre de contexte est imposée.
    """
    seq = ([bos] if bos else []) + list(tokens) + ([eos] if eos else [])
    pairs = []
    for i in range(1, len(seq)):
        ctx = seq[:i]
        if max_context is not None:
            ctx = ctx[-max_context:]
        pairs.append((ctx, seq[i]))
    return pairs


def masked_pairs(tokens, rate: float = 0.15, seed: int = 0, mask: str = MASK):
    """Exemple d'entraînement d'un modèle masqué (à la BERT, version simplifiée).

    Renvoie (séquence masquée, {position: token à retrouver}). Au moins un token est masqué.
    """
    rng = np.random.default_rng(seed)
    tokens = list(tokens)
    n = len(tokens)
    k = max(1, int(round(rate * n)))
    pos = sorted(rng.choice(n, size=k, replace=False).tolist())
    masked = [mask if i in pos else t for i, t in enumerate(tokens)]
    return masked, {i: tokens[i] for i in pos}


def pairs_table(pairs, sep: str = " "):
    """Tableau pandas lisible (contexte | cible) à partir de paires autorégressives."""
    import pandas as pd

    return pd.DataFrame({"entrée (contexte)": [sep.join(c) for c, _ in pairs],
                         "sortie (prochain token)": [t for _, t in pairs]})


def labeling_cost(n_items: int, cost_per_label: float, annotators: int = 1) -> float:
    """Coût d'étiquetage manuel : n x coût unitaire x nombre d'annotateurs par exemple."""
    return n_items * cost_per_label * annotators


# =====================================================================================
# 4. Échelle des modèles
# =====================================================================================

#: tailles publiques de quelques modèles marquants (paramètres en milliards, tokens
#: d'entraînement en milliards quand ils sont publiés ; None sinon)
MODEL_SIZES = [
    # nom, date (année décimale), paramètres (Md), tokens d'entraînement (Md), organisation
    ("GPT-1", 2018.45, 0.117, None, "OpenAI"),
    ("BERT-large", 2018.79, 0.340, 3.3, "Google"),
    ("GPT-2", 2019.13, 1.5, None, "OpenAI"),
    ("Megatron-LM", 2019.71, 8.3, None, "NVIDIA"),
    ("T5-11B", 2019.81, 11.0, None, "Google"),
    ("GPT-3", 2020.41, 175.0, 300.0, "OpenAI"),
    ("Gopher", 2021.94, 280.0, 300.0, "DeepMind"),
    ("MT-NLG", 2021.78, 530.0, 270.0, "Microsoft/NVIDIA"),
    ("Chinchilla", 2022.24, 70.0, 1400.0, "DeepMind"),
    ("PaLM", 2022.26, 540.0, 780.0, "Google"),
    ("LLaMA-65B", 2023.15, 65.0, 1400.0, "Meta"),
    ("Llama 2 70B", 2023.55, 70.0, 2000.0, "Meta"),
    ("Llama 3.1 405B", 2024.56, 405.0, 15600.0, "Meta"),
    ("Qwen2.5-0.5B", 2024.72, 0.49, 18000.0, "Alibaba"),
]


def model_sizes_frame():
    import pandas as pd

    return pd.DataFrame(MODEL_SIZES, columns=["modèle", "date", "params_Md", "tokens_Md", "organisation"])


def transformer_params(n_layers: int, d_model: int, vocab: int, d_ff: int | None = None,
                       n_ctx: int = 0, tied: bool = True, gated: bool = False,
                       kv_ratio: float = 1.0) -> int:
    """Nombre approximatif de paramètres d'un transformer décodeur.

    Par couche : attention (Q et O de taille d², K et V de taille kv_ratio * d²) + MLP
    (2 * d * d_ff, ou 3 * d * d_ff si le MLP est « gated » comme SwiGLU). On ajoute les
    embeddings (V * d, et autant si la tête de sortie n'est pas partagée) et les positions
    apprises (n_ctx * d). Les normalisations et biais (négligeables) sont omis.
    Avec d_ff = 4d et MHA classique, on retrouve la règle ~ 12 * L * d².
    """
    d_ff = d_ff or 4 * d_model
    attn = (2 + 2 * kv_ratio) * d_model ** 2
    mlp = (3 if gated else 2) * d_model * d_ff
    emb = vocab * d_model * (1 if tied else 2) + n_ctx * d_model
    return int(n_layers * (attn + mlp) + emb)


def chinchilla_tokens(n_params: float, ratio: float = 20.0) -> float:
    """Nombre de tokens « compute-optimal » ≈ 20 x paramètres (Hoffmann et al., 2022)."""
    return ratio * n_params


def training_flops(n_params: float, n_tokens: float) -> float:
    """Coût d'entraînement approché : C ≈ 6 N D opérations en virgule flottante."""
    return 6.0 * n_params * n_tokens


def weights_memory_gb(n_params: float, bytes_per_param: float = 2.0) -> float:
    """Mémoire occupée par les poids seuls (2 octets/paramètre en bf16), en Go (10^9 octets)."""
    return n_params * bytes_per_param / 1e9


# =====================================================================================
# 5. Cas d'usage
# =====================================================================================

#: les huit catégories de cas d'usage du chapitre (reformulées), avec exemples grand
#: public et entreprise
USE_CASES = {
    "Code": (["assistant de code"], ["assistant de code", "migration de code"]),
    "Image et vidéo": (["retouche photo et vidéo", "design"], ["présentations", "publicités"]),
    "Écriture": (["e-mails", "posts et blogs"], ["rédaction marketing, SEO", "rapports, notes"]),
    "Éducation": (["tutorat", "correction de copies"], ["intégration des salariés", "formation continue"]),
    "Agents conversationnels": (["chatbot généraliste", "compagnon IA"], ["support client", "copilotes produit"]),
    "Agrégation d'information": (["résumé", "« parler à ses documents »"], ["résumé", "veille de marché"]),
    "Organisation des données": (["recherche d'images", "mémoire personnelle"], ["gestion des connaissances", "traitement de documents"]),
    "Automatisation de workflows": (["voyages", "événements"], ["extraction et saisie de données", "génération de prospects"]),
}

#: exposition des métiers à l'IA (Eloundou et al., 2023, groupe « Human β », tableau 1-2)
EXPOSURE_BETA = {
    "Enquêteurs (sondages)": 84.4,
    "Écrivains et auteurs": 82.5,
    "Interprètes et traducteurs": 82.4,
    "Relations publiques": 80.6,
    "Zootechniciens": 77.8,
}


# =====================================================================================
# 6. Planification : coûts, latence, décision, dernier kilomètre
# =====================================================================================

def request_cost(n_in: float, n_out: float, price_in_per_m: float, price_out_per_m: float) -> float:
    """Coût d'une requête (dollars) : tokens d'entrée et de sortie facturés au million."""
    return (n_in * price_in_per_m + n_out * price_out_per_m) / 1e6


def monthly_cost(requests_per_day: float, n_in: float, n_out: float, price_in_per_m: float,
                 price_out_per_m: float, days: int = 30) -> float:
    """Coût mensuel (dollars) d'une application à trafic constant."""
    return requests_per_day * days * request_cost(n_in, n_out, price_in_per_m, price_out_per_m)


def conversation_tokens(turns: int, system: int, context: int, question: int, answer: int) -> tuple[int, int]:
    """Tokens (entrée, sortie) facturés pour une conversation de `turns` tours.

    À chaque tour t (1..T), on renvoie le prompt système, le contexte récupéré, tout
    l'historique des t - 1 tours précédents (questions + réponses) et la nouvelle question :
    entrée totale = T (S + C + q) + (q + a) T (T - 1) / 2, sortie totale = T a.
    """
    T = turns
    n_in = T * (system + context + question) + (question + answer) * T * (T - 1) // 2
    return n_in, T * answer


def generation_latency(n_out, ttft: float, tpot: float):
    """Latence totale d'une réponse de `n_out` tokens : TTFT + (n_out - 1) x TPOT (scalaire ou tableau)."""
    out = ttft + np.maximum(0, np.asarray(n_out) - 1) * tpot
    return float(out) if np.ndim(out) == 0 else out


def measure_latency(model_id: str | None = None, prompt: str = "Explique ce qu'est un token.",
                    n_out: int = 64, repeats: int = 3) -> dict[str, float]:
    """Mesure TTFT et TPOT d'un modèle local (décodage glouton, médiane sur `repeats` essais).

    TTFT ≈ durée d'une génération de 1 token (pré-remplissage + 1 pas de décodage) ;
    TPOT ≈ (durée pour n_out tokens - TTFT) / (n_out - 1). Génération forcée à n_out tokens.
    """
    import time

    import torch

    from . import llm

    model_id = model_id or llm.SMALL
    tok, model = llm.load(model_id)
    inputs = llm._chat_inputs(tok, prompt).to(model.device)

    def run(n):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                           temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - t

    run(4)  # échauffement
    t1 = float(np.median([run(1) for _ in range(repeats)]))
    tn = float(np.median([run(n_out) for _ in range(repeats)]))
    return {"ttft": t1, "tpot": (tn - t1) / (n_out - 1), "total": tn, "n_prompt": int(inputs["input_ids"].shape[1])}


#: critères de la grille de décision (inspirés des sections *Use Case Evaluation* et suivantes)
DECISION_CRITERIA = {
    "enjeu": "pourquoi construire ? (3 = menace existentielle, 1 = simple exploration)",
    "tolerance_erreurs": "l'application tolère-t-elle les erreurs ? (IA complémentaire, humain dans la boucle)",
    "evaluabilite": "peut-on mesurer objectivement la qualité ? (tâche fermée = facile)",
    "capacite_modele": "les modèles actuels sont-ils proches du seuil d'utilité ?",
    "valeur_vs_cout": "le gain attendu dépasse-t-il nettement le coût (inférence + équipe) ?",
    "defendabilite": "avantage durable : données propriétaires, distribution ?",
    "risque_reglementaire": "faible exposition réglementaire / propriété intellectuelle (3 = faible risque)",
}

DEFAULT_WEIGHTS = {"enjeu": 2, "tolerance_erreurs": 2, "evaluabilite": 2, "capacite_modele": 3,
                   "valeur_vs_cout": 3, "defendabilite": 1, "risque_reglementaire": 2}


def decision_score(scores: dict[str, float], weights: dict[str, float] | None = None,
                   low: float = 0, high: float = 3) -> float:
    """Score pondéré ramené sur 0-100 (notes de `low` à `high` pour chaque critère)."""
    weights = weights or DEFAULT_WEIGHTS
    missing = set(weights) - set(scores)
    if missing:
        raise KeyError(f"critères manquants : {sorted(missing)}")
    tot = sum(weights.values())
    s = sum(weights[k] * (scores[k] - low) / (high - low) for k in weights)
    return 100.0 * s / tot


def decision_verdict(score: float, scores: dict[str, float] | None = None, veto: float = 0) -> str:
    """Verdict lisible ; un critère à `veto` (0) sur la capacité du modèle ou la valeur bloque tout."""
    if scores and any(scores.get(k, 1) <= veto for k in ("capacite_modele", "valeur_vs_cout")):
        return "ne pas construire (critère bloquant)"
    if score >= 70:
        return "construire"
    if score >= 50:
        return "prototype limité, puis réévaluer"
    return "ne pas construire pour l'instant"


def quality_exponential(t, q_max: float = 100.0, tau: float = 1.0):
    """Modèle naïf : q(t) = q_max (1 - exp(-t / tau)) (l'écart restant décroît exponentiellement)."""
    return q_max * (1.0 - np.exp(-np.asarray(t, dtype=float) / tau))


def quality_power_law(t, a: float, b: float, q_max: float = 100.0):
    """Modèle « dernier kilomètre » : l'erreur restante décroît en loi de puissance, e(t) = a t^-b."""
    t = np.asarray(t, dtype=float)
    return q_max - a * np.power(t, -b)


def fit_power_law_error(t, quality, q_max: float = 100.0) -> tuple[float, float]:
    """Ajuste e(t) = q_max - q(t) = a t^-b par moindres carrés en échelle log-log. Renvoie (a, b)."""
    t = np.asarray(t, dtype=float)
    e = q_max - np.asarray(quality, dtype=float)
    slope, intercept = np.polyfit(np.log(t), np.log(e), 1)
    return float(np.exp(intercept)), float(-slope)


def time_to_quality(target: float, a: float, b: float, q_max: float = 100.0) -> float:
    """Temps nécessaire pour atteindre la qualité `target` sous le modèle en loi de puissance."""
    return float((a / (q_max - target)) ** (1.0 / b))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance de Wilson pour une proportion k / n (95 % par défaut)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


def samples_to_certify(p_true: float, threshold: float, z: float = 1.645, n_max: int = 100_000) -> int:
    """Plus petit n tel que la borne basse de Wilson (observée à p_true) dépasse `threshold`.

    Répond à : « combien de suggestions faut-il faire relire pour affirmer (à 95 %
    unilatéral) que le taux d'acceptation dépasse le seuil ? ».
    """
    if p_true <= threshold:
        return -1
    lo, hi = 1, n_max
    if wilson_interval(round(p_true * hi), hi, z)[0] <= threshold:
        return -1
    while lo < hi:
        mid = (lo + hi) // 2
        if wilson_interval(round(p_true * mid), mid, z)[0] > threshold:
            hi = mid
        else:
            lo = mid + 1
    return lo


def api_cumulative_cost(months: int, monthly_tokens_m: float, price_per_m: float,
                        halving_months: float | None = None) -> np.ndarray:
    """Coût cumulé d'une API dont le prix (par million de tokens) est divisé par 2 tous les
    `halving_months` mois (None : prix constant). Renvoie un tableau de longueur `months + 1`."""
    m = np.arange(months)
    price = price_per_m * (0.5 ** (m / halving_months) if halving_months else np.ones(months))
    return np.concatenate([[0.0], np.cumsum(monthly_tokens_m * price)])


def inhouse_cumulative_cost(months: int, upfront: float, monthly: float) -> np.ndarray:
    """Coût cumulé d'une solution interne : investissement initial + coût mensuel fixe."""
    return upfront + monthly * np.arange(months + 1, dtype=float)


def break_even_month(cost_a: np.ndarray, cost_b: np.ndarray) -> int | None:
    """Premier mois où la courbe cumulée `cost_b` devient inférieure ou égale à `cost_a`
    (après avoir été au-dessus) ; None s'il n'y a pas de croisement."""
    diff = np.asarray(cost_b) - np.asarray(cost_a)
    for i in range(1, len(diff)):
        if diff[i - 1] > 0 and diff[i] <= 0:
            return i
    return None
