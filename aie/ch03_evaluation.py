"""Chapitre 3 - Méthodologie d'évaluation : fonctions réutilisables.

Le module regroupe, dans l'ordre du chapitre :

1. **Métriques de modélisation du langage** : entropie, entropie croisée, divergence KL,
   conversions bits / nats, bits par caractère (BPC), bits par octet (BPB), perplexité,
   mesure de ces grandeurs sur un vrai modèle (``text_metrics``), score « Min-K% » pour
   la détection de contamination.
2. **Évaluation exacte** : estimateur non biaisé de pass@k, extraction et exécution
   isolée de code généré sur des tests unitaires, petit banc d'essai de problèmes Python.
3. **Similarité avec des références** : correspondance exacte (normalisée ou par
   inclusion), F1 sur les mots, distance d'édition, n-grammes, BLEU, ROUGE-N, ROUGE-L,
   similarité cosinus.
4. **IA comme juge** : probabilités d'un juge sur un ensemble fermé de réponses
   (``choice_probs``), score attendu sur une grille 1-5, coût d'un juge.
5. **Classement par comparaison** : Elo, Bradley-Terry (algorithme MM), simulation de
   matchs, probabilités de victoire, taille d'échantillon nécessaire.

Toutes les fonctions sans modèle sont en numpy pur (testées rapidement, sans GPU).
"""

from __future__ import annotations

import math
import random
import re
import string
import subprocess
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass

import numpy as np

LN2 = math.log(2.0)

# ======================================================================================
# 1. Métriques de modélisation du langage
# ======================================================================================


def entropy(p, base: float = 2.0) -> float:
    """Entropie H(P) = - sum p log p (les termes p = 0 valent 0 par convention)."""
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum() / math.log(base))


def cross_entropy(p, q, base: float = 2.0) -> float:
    """Entropie croisée H(P, Q) = - sum p log q (infinie si q = 0 là où p > 0)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p, q = p / p.sum(), q / q.sum()
    nz = p > 0
    if np.any(q[nz] == 0):
        return float("inf")
    return float(-(p[nz] * np.log(q[nz])).sum() / math.log(base))


def kl_divergence(p, q, base: float = 2.0) -> float:
    """Divergence de Kullback-Leibler D_KL(P || Q) = sum p log(p / q) >= 0."""
    return cross_entropy(p, q, base) - entropy(p, base)


def nats_to_bits(x):
    """Conversion nats -> bits (division par ln 2)."""
    return np.asarray(x) / LN2 if np.ndim(x) else float(x) / LN2


def bits_to_nats(x):
    return np.asarray(x) * LN2 if np.ndim(x) else float(x) * LN2


def perplexity_from_ce(ce: float, unit: str = "nats") -> float:
    """Perplexité à partir d'une entropie croisée par token : e^H (nats) ou 2^H (bits)."""
    return float(math.exp(ce) if unit == "nats" else 2.0 ** ce)


def bpc_from_bits_per_token(bits_per_token: float, chars_per_token: float) -> float:
    """Bits par caractère = bits par token / caractères par token."""
    return bits_per_token / chars_per_token


def bpb_from_bpc(bpc: float, bytes_per_char: float) -> float:
    """Bits par octet = bits par caractère / octets par caractère."""
    return bpc / bytes_per_char


@dataclass
class TextMetrics:
    """Mesures de modélisation du langage d'un modèle sur un texte."""

    model_id: str
    n_tokens: int
    n_chars: int
    n_bytes: int
    nll_nats: float  # somme des -log p (nats) sur tous les tokens du texte
    tokens: list
    logprobs: np.ndarray  # log-probabilité (nats) de chaque token

    @property
    def ce_nats(self) -> float:
        """Entropie croisée moyenne par token (nats)."""
        return self.nll_nats / self.n_tokens

    @property
    def ce_bits(self) -> float:
        return self.ce_nats / LN2

    @property
    def ppl(self) -> float:
        """Perplexité par token."""
        return math.exp(self.ce_nats)

    @property
    def bpc(self) -> float:
        return self.nll_nats / LN2 / self.n_chars

    @property
    def bpb(self) -> float:
        return self.nll_nats / LN2 / self.n_bytes

    @property
    def char_ppl(self) -> float:
        """Perplexité « par caractère » : 2^BPC (comparable entre tokenizers)."""
        return 2.0 ** self.bpc

    def as_dict(self) -> dict:
        return dict(modele=self.model_id.split("/")[-1], tokens=self.n_tokens, caracteres=self.n_chars,
                    octets=self.n_bytes, ce_nats=self.ce_nats, ce_bits=self.ce_bits, ppl=self.ppl,
                    bpc=self.bpc, bpb=self.bpb)


def text_metrics(text: str, model_id: str = "gpt2", max_tokens: int = 1024) -> TextMetrics:
    """Mesure CE, perplexité, BPC et BPB d'un modèle causal sur ``text``.

    Un token de début (``<|endoftext|>``, présent chez GPT-2 comme chez Qwen) est ajouté
    devant le texte : ainsi **tous** les tokens du texte sont prédits, et les métriques
    par caractère / par octet portent sur le texte entier (comparables entre tokenizers).
    """
    import torch

    from aie import llm

    tok, model = llm.load(model_id)
    bos = tok.convert_tokens_to_ids("<|endoftext|>")
    if bos is None or bos == tok.unk_token_id:
        bos = tok.eos_token_id
    ids = tok(text, add_special_tokens=False)["input_ids"][:max_tokens]
    x = torch.tensor([[bos] + ids], device=model.device)
    with torch.no_grad():
        logits = model(x).logits[0, :-1].float()
    lp = torch.log_softmax(logits, -1)
    target = x[0, 1:]
    out = lp[torch.arange(len(target)), target].cpu().numpy().astype(np.float64)
    used = tok.decode(ids) if len(ids) < len(tok(text, add_special_tokens=False)["input_ids"]) else text
    return TextMetrics(model_id, len(ids), len(used), len(used.encode("utf-8")), float(-out.sum()),
                       [tok.decode([i]) for i in ids], out)


def min_k_prob(logprobs, k: float = 0.2) -> float:
    """Score « Min-K% Prob » (Shi et al., 2023) : moyenne des k % log-probabilités les plus basses.

    Un texte mémorisé n'a presque aucun token très improbable : son score est élevé
    (proche de 0). Un texte jamais vu contient des tokens « surprenants » : score très négatif.
    """
    lp = np.sort(np.asarray(logprobs, dtype=np.float64))
    n = max(1, int(math.ceil(k * len(lp))))
    return float(lp[:n].mean())


def shuffle_words(text: str, seed: int = 0) -> str:
    """Mélange aléatoirement l'ordre des mots (même vocabulaire, structure détruite)."""
    words = text.split()
    rng = random.Random(seed)
    rng.shuffle(words)
    return " ".join(words)


def perturb_words(text: str, frac: float = 0.15, seed: int = 0) -> str:
    """Remplace une fraction des mots par des mots d'une petite liste neutre (paraphrase grossière)."""
    fillers = ["really", "quite", "also", "indeed", "perhaps", "simply", "often", "still"]
    rng = random.Random(seed)
    words = text.split()
    idx = rng.sample(range(len(words)), max(1, int(frac * len(words))))
    for i in idx:
        words[i] = rng.choice(fillers)
    return " ".join(words)


#: Textes très probablement vus (et répétés) pendant le pré-entraînement : domaine public
#: ou licences reproduites des millions de fois sur le Web.
MEMORIZED_TEXTS = {
    "Déclaration d'indépendance": (
        "When in the Course of human events, it becomes necessary for one people to dissolve the "
        "political bands which have connected them with another, and to assume among the powers of the "
        "earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle "
        "them, a decent respect to the opinions of mankind requires that they should declare the causes "
        "which impel them to the separation."),
    "Discours de Gettysburg": (
        "Four score and seven years ago our fathers brought forth on this continent, a new nation, "
        "conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we "
        "are engaged in a great civil war, testing whether that nation, or any nation so conceived and "
        "so dedicated, can long endure."),
    "Licence MIT": (
        "Permission is hereby granted, free of charge, to any person obtaining a copy of this software "
        "and associated documentation files (the \"Software\"), to deal in the Software without "
        "restriction, including without limitation the rights to use, copy, modify, merge, publish, "
        "distribute, sublicense, and/or sell copies of the Software."),
    "Zen of Python": (
        "Beautiful is better than ugly. Explicit is better than implicit. Simple is better than complex. "
        "Complex is better than complicated. Flat is better than nested. Sparse is better than dense. "
        "Readability counts. Special cases aren't special enough to break the rules."),
    "Genèse (King James)": (
        "In the beginning God created the heaven and the earth. And the earth was without form, and void; "
        "and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the "
        "waters. And God said, Let there be light: and there was light."),
    "Hamlet": (
        "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer the slings "
        "and arrows of outrageous fortune, or to take arms against a sea of troubles, and by opposing end "
        "them. To die: to sleep; no more; and by a sleep to say we end the heart-ache."),
}

#: Textes rédigés pour ce cours (donc absents des données d'entraînement), sur des thèmes
#: et dans des registres proches des textes mémorisés : seule la « nouveauté » change.
FRESH_TEXTS = {
    "Déclaration (inédite)": (
        "When the members of a small rowing club decide to leave the regional federation, courtesy "
        "suggests that they publish a short letter explaining why the annual fees, the travel rules and "
        "the lack of support for junior crews pushed them to organise their own races next spring."),
    "Discours (inédit)": (
        "Twelve winters ago our grandparents opened a modest library in this village, built from donated "
        "planks and stocked with books nobody else wanted. Today we gather to ask whether that library, "
        "or any library run by volunteers, can survive the arrival of the new motorway."),
    "Licence (inédite)": (
        "Anyone who receives a copy of these recipe cards may cook, adapt and share the dishes at family "
        "gatherings, provided that the name of the original baker is mentioned on the menu and that no "
        "card is sold separately from the wooden box in which it was delivered."),
    "Principes (inédits)": (
        "Short meetings beat long meetings. Written notes beat vague memories. A tested script beats a "
        "clever trick. Quiet servers beat noisy dashboards. Boring releases are good releases. Naming "
        "things carefully saves more time than any shortcut discovered at midnight."),
    "Récit (inédit)": (
        "On the first morning the lighthouse keeper repainted the stairs, and the gulls watched him from "
        "the railing without a sound. By noon the fog had covered the harbour, and the old radio began to "
        "repeat a weather report that nobody in the town remembered requesting."),
    "Monologue (inédit)": (
        "To stay or to move, that is the puzzle my landlord keeps raising: whether it is wiser to accept "
        "the noisy flat above the bakery, or to pack forty boxes, rent a van and gamble on a quieter "
        "street where the rent climbs every single time the tram line is extended."),
}

#: Textes à structure plus ou moins forte (pour illustrer « plus de structure, moins de perplexité »).
STRUCTURED_TEXTS = {
    "HTML": ("<html>\n<head>\n<title>Accueil</title>\n</head>\n<body>\n<ul>\n<li>Un</li>\n<li>Deux</li>\n"
             "<li>Trois</li>\n<li>Quatre</li>\n</ul>\n</body>\n</html>"),
    "JSON": ('{"users": [{"id": 1, "name": "Alice", "active": true}, {"id": 2, "name": "Bob", '
             '"active": false}, {"id": 3, "name": "Carol", "active": true}]}'),
    "Python": ("def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n\n\n"
               "def mul(a, b):\n    return a * b\n"),
    "Prose": ("The committee met on Tuesday to review the budget, but the discussion quickly drifted "
              "towards the broken coffee machine and the parking rules near the new building."),
    "Liste de nombres aléatoires": "93 17 58 4 71 26 88 35 9 62 47 13 80 51 2 69 38 95 24 76",
}


# ======================================================================================
# 2. Évaluation exacte : correction fonctionnelle et pass@k
# ======================================================================================


def pass_at_k(n: int, c: int, k: int) -> float:
    """Estimateur non biaisé de pass@k (Chen et al., 2021).

    n échantillons générés, dont c corrects : pass@k = 1 - C(n-c, k) / C(n, k).
    Calcul stable sous forme de produit : 1 - prod_{i=n-c+1}^{n} (1 - k / i).
    """
    if k > n:
        raise ValueError("k doit être <= n")
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def pass_at_k_naive(samples_ok, k: int, rng=None) -> float:
    """Estimateur « naïf » : tire k échantillons au hasard, renvoie 1 si l'un passe."""
    rng = rng or np.random.default_rng()
    s = np.asarray(samples_ok, dtype=bool)
    return float(rng.choice(s, size=k, replace=False).any())


def pass_at_k_true(p: float, k: int) -> float:
    """pass@k théorique pour un problème résolu avec probabilité p par échantillon."""
    return 1.0 - (1.0 - p) ** k


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    """Extrait le premier bloc de code Markdown (```python ... ```), sinon le texte brut."""
    m = _FENCE.search(text)
    if m:
        return m.group(1).strip("\n")
    m = re.search(r"```(?:python|py)?\s*\n(.*)", text, re.S)  # bloc non refermé (troncature)
    return (m.group(1) if m else text).strip("\n")


def sanitize_code(code: str) -> str:
    """Nettoie un programme généré : ne garde que les ``import`` et les définitions.

    1. Si le code ne se compile pas (souvent une génération tronquée), on retire des lignes
       à la fin jusqu'à obtenir un code valide.
    2. On ne conserve que les nœuds ``import``, ``def`` et ``class`` de premier niveau : les
       exemples d'utilisation, ``print`` et tests ajoutés par le modèle sont supprimés.
    C'est le type de post-traitement qu'appliquent les harnais d'évaluation de code.
    """
    import ast

    lines = code.splitlines()
    tree = None
    for end in range(len(lines), 0, -1):
        try:
            tree = ast.parse("\n".join(lines[:end]))
            break
        except SyntaxError:
            continue
    if tree is None:
        return code
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    return "\n\n".join(ast.unparse(n) for n in keep)


@dataclass
class ExecResult:
    passed: bool
    n_passed: int
    n_tests: int
    error: str


_RUNNER = """
import sys
ns = {}
try:
    exec(compile(SRC, "<candidate>", "exec"), ns)
except BaseException as e:
    print("COMPILE", type(e).__name__); sys.exit(0)
ok = 0
for t in TESTS:
    try:
        exec(t, ns); ok += 1
    except BaseException as e:
        print("FAIL", type(e).__name__)
print("PASSED", ok)
"""


def run_tests(code: str, tests: list[str], timeout: float = 5.0,
              preamble: str = "from typing import *\n") -> ExecResult:
    """Exécute ``code`` puis chaque ``assert`` de ``tests`` dans un **processus séparé**.

    Erreurs possibles : ``SyntaxError`` (code invalide), ``LoadError`` (le code lève une
    exception dès son chargement, par exemple dans ses propres exemples d'utilisation),
    ``Timeout``, ou le type de la première exception levée par un test.

    Isolation minimale (processus Python ``-I``, délai maximal) : suffisant pour un
    cours, mais un vrai banc d'essai exécute le code non fiable dans un bac à sable
    (conteneur, gVisor, Firecracker...). ``preamble`` est ajouté devant le code (comme
    les ``import`` fournis par les énoncés HumanEval).
    """
    code = preamble + code
    prog = f"SRC = {code!r}\nTESTS = {tests!r}\n" + _RUNNER
    try:
        r = subprocess.run([sys.executable, "-I", "-c", prog], capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return ExecResult(False, 0, len(tests), "Timeout")
    out = r.stdout
    m = re.search(r"PASSED (\d+)", out)
    if not m:
        c = re.search(r"COMPILE (\w+)", out)
        if c:  # syntaxe invalide, ou exception levée par le code au chargement (ex. ses propres tests)
            err = "SyntaxError" if c.group(1) in ("SyntaxError", "IndentationError", "TabError") else "LoadError"
        else:
            err = (r.stderr.strip().splitlines() or ["Crash"])[-1]
        return ExecResult(False, 0, len(tests), err[:80])
    k = int(m.group(1))
    fails = re.findall(r"FAIL (\w+)", out)
    return ExecResult(k == len(tests), k, len(tests), fails[0] if fails else "")


#: Petit banc d'essai façon HumanEval / MBPP : énoncé, signature, tests unitaires.
CODE_PROBLEMS = [
    dict(name="pgcd", signature="def gcd(a: int, b: int) -> int:",
         doc="Return the greatest common divisor of two positive integers a and b.",
         tests=["assert gcd(15, 20) == 5", "assert gcd(7, 13) == 1", "assert gcd(100, 75) == 25",
                "assert gcd(12, 12) == 12"]),
    dict(name="palindrome", signature="def is_palindrome(s: str) -> bool:",
         doc="Return True if s reads the same forwards and backwards, ignoring case and any "
             "non-alphanumeric character.",
         tests=["assert is_palindrome('A man, a plan, a canal: Panama')", "assert not is_palindrome('hello')",
                "assert is_palindrome('')", "assert is_palindrome('No lemon, no melon')"]),
    dict(name="voyelles", signature="def count_vowels(s: str) -> int:",
         doc="Return the number of vowels (a, e, i, o, u, case-insensitive) in s.",
         tests=["assert count_vowels('Hello World') == 3", "assert count_vowels('xyz') == 0",
                "assert count_vowels('AEIOU aeiou') == 10"]),
    dict(name="second_max", signature="def second_largest(nums: list) -> int:",
         doc="Return the second largest DISTINCT value of the list nums (which contains at least "
             "two distinct values).",
         tests=["assert second_largest([3, 1, 4, 1, 5]) == 4", "assert second_largest([10, 10, 9]) == 9",
                "assert second_largest([-2, -1]) == -2", "assert second_largest([5, 5, 5, 2]) == 2"]),
    dict(name="rle", signature="def run_length_encode(s: str) -> str:",
         doc="Run-length encode s: each maximal run of a character c of length n becomes c followed "
             "by n. Example: 'aaabcc' -> 'a3b1c2'.",
         tests=["assert run_length_encode('aaabcc') == 'a3b1c2'", "assert run_length_encode('') == ''",
                "assert run_length_encode('abc') == 'a1b1c1'", "assert run_length_encode('zzzzzzzzzzzz') == 'z12'"]),
    dict(name="parentheses", signature="def is_balanced(s: str) -> bool:",
         doc="Return True if the brackets '()', '[]' and '{}' in s are balanced and properly nested. "
             "Other characters are ignored.",
         tests=["assert is_balanced('([]{})')", "assert not is_balanced('([)]')", "assert is_balanced('a(b)c')",
                "assert not is_balanced('((')", "assert not is_balanced('}{')"]),
    dict(name="romain", signature="def roman_to_int(s: str) -> int:",
         doc="Convert a Roman numeral (I, V, X, L, C, D, M, with subtractive notation such as IV or "
             "CM) to an integer.",
         tests=["assert roman_to_int('III') == 3", "assert roman_to_int('IV') == 4", "assert roman_to_int('MCMXCIV') == 1994",
                "assert roman_to_int('LVIII') == 58"]),
    dict(name="intervalles", signature="def merge_intervals(intervals: list) -> list:",
         doc="Merge overlapping closed intervals given as [start, end] lists and return the merged "
             "intervals sorted by start. Touching intervals such as [1, 2] and [2, 3] overlap.",
         tests=["assert merge_intervals([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]",
                "assert merge_intervals([[5, 6], [1, 2], [2, 3]]) == [[1, 3], [5, 6]]",
                "assert merge_intervals([]) == []", "assert merge_intervals([[1, 10], [2, 3]]) == [[1, 10]]"]),
    dict(name="mots_freq", signature="def top_word(text: str) -> str:",
         doc="Return the most frequent word of text (words are separated by spaces, comparison is "
             "case-insensitive, the result is lowercase). Ties are broken by choosing the word that "
             "appears first in the text.",
         tests=["assert top_word('the cat and the dog') == 'the'", "assert top_word('B a b A') == 'b'",
                "assert top_word('one') == 'one'"]),
    dict(name="spirale", signature="def spiral_order(matrix: list) -> list:",
         doc="Return the elements of a rectangular matrix (list of rows) in clockwise spiral order, "
             "starting from the top-left corner.",
         tests=["assert spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]",
                "assert spiral_order([[1, 2], [3, 4], [5, 6]]) == [1, 2, 4, 6, 5, 3]",
                "assert spiral_order([]) == []", "assert spiral_order([[7]]) == [7]"]),
]


#: Solutions de référence (« canoniques ») des problèmes, utiles pour les métriques lexicales.
REFERENCE_SOLUTIONS = {
    "pgcd": "def gcd(a: int, b: int) -> int:\n    while b:\n        a, b = b, a % b\n    return a",
    "palindrome": ("def is_palindrome(s: str) -> bool:\n    t = [c.lower() for c in s if c.isalnum()]\n"
                   "    return t == t[::-1]"),
    "voyelles": "def count_vowels(s: str) -> int:\n    return sum(1 for c in s.lower() if c in 'aeiou')",
    "second_max": "def second_largest(nums: list) -> int:\n    return sorted(set(nums))[-2]",
    "rle": ("def run_length_encode(s: str) -> str:\n    out = []\n    i = 0\n    while i < len(s):\n"
            "        j = i\n        while j < len(s) and s[j] == s[i]:\n            j += 1\n"
            "        out.append(s[i] + str(j - i))\n        i = j\n    return ''.join(out)"),
    "parentheses": ("def is_balanced(s: str) -> bool:\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n"
                    "    for c in s:\n        if c in '([{':\n            stack.append(c)\n        elif c in pairs:\n"
                    "            if not stack or stack.pop() != pairs[c]:\n                return False\n"
                    "    return not stack"),
    "romain": ("def roman_to_int(s: str) -> int:\n    v = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
               "    total = 0\n    for i, c in enumerate(s):\n        if i + 1 < len(s) and v[c] < v[s[i + 1]]:\n"
               "            total -= v[c]\n        else:\n            total += v[c]\n    return total"),
    "intervalles": ("def merge_intervals(intervals: list) -> list:\n    out = []\n"
                    "    for a, b in sorted(intervals):\n        if out and a <= out[-1][1]:\n"
                    "            out[-1][1] = max(out[-1][1], b)\n        else:\n            out.append([a, b])\n"
                    "    return out"),
    "mots_freq": ("def top_word(text: str) -> str:\n    words = text.lower().split()\n"
                  "    return max(words, key=lambda w: (words.count(w), -words.index(w)))"),
    "spirale": ("def spiral_order(matrix: list) -> list:\n    out = []\n    m = [row[:] for row in matrix]\n"
                "    while m:\n        out += m.pop(0)\n        m = [list(r) for r in zip(*m)][::-1]\n    return out"),
}


def code_prompt(problem: dict) -> str:
    """Consigne envoyée au modèle pour un problème du banc d'essai."""
    return (f"Write a Python function with this exact signature:\n\n{problem['signature']}\n\n"
            f"{problem['doc']}\nAnswer with the code only, in a single ```python block: no explanation, "
            "no docstring, no comments, no usage examples.")


def sample_solutions(problem: dict, model_id: str | None = None, n: int = 20, temperature: float = 0.8,
                     top_p: float = 0.95, max_new_tokens: int = 384, seed: int = 0) -> list[str]:
    """Génère ``n`` réponses (en un seul lot) d'un modèle de chat pour un problème de code.

    temperature = 0 : une seule réponse gloutonne, renvoyée dans une liste.
    """
    import torch

    from aie import llm

    model_id = model_id or llm.SMALL
    tok, model = llm.load(model_id)
    msgs = [{"role": "user", "content": code_prompt(problem)}]
    x = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                return_dict=True).to(model.device)
    llm.seed_everything(seed)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=top_p, num_return_sequences=n)
    else:
        kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    with torch.no_grad():
        out = model.generate(**x, **kw)
    return tok.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)


# ======================================================================================
# 3. Similarité avec des références
# ======================================================================================

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """Normalisation façon SQuAD : minuscules, sans ponctuation, sans articles, espaces réduits."""
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, refs, normalize: bool = True) -> bool:
    """Correspondance exacte avec l'une des références (après normalisation par défaut)."""
    refs = [refs] if isinstance(refs, str) else list(refs)
    f = normalize_answer if normalize else (lambda x: x)
    return any(f(pred) == f(r) for r in refs)


def contains_match(pred: str, refs) -> bool:
    """Variante laxiste : la réponse contient l'une des références (risque de faux positifs)."""
    refs = [refs] if isinstance(refs, str) else list(refs)
    p = normalize_answer(pred)
    return any(re.search(rf"\b{re.escape(normalize_answer(r))}\b", p) is not None for r in refs)


def words(text: str) -> list[str]:
    """Découpage simple en mots (minuscules, caractères alphanumériques Unicode)."""
    return re.findall(r"\w+", text.lower())


def token_f1(pred: str, ref: str) -> float:
    """F1 sur les mots (sac de mots), métrique classique de SQuAD."""
    p, r = normalize_answer(pred).split(), normalize_answer(ref).split()
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def edit_distance(a, b, transpositions: bool = False) -> int:
    """Distance de Levenshtein (suppression, insertion, substitution), programmation dynamique.

    transpositions=True : ajoute l'échange de deux symboles adjacents (distance « OSA »,
    variante restreinte de Damerau-Levenshtein). ``a`` et ``b`` peuvent être des chaînes
    ou des listes de mots.
    """
    n, m = len(a), len(b)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
            if transpositions and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i, j] = min(d[i, j], d[i - 2, j - 2] + 1)
    return int(d[n, m])


def edit_similarity(a: str, b: str) -> float:
    """Similarité normalisée 1 - distance / longueur maximale (dans [0, 1])."""
    if not a and not b:
        return 1.0
    return 1.0 - edit_distance(a, b) / max(len(a), len(b))


def ngrams(tokens, n: int) -> Counter:
    """Compteur des n-grammes d'une liste de tokens."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(candidate: str, references, max_n: int = 4, smooth: bool = True) -> float:
    """BLEU au niveau de la phrase (Papineni et al., 2002).

    BLEU = BP * exp(moyenne_n log p_n), où p_n est la précision des n-grammes **écrêtée**
    (un n-gramme du candidat ne compte pas plus de fois qu'il n'apparaît dans la meilleure
    référence) et BP = min(1, exp(1 - r / c)) la pénalité de brièveté (c : longueur du
    candidat, r : longueur de la référence la plus proche).
    smooth=True : lissage « +1 » pour n >= 2 (Lin et Och, 2004), utile sur des phrases courtes.
    """
    refs = [references] if isinstance(references, str) else list(references)
    cand = words(candidate)
    refs_t = [words(r) for r in refs]
    if not cand:
        return 0.0
    log_p = 0.0
    for n in range(1, max_n + 1):
        c_ng = ngrams(cand, n)
        max_ref = Counter()
        for r in refs_t:
            max_ref |= ngrams(r, n)
        clipped = sum(min(v, max_ref[g]) for g, v in c_ng.items())
        total = max(1, sum(c_ng.values()))
        if smooth and n > 1:
            p = (clipped + 1) / (total + 1)
        else:
            p = clipped / total
        if p == 0:
            return 0.0
        log_p += math.log(p) / max_n
    c = len(cand)
    r = min((abs(len(rt) - c), len(rt)) for rt in refs_t)[1]
    bp = 1.0 if c > r else math.exp(1 - r / c)
    return float(bp * math.exp(log_p))


def rouge_n(candidate: str, reference: str, n: int = 1) -> dict:
    """ROUGE-N (Lin, 2004) : rappel, précision et F1 des n-grammes communs."""
    c, r = ngrams(words(candidate), n), ngrams(words(reference), n)
    overlap = sum((c & r).values())
    rec = overlap / max(1, sum(r.values()))
    prec = overlap / max(1, sum(c.values()))
    f1 = 0.0 if overlap == 0 else 2 * prec * rec / (prec + rec)
    return dict(rappel=rec, precision=prec, f1=f1)


def lcs_length(a, b) -> int:
    """Longueur de la plus longue sous-séquence commune (non nécessairement contiguë)."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(candidate: str, reference: str) -> dict:
    """ROUGE-L : rappel, précision et F1 fondés sur la plus longue sous-séquence commune."""
    c, r = words(candidate), words(reference)
    lcs = lcs_length(c, r)
    rec, prec = lcs / max(1, len(r)), lcs / max(1, len(c))
    f1 = 0.0 if lcs == 0 else 2 * prec * rec / (prec + rec)
    return dict(rappel=rec, precision=prec, f1=f1)


def cosine_similarity(a, b) -> float | np.ndarray:
    """Similarité cosinus A.B / (||A|| ||B||). Matrices : similarités deux à deux des lignes."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim == 1 and b.ndim == 1:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


# ======================================================================================
# 4. IA comme juge
# ======================================================================================


def choice_probs(messages, choices, model_id: str | None = None, prefix: str = "",
                 system: str | None = None) -> dict:
    """Probabilités qu'un modèle de chat commence sa réponse par chacun des ``choices``.

    Une seule passe avant (aucune génération) : on lit la distribution du premier token
    de la réponse, restreinte aux premiers tokens des choix (« 1 »...« 5 », « A »/« B »...).
    Renvoie {choix: probabilité renormalisée} et la clé spéciale ``"_masse"`` (probabilité
    totale que le modèle réponde par l'un des choix, avant renormalisation).
    """
    import torch

    from aie import llm

    model_id = model_id or llm.SMALL
    tok, model = llm.load(model_id)
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if system is not None:
        messages = [{"role": "system", "content": system}] + list(messages)
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) + prefix
    x = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        logits = model(**x).logits[0, -1].float()
    p = torch.softmax(logits, -1).cpu().numpy().astype(np.float64)
    ids = [tok.encode(str(c), add_special_tokens=False)[0] for c in choices]
    raw = np.array([p[i] for i in ids])
    out = {c: float(v) for c, v in zip(choices, raw / raw.sum())}
    out["_masse"] = float(raw.sum())
    return out


def expected_score(probs: dict) -> float:
    """Score attendu E[s] = sum s p(s) à partir de ``choice_probs`` sur des scores numériques."""
    return float(sum(float(k) * v for k, v in probs.items() if not k.startswith("_")))


RUBRIC_1_5 = """You are a strict grader. Rate the ANSWER to the QUESTION for factual correctness and helpfulness.
Scoring rubric:
1 = wrong or irrelevant answer.
2 = mostly wrong, with a small correct element.
3 = partially correct or incomplete.
4 = correct but with a minor flaw.
5 = fully correct, clear and complete.
Reply with a single digit from 1 to 5."""

PAIRWISE_PROMPT = """You are an impartial judge. Given the QUESTION and two answers, decide which answer is more correct and helpful.
Reply with a single letter: A or B.

QUESTION: {question}

ANSWER A: {a}

ANSWER B: {b}"""


def judge_pointwise(question: str, answer: str, model_id: str | None = None, rubric: str = RUBRIC_1_5) -> dict:
    """Juge « ponctuel » : distribution sur les notes 1..5 et note attendue."""
    msg = f"{rubric}\n\nQUESTION: {question}\nANSWER: {answer}"
    pr = choice_probs(msg, ["1", "2", "3", "4", "5"], model_id=model_id, prefix="Score: ")
    return dict(probs=pr, score=expected_score(pr), argmax=int(max("12345", key=lambda s: pr[s])))


def judge_pairwise(question: str, a: str, b: str, model_id: str | None = None) -> float:
    """Juge par paires : probabilité que le juge préfère la réponse affichée en position A."""
    pr = choice_probs(PAIRWISE_PROMPT.format(question=question, a=a, b=b), ["A", "B"], model_id=model_id)
    return pr["A"]


#: Petit jeu étiqueté pour évaluer un juge : (question, réponse, note humaine de 1 à 5).
JUDGE_QA = [
    ("What is the capital of Australia?", "The capital of Australia is Canberra.", 5),
    ("What is the capital of Australia?", "The capital of Australia is Sydney.", 1),
    ("What is the capital of Australia?", "Australia's capital is a city in the southeast.", 3),
    ("How many legs does a spider have?", "A spider has eight legs.", 5),
    ("How many legs does a spider have?", "Spiders have six legs, like all insects.", 1),
    ("What is 12 times 11?", "12 times 11 is 132.", 5),
    ("What is 12 times 11?", "12 times 11 is 122.", 1),
    ("Who wrote 'Pride and Prejudice'?", "Jane Austen wrote it, publishing it in 1813.", 5),
    ("Who wrote 'Pride and Prejudice'?", "It was written by Charlotte Bronte.", 1),
    ("What gas do plants absorb from the air for photosynthesis?", "Plants absorb carbon dioxide.", 5),
    ("What gas do plants absorb from the air for photosynthesis?", "Plants absorb oxygen and release nitrogen.", 1),
    ("What gas do plants absorb from the air for photosynthesis?", "They absorb a gas from the air.", 2),
    ("At what temperature does water boil at sea level, in Celsius?", "100 degrees Celsius.", 5),
    ("At what temperature does water boil at sea level, in Celsius?", "Around 90 degrees Celsius.", 2),
    ("What is the largest planet of the Solar System?", "Jupiter is the largest planet.", 5),
    ("What is the largest planet of the Solar System?", "Saturn, because of its rings.", 1),
]

#: Paires (question, bonne réponse courte, réponse longue et fausse) pour le biais de verbosité.
VERBOSE_PAIRS = [
    ("What is the capital of Australia?", "Canberra.",
     "Great question! Australia is a vast country with many famous cities such as Sydney, Melbourne and "
     "Brisbane. After careful consideration of its history, its federal institutions and its role in the "
     "region, the capital of Australia is Sydney, which hosts the national parliament and many government offices."),
    ("How many legs does a spider have?", "Eight.",
     "Spiders are fascinating arthropods found on every continent except Antarctica. They play an important "
     "role in ecosystems by controlling insect populations. Like all insects, a spider has six legs, which it "
     "uses for walking, sensing vibrations and handling prey with remarkable precision."),
    ("What is 12 times 11?", "132.",
     "To compute this product, we can decompose the multiplication: 12 times 11 equals 12 times 10 plus 12 "
     "times 1. Twelve times ten is one hundred and twenty, and adding twelve gives a total of 122. Therefore, "
     "the final answer to 12 times 11 is 122."),
    ("Who wrote 'Pride and Prejudice'?", "Jane Austen.",
     "Pride and Prejudice is one of the most beloved novels in English literature, famous for its witty "
     "dialogue and its portrait of the Bennet family. It was written by Charlotte Bronte, who published it in "
     "the early nineteenth century, and it remains widely read today."),
    ("What gas do plants absorb for photosynthesis?", "Carbon dioxide.",
     "Photosynthesis is the process by which green plants convert light energy into chemical energy stored "
     "in sugars. During this process, plants absorb oxygen from the air through small pores called stomata "
     "and combine it with water, releasing nitrogen as a by-product into the atmosphere."),
    ("What is the largest planet of the Solar System?", "Jupiter.",
     "The Solar System contains eight planets, divided into rocky inner planets and giant outer planets. "
     "Among the giants, the most impressive is Saturn, whose spectacular rings and enormous diameter make it "
     "the largest planet orbiting the Sun, far ahead of Jupiter and Neptune."),
]


def judge_cost(n_requests: int, gen_in: int, gen_out: int, judge_out: int, price_in: float, price_out: float,
               n_criteria: int = 1, sample_rate: float = 1.0, judge_overhead: int = 300) -> dict:
    """Coût (en dollars) d'une application avec juge IA.

    Le juge relit la question et la réponse (gen_in + gen_out tokens) plus sa propre consigne
    (judge_overhead tokens), pour chaque critère, sur une fraction ``sample_rate`` des requêtes.
    Prix exprimés en dollars par million de tokens.
    """
    gen = n_requests * (gen_in * price_in + gen_out * price_out) / 1e6
    n_judged = n_requests * sample_rate * n_criteria
    judge = n_judged * ((gen_in + gen_out + judge_overhead) * price_in + judge_out * price_out) / 1e6
    return dict(generation=gen, juge=judge, total=gen + judge, surcout=judge / gen if gen else float("nan"))


def spot_check_sample_size(p: float, margin: float, confidence: float = 0.95) -> int:
    """Nombre d'échantillons pour estimer un taux p à +/- margin (approximation normale)."""
    from scipy.stats import norm

    z = norm.ppf(0.5 + confidence / 2)
    return int(math.ceil(z ** 2 * p * (1 - p) / margin ** 2))


# ======================================================================================
# 5. Classement par comparaison : Elo et Bradley-Terry
# ======================================================================================


def elo_expected(ra: float, rb: float, scale: float = 400.0) -> float:
    """Probabilité de victoire de A selon Elo : 1 / (1 + 10^((rb - ra) / scale))."""
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / scale))


def elo_update(ra: float, rb: float, score_a: float, k: float = 32.0, scale: float = 400.0):
    """Mise à jour Elo après un match (score_a = 1 victoire, 0 défaite, 0.5 nul)."""
    ea = elo_expected(ra, rb, scale)
    delta = k * (score_a - ea)
    return ra + delta, rb - delta


def elo_ratings(matches, n_models: int, k: float = 32.0, init: float = 1000.0, scale: float = 400.0,
                history: bool = False):
    """Classement Elo séquentiel. ``matches`` : itérable de (a, b, score_a)."""
    r = np.full(n_models, init, dtype=np.float64)
    hist = []
    for a, b, s in matches:
        r[a], r[b] = elo_update(r[a], r[b], s, k, scale)
        if history:
            hist.append(r.copy())
    return (r, np.array(hist)) if history else r


def win_matrix(matches, n_models: int) -> np.ndarray:
    """W[i, j] = nombre de victoires de i contre j (un nul compte 0,5 pour chacun)."""
    w = np.zeros((n_models, n_models))
    for a, b, s in matches:
        w[a, b] += s
        w[b, a] += 1 - s
    return w


def bradley_terry(wins: np.ndarray, n_iter: int = 1000, tol: float = 1e-10, prior: float = 0.0) -> np.ndarray:
    """Ajuste Bradley-Terry par l'algorithme MM (Hunter, 2004) ; renvoie les forces log(pi).

    Modèle : P(i bat j) = pi_i / (pi_i + pi_j). Itération :
    pi_i <- W_i / sum_j n_ij / (pi_i + pi_j), avec W_i les victoires de i et n_ij le nombre de
    matchs i-j. ``prior`` > 0 ajoute des pseudo-matchs nuls (régularisation, utile si un
    modèle n'a jamais perdu). Les forces sont centrées (moyenne nulle).
    """
    w = np.asarray(wins, dtype=np.float64).copy()
    m = w.shape[0]
    if prior > 0:
        w = w + prior * (1 - np.eye(m)) / 2
    n = w + w.T
    pi = np.ones(m)
    for _ in range(n_iter):
        denom = (n / (pi[:, None] + pi[None, :])).sum(axis=1)
        new = w.sum(axis=1) / denom
        new = new / np.exp(np.log(new).mean())
        if np.max(np.abs(new - pi)) < tol:
            pi = new
            break
        pi = new
    s = np.log(pi)
    return s - s.mean()


def bt_to_elo(strength, scale: float = 400.0, base: float = 1000.0) -> np.ndarray:
    """Convertit des forces log(pi) (base e) en échelle « Elo » : base + scale * log10(pi)."""
    return base + scale * np.asarray(strength) / math.log(10)


def bt_prob(si: float, sj: float) -> float:
    """P(i bat j) pour des forces log(pi) de Bradley-Terry."""
    return 1.0 / (1.0 + math.exp(sj - si))


def simulate_matches(prob_matrix: np.ndarray, n_matches: int, rng=None, pairs=None) -> list:
    """Simule des matchs : paires tirées uniformément, vainqueur selon prob_matrix[a, b].

    prob_matrix[a, b] = P(a bat b). Renvoie une liste de (a, b, score_a).
    """
    rng = rng or np.random.default_rng(0)
    m = prob_matrix.shape[0]
    out = []
    for _ in range(n_matches):
        if pairs is None:
            a, b = rng.choice(m, size=2, replace=False)
        else:
            a, b = pairs[rng.integers(len(pairs))]
        out.append((int(a), int(b), float(rng.random() < prob_matrix[a, b])))
    return out


def prob_matrix_from_elo(ratings, scale: float = 400.0) -> np.ndarray:
    """Matrice P[i, j] = P(i bat j) induite par des scores Elo (modèle transitif)."""
    r = np.asarray(ratings, dtype=np.float64)
    return 1.0 / (1.0 + 10.0 ** ((r[None, :] - r[:, None]) / scale))


def ranking_accuracy(scores, matches) -> float:
    """Fraction des matchs (hors nuls) dont le vainqueur est le mieux classé."""
    scores = np.asarray(scores)
    ok = [(scores[a] > scores[b]) == (s == 1.0) for a, b, s in matches if s in (0.0, 1.0)]
    return float(np.mean(ok)) if ok else float("nan")


def matches_needed(p: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Nombre de matchs pour distinguer un taux de victoire p de 0,5 (test bilatéral).

    n = ((z_{1-alpha/2} * 0.5 + z_{power} * sqrt(p (1 - p))) / (p - 0.5))^2.
    """
    from scipy.stats import norm

    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return int(math.ceil(((za * 0.5 + zb * math.sqrt(p * (1 - p))) / (p - 0.5)) ** 2))


#: Table 3-6 du livre (section *Ranking Models with Comparative Evaluation*) :
#: (modèle A, modèle B, taux de victoire de A), 1 000 matchs par paire, modèles numérotés de 1 à 5.
BOOK_WIN_RATES = [
    (1, 2, 0.90), (1, 3, 0.40), (1, 4, 0.15), (1, 5, 0.10), (2, 3, 0.60),
    (2, 4, 0.80), (2, 5, 0.80), (3, 4, 0.70), (3, 5, 0.10), (4, 5, 0.20),
]


def book_win_matrix(n_per_pair: int = 1000) -> np.ndarray:
    """Matrice des victoires (indices 0..4) correspondant à la Table 3-6 du livre."""
    w = np.zeros((5, 5))
    for a, b, p in BOOK_WIN_RATES:
        w[a - 1, b - 1] = p * n_per_pair
        w[b - 1, a - 1] = (1 - p) * n_per_pair
    return w


def find_cycles3(prob_matrix: np.ndarray) -> list[tuple]:
    """Triplets intransitifs (i bat j, j bat k, k bat i, au sens d'une probabilité > 0,5)."""
    m = prob_matrix.shape[0]
    beats = prob_matrix > 0.5
    cyc = set()
    for i in range(m):
        for j in range(m):
            for k in range(m):
                if len({i, j, k}) == 3 and beats[i, j] and beats[j, k] and beats[k, i]:
                    cyc.add(tuple(np.roll([i, j, k], -int(np.argmin([i, j, k])))))
    return sorted(cyc)


def wrap(text: str, width: int = 90, max_chars: int | None = None) -> str:
    """Coupe un texte long pour l'affichage (avec troncature optionnelle)."""
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip() + " [...]"
    return textwrap.fill(text, width)
