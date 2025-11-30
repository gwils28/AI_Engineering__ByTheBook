"""Chapitre 8 : ingénierie des données (Dataset Engineering).

Fonctions réutilisables des notebooks du chapitre 8, regroupées par thème :

- **curation** : données synthétiques à sous-populations, bruit d'étiquetage, courbes
  d'apprentissage, ajustement d'une loi de puissance, mesures de diversité ;
- **annotation** : kappa de Cohen (pondéré ou non), kappa de Fleiss, simulation
  d'annotateurs ;
- **augmentation et synthèse** : gabarits, substitution de genre, perturbations (fautes de
  frappe, bruit de tokens), problèmes arithmétiques vérifiables, filtres « Self-Instruct » ;
- **effondrement de modèle** : simulations gaussienne et catégorielle ;
- **distillation** : perte de distillation et petit réseau entraînable sur cibles douces ;
- **traitement** : jeu de données « sale », détection de langue, nettoyage (HTML, PII),
  hachage exact, MinHash et LSH, mise au format SFT avec masquage de la perte.

Tout est écrit en numpy / Python pur (torch seulement pour le réseau de distillation)
afin que chaque formule du cours soit visible dans le code.
"""

from __future__ import annotations

import hashlib
import html
import itertools
import re
import zlib
from collections import Counter, defaultdict

import numpy as np

# ======================================================================================
# 1. Curation : données synthétiques, bruit, courbes d'apprentissage, diversité
# ======================================================================================


def make_subpop_data(n: int, n_classes: int = 4, n_subpops: int = 3, dim: int = 10,
                     sep: float = 3.0, noise: float = 1.0, seed: int = 0, centers=None):
    """Données gaussiennes où chaque classe est un mélange de `n_subpops` sous-populations.

    Les sous-populations jouent le rôle des « manières d'exprimer une demande » : une classe
    (une intention utilisateur) peut apparaître sous plusieurs formes très différentes.
    Renvoie (X, y, s, centers) où s est l'indice global de sous-population
    (classe * n_subpops + k). Passer `centers` pour tirer de nouveaux points autour des
    mêmes centres (jeu de test cohérent avec le jeu d'entraînement).
    """
    rng = np.random.default_rng(seed)
    n_groups = n_classes * n_subpops
    if centers is None:
        centers = np.random.default_rng(seed + 10_000).normal(0, sep, size=(n_groups, dim))
    s = rng.integers(0, n_groups, size=n)
    X = centers[s] + rng.normal(0, noise, size=(n, dim))
    y = s // n_subpops
    return X, y, s, centers


def flip_labels(y, rate: float, n_classes: int | None = None, seed: int = 0,
                target: int | None = None):
    """Bruit d'étiquetage : une fraction `rate` des étiquettes est remplacée.

    target=None : bruit symétrique (la nouvelle étiquette est tirée uniformément parmi les
    autres classes). target=k : bruit systématique (les étiquettes corrompues deviennent k),
    comme un annotateur qui a mal compris une consigne.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).copy()
    n_classes = n_classes or int(y.max()) + 1
    idx = rng.choice(len(y), size=int(round(rate * len(y))), replace=False)
    if target is None:
        shift = rng.integers(1, n_classes, size=len(idx))
        y[idx] = (y[idx] + shift) % n_classes
    else:
        y[idx] = target
    return y


def learning_curve(X_train, y_train, X_test, y_test, sizes, n_repeats: int = 5, seed: int = 0,
                   model_factory=None):
    """Exactitude de test moyenne (et écart-type) pour des sous-ensembles de tailles `sizes`.

    Pour chaque taille, on tire `n_repeats` sous-échantillons aléatoires du jeu
    d'entraînement, on entraîne le modèle et on mesure l'exactitude sur le jeu de test.
    Renvoie (moyennes, écarts-types) sous forme de tableaux numpy.
    """
    from sklearn.linear_model import LogisticRegression

    model_factory = model_factory or (lambda: LogisticRegression(max_iter=2000))
    rng = np.random.default_rng(seed)
    means, stds = [], []
    for n in sizes:
        scores = []
        for _ in range(n_repeats):
            idx = rng.choice(len(y_train), size=int(n), replace=False)
            if len(np.unique(y_train[idx])) < 2:
                scores.append(np.mean(y_test == np.bincount(y_train[idx]).argmax()))
                continue
            m = model_factory().fit(X_train[idx], y_train[idx])
            scores.append(m.score(X_test, y_test))
        means.append(np.mean(scores))
        stds.append(np.std(scores))
    return np.array(means), np.array(stds)


def power_law(n, a, b, c):
    """Erreur modélisée par une loi de puissance : err(n) = a * n^(-b) + c."""
    return a * np.power(n, -b) + c


def fit_power_law(sizes, errors):
    """Ajuste err(n) = a n^-b + c (c = erreur irréductible). Renvoie (a, b, c)."""
    from scipy.optimize import curve_fit

    sizes, errors = np.asarray(sizes, float), np.asarray(errors, float)
    p0 = (errors[0] * sizes[0] ** 0.5, 0.5, max(errors.min() * 0.5, 1e-4))
    params, _ = curve_fit(power_law, sizes, errors, p0=p0,
                          bounds=([0, 0, 0], [np.inf, 3, 1]), maxfev=20_000)
    return tuple(float(v) for v in params)


def distinct_n(texts, n: int = 2) -> float:
    """Proportion de n-grammes (de mots) distincts parmi tous les n-grammes des textes."""
    grams = []
    for t in texts:
        w = t.lower().split()
        grams += [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def mean_pairwise_distance(E) -> float:
    """Distance cosinus moyenne entre paires d'embeddings (supposés normalisés)."""
    E = np.asarray(E, float)
    S = E @ E.T
    n = len(E)
    return float(1 - (S.sum() - np.trace(S)) / (n * (n - 1)))


def vendi_score(E) -> float:
    """Vendi score : « nombre effectif » d'exemples distincts d'un ensemble.

    Avec K la matrice de similarité cosinus (n x n) et lambda_i les valeurs propres de K/n,
    VS = exp(- sum lambda_i log lambda_i). Vaut 1 si tous les exemples sont identiques et n
    s'ils sont tous orthogonaux.
    """
    E = np.asarray(E, float)
    K = np.clip(E @ E.T, 0, None) / len(E)
    lam = np.linalg.eigvalsh(K)
    lam = lam[lam > 1e-12]
    return float(np.exp(-(lam * np.log(lam)).sum()))


# ======================================================================================
# 2. Annotation : accord inter-annotateurs
# ======================================================================================


def cohen_kappa(a, b, weights: str | None = None, labels=None) -> float:
    """Kappa de Cohen entre deux annotateurs.

    kappa = (p_o - p_e) / (1 - p_e) où p_o est l'accord observé et p_e l'accord attendu
    par hasard (produit des distributions marginales). Avec weights="linear" ou
    "quadratic", les désaccords sont pondérés par leur distance (échelles ordinales
    comme une note de 1 à 5) : kappa = 1 - sum(w O) / sum(w E).
    """
    a, b = np.asarray(a), np.asarray(b)
    labels = np.unique(np.concatenate([a, b])) if labels is None else np.asarray(labels)
    k = len(labels)
    pos = {v: i for i, v in enumerate(labels)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[pos[x], pos[y]] += 1
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    i, j = np.indices((k, k))
    if weights is None:
        W = (i != j).astype(float)
    elif weights == "linear":
        W = np.abs(i - j) / max(k - 1, 1)
    elif weights == "quadratic":
        W = ((i - j) / max(k - 1, 1)) ** 2
    else:
        raise ValueError("weights doit valoir None, 'linear' ou 'quadratic'")
    den = (W * E).sum()
    if den == 0:
        return 1.0
    return float(1 - (W * O).sum() / den)


def ratings_to_counts(ratings, n_categories: int | None = None):
    """Matrice (items x annotateurs) d'étiquettes entières -> matrice (items x catégories)
    du nombre d'annotateurs ayant choisi chaque catégorie (entrée du kappa de Fleiss)."""
    ratings = np.asarray(ratings, int)
    n_categories = n_categories or int(ratings.max()) + 1
    return np.stack([(ratings == c).sum(1) for c in range(n_categories)], axis=1)


def fleiss_kappa(counts) -> float:
    """Kappa de Fleiss pour m annotateurs (m identique pour chaque item).

    counts[i, j] = nombre d'annotateurs ayant mis l'item i dans la catégorie j.
    P_i = (sum_j n_ij^2 - m) / (m (m - 1)) est l'accord sur l'item i ;
    P_e = sum_j p_j^2 avec p_j la proportion globale de la catégorie j.
    kappa = (mean(P_i) - P_e) / (1 - P_e).
    """
    counts = np.asarray(counts, float)
    m = counts.sum(1)
    if not np.allclose(m, m[0]):
        raise ValueError("chaque item doit avoir le même nombre d'annotations")
    m = m[0]
    P_i = ((counts ** 2).sum(1) - m) / (m * (m - 1))
    p_j = counts.sum(0) / counts.sum()
    P_e = (p_j ** 2).sum()
    if np.isclose(P_e, 1):
        return 1.0
    return float((P_i.mean() - P_e) / (1 - P_e))


def simulate_annotators(true_labels, accuracies, n_classes: int, seed: int = 0, bias=None):
    """Simule des annotateurs : l'annotateur j donne la bonne étiquette avec la probabilité
    accuracies[j], sinon une étiquette erronée. bias[j] (optionnel, entier ou None) : classe
    vers laquelle l'annotateur j se trompe systématiquement. Renvoie (items x annotateurs)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(true_labels)
    bias = bias or [None] * len(accuracies)
    out = np.empty((len(y), len(accuracies)), dtype=int)
    for j, (acc, bj) in enumerate(zip(accuracies, bias)):
        ok = rng.random(len(y)) < acc
        if bj is None:
            wrong = (y + rng.integers(1, n_classes, size=len(y))) % n_classes
        else:
            wrong = np.full(len(y), bj)
        out[:, j] = np.where(ok, y, wrong)
    return out


def kappa_interpretation(k: float) -> str:
    """Grille de lecture usuelle (Landis et Koch, 1977) d'un kappa."""
    bornes = [(0.0, "pire que le hasard"), (0.2, "faible"), (0.4, "passable"),
              (0.6, "modéré"), (0.8, "substantiel"), (1.01, "quasi parfait")]
    if k < 0:
        return bornes[0][1]
    for b, txt in bornes[1:]:
        if k < b:
            return txt
    return "quasi parfait"


# ======================================================================================
# 3. Augmentation et synthèse traditionnelles
# ======================================================================================


def fill_template(template: str, fields: dict, rng) -> str:
    """Remplit un gabarit « {champ} » en tirant chaque champ dans sa liste de valeurs
    (ou en appelant la fonction associée avec le générateur aléatoire)."""
    vals = {}
    for k, v in fields.items():
        vals[k] = v(rng) if callable(v) else v[rng.integers(len(v))]
    return template.format(**vals)


_GENDER_PAIRS = [("he", "she"), ("his", "her"), ("him", "her"), ("himself", "herself"),
                 ("man", "woman"), ("men", "women"), ("boy", "girl"), ("boys", "girls"),
                 ("father", "mother"), ("dad", "mom"), ("son", "daughter"), ("brother", "sister"),
                 ("husband", "wife"), ("mr", "ms"), ("king", "queen"), ("uncle", "aunt"),
                 ("actor", "actress"), ("waiter", "waitress"), ("gentleman", "lady")]


def _gender_map():
    m = {}
    for a, b in _GENDER_PAIRS:
        m.setdefault(a, b)
        m.setdefault(b, a)  # "her" -> "his" (première occurrence) : ambiguïté assumée
    return m


GENDER_MAP = _gender_map()


def _match_case(src: str, dst: str) -> str:
    if src.isupper() and len(src) > 1:
        return dst.upper()
    if src[0].isupper():
        return dst[0].upper() + dst[1:]
    return dst


def gender_swap(text: str, mapping: dict | None = None) -> str:
    """Augmentation contrefactuelle : échange les mots genrés (anglais) en gardant la casse.

    Limite connue (discutée en exercice) : « her » peut correspondre à « him » ou « his »
    selon la fonction grammaticale ; un dictionnaire ne peut pas trancher.
    """
    mapping = mapping or GENDER_MAP

    def rep(m):
        w = m.group(0)
        low = w.lower()
        return _match_case(w, mapping[low]) if low in mapping else w

    return re.sub(r"[A-Za-z]+", rep, text)


_KEYBOARD = {c: n for c, n in zip("azertyuiopqsdfghjklmwxcvbn",
                                  ["qze", "aqsde", "zsdfr", "edfgt", "rfghy", "tghju", "yhjki",
                                   "ujklo", "iklmp", "olm", "azsw", "qzedxw", "sezfcx", "drtgvc",
                                   "ftyhbv", "gyujnb", "huiknb", "jiolm", "kopm", "lp", "qsx",
                                   "wsdc", "xdfv", "cfgb", "vghn", "bhj"])}


def typo_noise(text: str, rate: float = 0.05, seed: int = 0) -> str:
    """Perturbation de texte : chaque lettre subit, avec la probabilité `rate`, une faute de
    frappe (inversion avec la suivante, suppression, doublement ou touche voisine AZERTY)."""
    rng = np.random.default_rng(seed)
    chars = list(text)
    out = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c.isalpha() and rng.random() < rate:
            op = rng.integers(4)
            if op == 0 and i + 1 < len(chars):  # inversion
                out += [chars[i + 1], c]
                i += 2
                continue
            if op == 1:  # suppression
                i += 1
                continue
            if op == 2:  # doublement
                out += [c, c]
            else:  # touche voisine
                nb = _KEYBOARD.get(c.lower(), c)
                out.append(nb[rng.integers(len(nb))])
        else:
            out.append(c)
        i += 1
    return "".join(out)


def token_noise(tokens, rate: float, vocab, seed: int = 0):
    """Perturbation à la BERT : chaque token est remplacé par un token aléatoire du
    vocabulaire avec la probabilité `rate`."""
    rng = np.random.default_rng(seed)
    vocab = list(vocab)
    return [vocab[rng.integers(len(vocab))] if rng.random() < rate else t for t in tokens]


_PROBLEM_TEMPLATES = [
    ("Une boulangerie a {a} croissants. Elle en vend {b}, puis en cuit {c} de plus. "
     "Combien de croissants a-t-elle maintenant ?", lambda a, b, c: a - b + c),
    ("Léa possède {a} billes. Elle en donne {b} à son frère et en gagne {c} à la récréation. "
     "Combien de billes a-t-elle ?", lambda a, b, c: a - b + c),
    ("Un train transporte {a} passagers. À la gare, {b} descendent et {c} montent. "
     "Combien de passagers y a-t-il dans le train ?", lambda a, b, c: a - b + c),
    ("Un magasin reçoit {c} cartons de {b} bouteilles et en avait déjà {a}. "
     "Combien de bouteilles a-t-il au total ?", lambda a, b, c: a + b * c),
    ("Paul lit {b} pages par jour pendant {c} jours, puis {a} pages le dernier jour. "
     "Combien de pages a-t-il lues en tout ?", lambda a, b, c: b * c + a),
    ("Une salle compte {c} rangées de {b} chaises. On retire {a} chaises abîmées. "
     "Combien reste-t-il de chaises ?", lambda a, b, c: b * c - a),
]


def make_word_problems(n: int, seed: int = 0):
    """Problèmes arithmétiques générés par gabarits, avec leur réponse exacte.

    Ce sont des données **vérifiables** : la réponse d'un modèle peut être contrôlée par un
    calcul (correction fonctionnelle). Renvoie une liste de dicts {question, answer, template}.
    """
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        t = int(rng.integers(len(_PROBLEM_TEMPLATES)))
        tpl, f = _PROBLEM_TEMPLATES[t]
        b = int(rng.integers(3, 15))
        c = int(rng.integers(2, 12))
        a = int(rng.integers(b + 1, 60))
        if f(a, b, c) < 0:  # « il reste -16 chaises » : on rejette les énoncés absurdes
            continue
        out.append({"question": tpl.format(a=a, b=b, c=c), "answer": int(f(a, b, c)), "template": t})
    return out


def extract_number(text: str):
    """Dernier nombre entier (éventuellement négatif) d'un texte, ou None."""
    nums = re.findall(r"-?\d+(?:[  ]\d{3})*", text.replace(",", " "))
    if not nums:
        return None
    return int(re.sub(r"[  ]", "", nums[-1]))


def self_instruct_filter(instruction: str, response: str, seen_instructions=(),
                         min_words: int = 3, max_words: int = 150, rouge_max: float = 0.7):
    """Heuristiques de filtrage inspirées de Self-Instruct (Wang et al., 2022).

    Renvoie (garder, raison). Règles : instruction trop courte ou trop longue, réponse
    vide, réponse qui répète l'instruction, instruction trop proche (recouvrement de mots
    > rouge_max) d'une instruction déjà retenue.
    """
    w = instruction.split()
    if len(w) < min_words:
        return False, "instruction trop courte"
    if len(w) > max_words:
        return False, "instruction trop longue"
    if not response.strip():
        return False, "réponse vide"
    if response.strip().lower() == instruction.strip().lower() or word_overlap(instruction, response) > 0.9:
        return False, "réponse = instruction"
    for s in seen_instructions:
        if word_overlap(instruction, s) > rouge_max:
            return False, "instruction quasi dupliquée"
    return True, "ok"


def word_overlap(a: str, b: str) -> float:
    """Recouvrement de mots façon ROUGE-L simplifié : F1 sur les ensembles de mots."""
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa or not wb:
        return 0.0
    inter = len(wa & wb)
    if inter == 0:
        return 0.0
    p, r = inter / len(wb), inter / len(wa)
    return 2 * p * r / (p + r)


# ======================================================================================
# 4. Effondrement de modèle (model collapse)
# ======================================================================================


def gaussian_collapse(n_samples: int = 50, n_generations: int = 200, real_frac: float = 0.0,
                      seed: int = 0, mu0: float = 0.0, sigma0: float = 1.0):
    """Chaîne « entraîner sur ses propres sorties » pour une gaussienne.

    À la génération t, on tire n_samples points : une fraction real_frac vient des vraies
    données N(mu0, sigma0^2), le reste du modèle courant N(mu_t, sigma_t^2) ; on réestime
    (mu_{t+1}, sigma_{t+1}) par maximum de vraisemblance. Renvoie (mus, sigmas) de
    longueur n_generations + 1.
    """
    rng = np.random.default_rng(seed)
    mus, sigmas = [mu0], [sigma0]
    n_real = int(round(real_frac * n_samples))
    for _ in range(n_generations):
        synth = rng.normal(mus[-1], sigmas[-1], size=n_samples - n_real)
        real = rng.normal(mu0, sigma0, size=n_real)
        data = np.concatenate([synth, real])
        mus.append(float(data.mean()))
        sigmas.append(float(data.std()))  # estimateur MV (biaisé, divise par n)
    return np.array(mus), np.array(sigmas)


def categorical_collapse(p0, n_samples: int = 1000, n_generations: int = 30, real_frac: float = 0.0,
                         temperature: float = 1.0, seed: int = 0):
    """Effondrement d'une distribution catégorielle (par ex. une loi de Zipf sur des
    « événements » du plus fréquent au plus rare).

    À chaque génération : on « génère » n_samples événements avec la distribution courante
    affûtée par la température (p^(1/T) renormalisée ; T < 1 imite un décodage qui favorise
    les événements probables), on mélange avec une fraction real_frac de vrais échantillons
    tirés de p0, puis on réestime les fréquences. Une catégorie qui tombe à zéro ne revient
    jamais sans données réelles. Renvoie la matrice (n_generations + 1, K) des distributions.
    """
    rng = np.random.default_rng(seed)
    p0 = np.asarray(p0, float) / np.sum(p0)
    hist = [p0]
    n_real = int(round(real_frac * n_samples))
    for _ in range(n_generations):
        p = hist[-1]
        q = np.zeros_like(p)
        q[p > 0] = p[p > 0] ** (1 / temperature)
        q /= q.sum()
        counts = rng.multinomial(n_samples - n_real, q) + rng.multinomial(n_real, p0)
        hist.append(counts / counts.sum())
    return np.array(hist)


# ======================================================================================
# 5. Distillation
# ======================================================================================


def kd_loss(student_logits, teacher_logits, T: float = 2.0) -> float:
    """Perte de distillation de Hinton et al. (2015) : T^2 * KL(p_T^teacher || p_T^student),
    moyennée sur les exemples, avec p_T = softmax(logits / T)."""
    from .llm import softmax

    pt = softmax(np.asarray(teacher_logits), T)
    ps = softmax(np.asarray(student_logits), T)
    kl = (pt * (np.log(pt + 1e-12) - np.log(ps + 1e-12))).sum(-1)
    return float(T ** 2 * kl.mean())


def train_mlp(X, targets, n_classes: int, hidden: int = 16, epochs: int = 300, lr: float = 0.01,
              weight_decay: float = 1e-4, seed: int = 0, batch_size: int = 256):
    """Entraîne un petit MLP (une couche cachée) sur CPU.

    targets : étiquettes entières (forme (n,)) ou distributions de probabilité « douces »
    (forme (n, n_classes)), auquel cas la perte est l'entropie croisée avec cibles douces.
    Renvoie une fonction predict_proba(X) -> np.ndarray.
    """
    import torch

    torch.manual_seed(seed)
    X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
    tg = np.asarray(targets)
    soft = tg.ndim == 2
    Y_t = torch.tensor(tg, dtype=torch.float32 if soft else torch.long)
    net = torch.nn.Sequential(torch.nn.Linear(X_t.shape[1], hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, n_classes))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(X_t), generator=g)
        for i in range(0, len(X_t), batch_size):
            idx = perm[i:i + batch_size]
            logits = net(X_t[idx])
            if soft:
                loss = -(Y_t[idx] * torch.log_softmax(logits, -1)).sum(-1).mean()
            else:
                loss = torch.nn.functional.cross_entropy(logits, Y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()

    def predict_proba(Xn):
        with torch.no_grad():
            return torch.softmax(net(torch.tensor(np.asarray(Xn), dtype=torch.float32)), -1).numpy()

    predict_proba.logits = lambda Xn: net(torch.tensor(np.asarray(Xn), dtype=torch.float32)).detach().numpy()
    return predict_proba


# ======================================================================================
# 6. Traitement : jeu de données sale, inspection, nettoyage
# ======================================================================================

#: paires (instruction, réponse) propres, rédigées à la main ; (langue, sujet, instruction, réponse)
BASE_PAIRS = [
    ("fr", "cuisine", "Comment réussir une pâte à crêpes sans grumeaux ?",
     "Versez la farine dans un saladier, creusez un puits, ajoutez les œufs puis le lait petit à petit en fouettant. Laissez reposer une heure."),
    ("fr", "cuisine", "Quelle est la différence entre une béchamel et un velouté ?",
     "La béchamel se fait avec un roux et du lait, le velouté avec un roux et un fond blanc (volaille, veau ou poisson)."),
    ("fr", "cuisine", "Propose un menu végétarien pour un dîner de quatre personnes.",
     "Entrée : velouté de potimarron. Plat : lasagnes aux épinards et ricotta. Dessert : tarte fine aux pommes."),
    ("fr", "informatique", "Écris une fonction Python qui inverse une chaîne de caractères.",
     "def inverser(s):\n    return s[::-1]"),
    ("fr", "informatique", "Qu'est-ce qu'une clé primaire dans une base de données ?",
     "C'est une colonne (ou un groupe de colonnes) dont la valeur identifie de façon unique chaque ligne d'une table."),
    ("fr", "informatique", "Explique la différence entre une liste et un tuple en Python.",
     "Une liste est modifiable (mutable) alors qu'un tuple ne l'est pas ; un tuple peut donc servir de clé de dictionnaire."),
    ("fr", "informatique", "Comment annuler le dernier commit Git sans perdre les modifications ?",
     "Utilisez git reset --soft HEAD~1 : le commit est annulé mais les fichiers restent modifiés et indexés."),
    ("fr", "santé", "Combien d'heures de sommeil recommande-t-on pour un adulte ?",
     "Entre sept et neuf heures par nuit pour la plupart des adultes, selon les recommandations usuelles."),
    ("fr", "santé", "Quels sont les bienfaits de la marche quotidienne ?",
     "Elle améliore la santé cardiovasculaire, aide à contrôler le poids, réduit le stress et renforce les os."),
    ("fr", "finance", "Qu'est-ce qu'un taux d'intérêt composé ?",
     "Les intérêts sont ajoutés au capital à chaque période et produisent eux-mêmes des intérêts : le capital croît de façon exponentielle."),
    ("fr", "finance", "Comment calculer un budget mensuel simple ?",
     "Listez vos revenus, retirez les dépenses fixes (loyer, abonnements), puis répartissez le reste entre dépenses variables et épargne."),
    ("fr", "voyage", "Quels documents faut-il pour voyager au Japon depuis la France ?",
     "Un passeport valide pour toute la durée du séjour ; pour un séjour touristique court, aucun visa n'est en général exigé."),
    ("fr", "voyage", "Donne trois conseils pour réduire le décalage horaire.",
     "Adaptez vos horaires de sommeil avant le départ, exposez-vous à la lumière du jour à l'arrivée et évitez les longues siestes."),
    ("fr", "éducation", "Comment expliquer les fractions à un enfant de huit ans ?",
     "Partez d'une pizza : si on la coupe en quatre parts égales et qu'on en mange une, on a mangé un quart."),
    ("fr", "éducation", "Résume la cause principale des saisons sur Terre.",
     "Les saisons viennent de l'inclinaison de l'axe de la Terre, qui change l'angle des rayons du Soleil au fil de l'année."),
    ("fr", "droit", "Quelle est la durée légale du travail en France ?",
     "La durée légale est de 35 heures par semaine ; au-delà, les heures sont des heures supplémentaires."),
    ("fr", "droit", "Peut-on résilier un abonnement téléphonique à tout moment ?",
     "Hors période d'engagement, oui ; pendant l'engagement, des frais de résiliation peuvent s'appliquer."),
    ("fr", "jardinage", "Quand faut-il planter des tomates ?",
     "Après les dernières gelées, en général de mi-avril à mi-mai selon les régions, quand le sol s'est réchauffé."),
    ("fr", "jardinage", "Comment lutter naturellement contre les pucerons ?",
     "Favorisez les coccinelles, pulvérisez de l'eau savonneuse ou plantez des capucines qui attirent les pucerons loin des cultures."),
    ("fr", "sport", "Comment débuter la course à pied sans se blesser ?",
     "Alternez marche et course, augmentez la durée de 10 % par semaine au plus et prévoyez des jours de repos."),
    ("en", "cooking", "How do I make a basic vinaigrette?",
     "Whisk three parts oil with one part vinegar, add a pinch of salt, pepper and a little mustard to emulsify."),
    ("en", "computing", "What is the difference between RAM and storage?",
     "RAM is fast, temporary memory used by running programs; storage keeps data permanently, even when the power is off."),
    ("en", "computing", "Write a SQL query that counts rows in a table named orders.",
     "SELECT COUNT(*) FROM orders;"),
    ("en", "health", "Why is staying hydrated important?",
     "Water regulates body temperature, carries nutrients and helps the kidneys remove waste."),
    ("en", "finance", "What is an index fund?",
     "A fund that tracks a market index such as the S&P 500, offering broad diversification at low cost."),
    ("en", "travel", "What should I pack for a weekend hiking trip?",
     "Layers of clothing, rain jacket, sturdy shoes, water, snacks, a map, a headlamp and a first-aid kit."),
    ("en", "education", "Explain photosynthesis in one sentence.",
     "Plants use sunlight, water and carbon dioxide to produce glucose and release oxygen."),
    ("es", "cocina", "¿Cómo se prepara una tortilla de patatas?",
     "Fríe las patatas y la cebolla, mézclalas con huevo batido y cuaja la mezcla en la sartén por ambos lados."),
    ("es", "viaje", "¿Qué ver en Madrid en un día?",
     "El Museo del Prado, el Parque del Retiro, la Plaza Mayor y el Palacio Real."),
    ("de", "kochen", "Wie koche ich ein weiches Ei?",
     "Lege das Ei in kochendes Wasser und lass es etwa sechs Minuten kochen, dann kalt abschrecken."),
]


_STOPWORDS = {
    "fr": set("le la les de des du un une et est en que qui pour dans ce il elle pas sur au aux avec "
              "se sont par plus ou comment quelle quels quel faut on vous votre nous je".split()),
    "en": set("the of and to a in is that for it on with as are be this was by or how what why "
              "you your do i an at from which should my".split()),
    "es": set("el la los las de del y en que un una por con para es se al lo como qué cómo ver".split()),
    "de": set("der die das und ist ich ein eine nicht zu mit den von wie auf dann es lass etwa".split()),
}


def detect_language(text: str) -> str:
    """Détection de langue par comptage de mots-outils (fr, en, es, de) ; 'inconnu' si aucun."""
    words = re.findall(r"[a-zà-öø-ÿœ]+", text.lower())
    scores = {lang: sum(w in sw for w in words) for lang, sw in _STOPWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "inconnu"


def _near_dup(text: str, rng) -> str:
    """Variante quasi identique : ponctuation, casse ou espaces, et souvent un mot remplacé
    ou supprimé (variante que la normalisation seule ne ramène pas à l'original)."""
    ops = list(rng.permutation(3)[:1]) + ([3] if rng.random() < 0.6 else [])
    t = text
    for op in ops:
        if op == 0:
            t = t.rstrip(" ?.!") + (" ?" if text.strip().endswith("?") else ".")
            t = t.replace(" ?", "?") if rng.random() < 0.5 else t
        elif op == 1:
            t = t[0].lower() + t[1:]
        elif op == 2:
            t = re.sub(" ", "  ", t, count=1)
        else:
            w = t.split(" ")
            if len(w) > 4:
                k = int(rng.integers(1, len(w) - 1))
                swap = {"une": "la", "un": "le", "la": "une", "le": "un", "des": "les",
                        "les": "des", "a": "the", "the": "a"}
                if w[k] in swap:
                    w[k] = swap[w[k]]
                else:
                    del w[k]
                t = " ".join(w)
    return t


def messy_instruction_dataset(seed: int = 0):
    """Jeu de données d'instructions « brut », volontairement sale, pour le cours 3.

    Construit à partir de BASE_PAIRS en y injectant des défauts réalistes : doublons exacts,
    quasi-doublons, balises HTML, données personnelles (PII), réponses vides, réponses qui
    répètent l'instruction, instructions trop courtes, même instruction avec réponses
    contradictoires. Colonnes : id, source, annotator, instruction, response, score, kind
    (le type de défaut injecté, inconnu en pratique : il sert à évaluer nos filtres).
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []

    def add(instr, resp, kind, source=None, annotator=None, score=None):
        rows.append(dict(source=source or rng.choice(["forum", "interne", "public_hf"], p=[.3, .4, .3]),
                         annotator=annotator or rng.choice(["A1", "A2", "A3"]),
                         instruction=instr, response=resp,
                         score=int(score if score is not None else rng.integers(3, 6)), kind=kind))

    for _, _, ins, rep in BASE_PAIRS:
        add(ins, rep, "propre")
    base = [(i, r) for _, _, i, r in BASE_PAIRS]
    for k in rng.choice(len(base), 8, replace=False):  # doublons exacts
        add(*base[k], "doublon_exact")
    for k in rng.choice(len(base), 10, replace=False):  # quasi-doublons
        add(_near_dup(base[k][0], rng), base[k][1], "quasi_doublon")
    for k in rng.choice(len(base), 6, replace=False):  # HTML
        i, r = base[k]
        add(i + " <br>", f"<div class=\"post\"><p>{html.escape(r)}</p>&nbsp;<a href=\"#\">Répondre</a></div>",
            "html", source="forum")
    pii = [
        ("Peux-tu reformuler ce message : « Bonjour, je suis Claire Martin, joignable au 06 12 34 56 78 » ?",
         "Bonjour, je m'appelle Claire Martin et vous pouvez me joindre au 06 12 34 56 78."),
        ("Rédige un courriel de relance pour la facture envoyée à jean.dupont@example.com.",
         "Bonjour, sauf erreur de notre part, la facture adressée à jean.dupont@example.com reste impayée."),
        ("Vérifie le format de cet IBAN : FR76 3000 6000 0112 3456 7890 189.",
         "Le format semble correct : FR76 suivi de 23 caractères."),
        ("Write a reminder for John Smith (john.smith@mail.com, +1 415 555 0134).",
         "Hi John, this is a reminder about tomorrow's meeting. Reach us at +1 415 555 0134."),
    ]
    for i, r in pii:
        add(i, r, "pii", source="interne")
    for k in rng.choice(len(base), 3, replace=False):
        add(base[k][0], "", "vide")
    for k in rng.choice(len(base), 3, replace=False):
        add(base[k][0], base[k][0], "echo")
    for txt in ["Aide", "ok", "Salut ?"]:
        add(txt, "Bonjour ! Comment puis-je vous aider ?", "trop_court")
    # même instruction, réponse contradictoire (conflit d'annotation)
    add(BASE_PAIRS[7][2], "Quatre heures suffisent largement pour un adulte en bonne santé.", "conflit",
        annotator="A3", score=5)
    add(BASE_PAIRS[15][2], "La durée légale est de 39 heures par semaine.", "conflit", annotator="A3", score=5)
    df = pd.DataFrame(rows)
    # l'annotateur A3 est « généreux » : il donne presque toujours 5
    df.loc[df.annotator == "A3", "score"] = np.where(rng.random((df.annotator == "A3").sum()) < 0.85, 5, 4)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "id", np.arange(len(df)))
    return df


_HTML_TAG = re.compile(r"<[^>]+>")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?:\+\d{1,3}[ .-]?)?(?:\(?\d{2,4}\)?[ .-]?){2,5}\d{2,4}")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")


def strip_html(text: str) -> str:
    """Retire les balises HTML, décode les entités (&amp;, &nbsp;...) et normalise les espaces."""
    t = _HTML_TAG.sub(" ", text)
    t = html.unescape(t).replace("\xa0", " ")
    t = re.sub(r"\bRépondre\b\s*$", "", t.strip())
    return re.sub(r"\s+", " ", t).strip()


def mask_pii(text: str):
    """Masque courriels, IBAN et numéros de téléphone. Renvoie (texte masqué, nb de masques).

    Expressions régulières volontairement simples : elles illustrent le principe et
    manquent des cas réels (noms propres, adresses), d'où l'intérêt de modèles de NER.
    """
    n = 0
    for pat, tag in [(_EMAIL, "[COURRIEL]"), (_IBAN, "[IBAN]"), (_PHONE, "[TÉLÉPHONE]")]:
        text, k = pat.subn(tag, text)
        n += k
    return text, n


def normalize_text(text: str) -> str:
    """Normalisation avant hachage : minuscules, espaces compactés, ponctuation finale ôtée."""
    t = re.sub(r"\s+", " ", text.lower()).strip()
    t = re.sub(r"\s+([?!.,;:])", r"\1", t)
    return t.rstrip(" ?!.")


def exact_hash(text: str, normalize: bool = True) -> str:
    """Empreinte SHA-1 (16 premiers caractères hexadécimaux) d'un texte, normalisé ou non."""
    t = normalize_text(text) if normalize else text
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:16]


def repetition_ratio(text: str, n: int = 3) -> float:
    """Part des n-grammes de mots qui sont des répétitions (0 = aucun n-gramme répété)."""
    w = text.lower().split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    if not grams:
        return 0.0
    return 1 - len(set(grams)) / len(grams)


# ======================================================================================
# 7. Quasi-doublons : shingles, Jaccard, MinHash, LSH
# ======================================================================================

MERSENNE = (1 << 31) - 1


def shingles(text: str, k: int = 5, unit: str = "char") -> set:
    """Ensemble des k-shingles d'un texte normalisé (k caractères ou k mots consécutifs)."""
    t = normalize_text(text)
    if unit == "word":
        w = t.split()
        return {" ".join(w[i:i + k]) for i in range(max(len(w) - k + 1, 1))}
    return {t[i:i + k] for i in range(max(len(t) - k + 1, 1))}


def jaccard(a: set, b: set) -> float:
    """Similarité de Jaccard |A inter B| / |A union B|."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class MinHasher:
    """Signatures MinHash par permutations universelles h_i(x) = (a_i x + b_i) mod p.

    Propriété clé : pour une permutation aléatoire h, P[min h(A) = min h(B)] = J(A, B).
    La proportion de composantes égales entre deux signatures est donc un estimateur sans
    biais de la similarité de Jaccard, d'écart-type sqrt(J(1-J)/num_perm).
    """

    def __init__(self, num_perm: int = 128, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, MERSENNE, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, MERSENNE, size=num_perm, dtype=np.uint64)

    @staticmethod
    def _base_hash(items) -> np.ndarray:
        return np.array([zlib.crc32(s.encode("utf-8")) % MERSENNE for s in items], dtype=np.uint64)

    def signature(self, items: set) -> np.ndarray:
        if not items:
            return np.full(self.num_perm, MERSENNE, dtype=np.uint64)
        x = self._base_hash(items)
        h = (self.a[:, None] * x[None, :] + self.b[:, None]) % np.uint64(MERSENNE)
        return h.min(axis=1)

    def signatures(self, sets) -> np.ndarray:
        return np.stack([self.signature(s) for s in sets])


def minhash_jaccard(sig1, sig2) -> float:
    """Jaccard estimé : proportion de composantes égales de deux signatures."""
    return float(np.mean(np.asarray(sig1) == np.asarray(sig2)))


def lsh_probability(s, bands: int, rows: int):
    """Probabilité qu'une paire de similarité s devienne candidate : 1 - (1 - s^r)^b."""
    s = np.asarray(s, float)
    return 1 - (1 - s ** rows) ** bands


def lsh_threshold(bands: int, rows: int) -> float:
    """Seuil approximatif (point d'inflexion de la courbe en S) : (1/b)^(1/r)."""
    return (1 / bands) ** (1 / rows)


def lsh_optimal_params(num_perm: int, threshold: float, fp_weight: float = 0.5,
                       fn_weight: float = 0.5):
    """(bands, rows) avec bands * rows <= num_perm minimisant l'aire pondérée des faux
    positifs (s < seuil mais candidate) et des faux négatifs (s >= seuil mais manquée)."""
    best, best_cost = None, np.inf
    for b in range(1, num_perm + 1):
        r = num_perm // b
        if r == 0:
            break
        s_lo = np.linspace(0, threshold, 200)
        s_hi = np.linspace(threshold, 1, 200)
        fp = np.trapezoid(lsh_probability(s_lo, b, r), s_lo)
        fn = np.trapezoid(1 - lsh_probability(s_hi, b, r), s_hi)
        cost = fp_weight * fp + fn_weight * fn
        if cost < best_cost:
            best, best_cost = (b, r), cost
    return best


def lsh_candidates(signatures, bands: int, rows: int) -> set:
    """Paires candidates (i < j) : documents partageant au moins un seau dans une bande.

    Chaque signature est découpée en `bands` bandes de `rows` composantes ; chaque bande est
    hachée dans un dictionnaire. Coût ~ O(n * bands) au lieu de O(n^2) comparaisons.
    """
    sig = np.asarray(signatures)
    cands = set()
    for band in range(bands):
        buckets = defaultdict(list)
        block = sig[:, band * rows:(band + 1) * rows]
        for i, row in enumerate(block):
            buckets[row.tobytes()].append(i)
        for ids in buckets.values():
            if len(ids) > 1:
                cands.update(itertools.combinations(ids, 2))
    return cands


def connected_groups(n: int, pairs) -> list:
    """Composantes connexes (union-find) : groupes de documents reliés par des paires."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [g for g in groups.values()]


def make_near_duplicate_corpus(n_docs: int = 1000, doc_len: int = 60, dup_frac: float = 0.2,
                               edit_rate: float = 0.1, vocab_size: int = 3000, seed: int = 0):
    """Corpus synthétique de « documents » (mots tirés d'une loi de Zipf) dont une fraction
    `dup_frac` sont des copies modifiées (une fraction `edit_rate` des mots remplacés)
    d'autres documents. Renvoie (docs, true_pairs) avec true_pairs l'ensemble des (i, j)."""
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(vocab_size)]
    ranks = np.arange(1, vocab_size + 1)
    p = 1 / ranks
    p /= p.sum()
    n_orig = int(round(n_docs * (1 - dup_frac)))
    docs = [list(rng.choice(vocab, size=doc_len, p=p)) for _ in range(n_orig)]
    pairs = set()
    for _ in range(n_docs - n_orig):
        src = int(rng.integers(n_orig))
        d = list(docs[src])
        for k in rng.choice(doc_len, size=int(round(edit_rate * doc_len)), replace=False):
            d[k] = vocab[rng.integers(vocab_size)]
        docs.append(d)
        pairs.add((src, len(docs) - 1))
    return [" ".join(d) for d in docs], pairs


# ======================================================================================
# 8. Mise au format : chat template et masquage de la perte
# ======================================================================================


def to_messages(instruction: str, response: str, system: str | None = None) -> list:
    """(instruction, réponse) -> liste de messages au format conversationnel."""
    msgs = [{"role": "system", "content": system}] if system else []
    return msgs + [{"role": "user", "content": instruction},
                   {"role": "assistant", "content": response}]


def build_sft_example(tok, messages, mask_prompt: bool = True) -> dict:
    """Tokenise une conversation avec le chat template et construit les étiquettes SFT.

    labels[i] = input_ids[i] pour les tokens des réponses de l'assistant (y compris le
    marqueur de fin de tour), -100 ailleurs (system, user, balises) : la perte n'est calculée
    que sur ce que le modèle doit apprendre à produire. Renvoie un dict
    {text, input_ids, labels}.
    """
    full = tok.apply_chat_template(messages, tokenize=False)
    ids = tok(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(ids) if mask_prompt else list(ids)
    if mask_prompt:
        for i, m in enumerate(messages):
            if m["role"] != "assistant":
                continue
            prefix = tok.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True)
            upto = tok.apply_chat_template(messages[:i + 1], tokenize=False)
            a = len(tok(prefix, add_special_tokens=False)["input_ids"])
            b = len(tok(upto, add_special_tokens=False)["input_ids"])
            for k in range(a, min(b, len(ids))):
                labels[k] = ids[k]
    return {"text": full, "input_ids": ids, "labels": labels}


def fewshot_to_examples(prompt: str, item_key: str = "Item:", label_key: str = "Label:",
                        sep: str = " -->"):
    """Convertit les exemples d'un prompt few-shot (« Item: x / Label: y ») en paires
    d'entraînement (« x --> », « y »), comme dans la section *Format Data*."""
    items = re.findall(rf"{re.escape(item_key)}\s*(.+)\n\s*{re.escape(label_key)}\s*(\S.*)", prompt)
    return [(f"{x.strip()}{sep}", y.strip()) for x, y in items if "{" not in x]


__all__ = [n for n in dir() if not n.startswith("_") and n not in
           ("annotations", "hashlib", "html", "itertools", "re", "zlib", "Counter", "defaultdict", "np")]
