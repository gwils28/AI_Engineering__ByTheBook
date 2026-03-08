"""Chapitre 10 : architecture d'une application d'IA et retours utilisateurs.

Ce module rassemble les briques réutilisées par les notebooks du chapitre 10 :

- ``architecture_diagram(step)`` : le diagramme d'architecture qui s'enrichit étape par étape ;
- garde-fous d'entrée (``detect_pii``, ``mask_pii``, ``unmask``, ``iban_valide``,
  ``detect_injection``) et de sortie (``validate_json``, ``toxicity_score``, ``is_refusal``,
  ``OutputGuard``, ``retry_until_valid``) ;
- routeur d'intention par embeddings (``IntentRouter``, jeu ``INTENTS_TRAIN`` / ``INTENTS_TEST``) ;
- passerelle de modèles (``TokenBucket``, ``Provider``, ``ModelGateway``) ;
- caches exact et sémantique (``ExactCache``, ``SemanticCache``, ``semantic_threshold_tradeoff``) ;
- la mini-application ``MiniApp`` qui grandit avec l'architecture (étapes 0 à 6) ;
- observabilité (``Span``, ``plot_trace``, ``simulate_production``, ``cusum``, ``devops_metrics``) ;
- orchestration (``Pipeline``) ;
- retours utilisateurs (``extract_signals``, ``CONVERSATIONS``, simulations de biais et de
  boucles de rétroaction dégénérées).

Toutes les fonctions acceptent des callables injectables (modèle de langage, embeddings) :
les tests tournent ainsi sans GPU avec ``hash_embed`` et de faux modèles.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .plotting import AMBER, GRAY, GREEN, INDIGO, ROSE, SKY, SLATE, TEAL, VIOLET

# =====================================================================================
# 0. Outils généraux
# =====================================================================================


def strip_accents(text: str) -> str:
    """Supprime les accents (é -> e) pour rendre les règles robustes aux variantes."""
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def normalize(text: str) -> str:
    """Minuscules, sans accents, ponctuation retirée, espaces simples."""
    t = strip_accents(text.lower())
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def hash_embed(texts, dim: int = 512) -> np.ndarray:
    """Embedding lexical bon marché : sac de trigrammes de caractères haché, normalisé.

    Ne capture aucune sémantique (seulement le recouvrement de caractères) : sert de
    référence « naïve » face aux vrais embeddings et de substitut rapide dans les tests.
    """
    import zlib

    if isinstance(texts, str):
        texts = [texts]
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        t = f"  {normalize(t)}  "
        for j in range(len(t) - 2):
            out[i, zlib.crc32(t[j:j + 3].encode()) % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)


def n_tokens(text: str) -> int:
    """Estimation grossière du nombre de tokens (≈ 1,3 token par mot en français)."""
    return max(1, int(round(1.3 * len(text.split()))))


# =====================================================================================
# 1. Diagramme d'architecture évolutif
# =====================================================================================

#: étapes de l'architecture (section *AI Engineering Architecture*)
STEPS = {
    0: "Architecture minimale : requête -> modèle -> réponse",
    1: "Étape 1 : enrichir le contexte (RAG, outils de lecture)",
    2: "Étape 2 : garde-fous d'entrée et de sortie",
    3: "Étape 3 : routeur et passerelle de modèles",
    4: "Étape 4 : caches (exact et sémantique)",
    5: "Étape 5 : motifs agentiques (boucles)",
    6: "Étape 6 : actions d'écriture",
    7: "Vue complète : observabilité et orchestration",
}

# nom -> (étape d'apparition, x, y, texte, sous-texte, couleur, kind, largeur)
_COMPONENTS = {
    "user": (0, 0.9, 3.5, "Utilisateur", None, INDIGO, "user", 1.25),
    "in_guard": (2, 3.0, 3.5, "Garde-fous", "entrée : PII, injection", ROSE, "box", 1.95),
    "router": (3, 5.3, 3.5, "Routeur", "intention", VIOLET, "box", 1.75),
    "gateway": (0, 7.6, 3.5, "API du modèle", "Qwen2.5-0.5B", TEAL, "model", 2.0),
    "out_guard": (2, 10.0, 3.5, "Garde-fous", "sortie : format, refus", ROSE, "box", 1.95),
    "reply": (0, 12.5, 3.5, "Réponse", None, INDIGO, "box", 1.6),
    "context": (1, 7.6, 5.7, "Contexte", "RAG + outils lecture", SKY, "box", 2.0),
    "docs": (1, 10.3, 5.7, "Documents", "base vectorielle", SLATE, "db", 1.8),
    "cache": (4, 3.0, 5.7, "Caches", "exact / sémantique", AMBER, "db", 1.9),
    "m_small": (3, 6.3, 1.45, "Qwen 0.5B", "rapide", TEAL, "model", 1.6),
    "m_big": (3, 8.25, 1.45, "Qwen 1.5B", "meilleur", TEAL, "model", 1.6),
    "scorer": (2, 10.25, 1.45, "Scoreurs", "petits modèles", ROSE, "model", 1.6),
    "actions": (6, 12.5, 1.45, "Actions", "écriture : email, commande", GREEN, "box", 2.0),
}


def architecture_diagram(step: int = 0, highlight_new: bool = True, show: bool = True):
    """Dessine l'architecture du chapitre 10 telle qu'elle est à l'étape ``step`` (0 à 7).

    Les composants apparus à cette étape sont mis en avant (fond plein) ; une pastille
    indique l'étape d'apparition de chaque composant. Renvoie le ``Diagram``.
    """
    from .diagrams import Diagram

    d = Diagram(14, 7.0, title="Architecture d'une application d'IA", subtitle=STEPS[step])
    present = {k for k, v in _COMPONENTS.items() if v[0] <= step}

    if step >= 3:
        d.group(4.4, 0.88, 11.3, 2.48, "modèles", color=TEAL)
    if step >= 7:
        d.group(0.15, 0.1, 13.85, 0.75, None, color=SLATE, alpha=0.5)
        d.note(7.0, 0.42, "Observabilité : métriques, journaux, traces à chaque composant ;"
               " un orchestrateur enchaîne le tout", color=SLATE, style="normal", fontsize=9.5)

    for name in _COMPONENTS:
        if name not in present:
            continue
        s, x, y, text, sub, color, kind, w = _COMPONENTS[name]
        if name == "gateway" and step >= 3:
            text, sub = "Passerelle", "repli, quotas, logs"
        is_new = highlight_new and s == step and step > 0
        d.box(name, x, y, text, sub=sub, color=color, kind=kind, w=w, h=0.9 if kind != "db" else 1.0,
              filled=is_new and kind not in ("user", "db"), fontsize=10)
        if kind != "user" and step > 0:
            if name == "gateway" and step >= 3:
                s = 3
            d.badge(x - w / 2 + 0.05, y + 0.5, str(s), color=color if s > 0 else GRAY)

    # chaîne principale
    chain = [c for c in ["user", "in_guard", "router", "gateway", "out_guard", "reply"] if c in present]
    for a, b in zip(chain[:-1], chain[1:]):
        lab = {("user", "gateway"): "requête", ("gateway", "reply"): "texte"}.get((a, b))
        d.arrow(a, b, lab)
    if "context" in present:
        prev = chain[chain.index("gateway") - 1]
        d.arrow(prev, "context", "requête" if step < 3 else None, color=SKY, rad=-0.12 if prev != "router" else 0)
        d.arrow("context", "gateway", "contexte", color=SKY)
        d.arrow("docs", "context", "lecture", color=SLATE, style="dashed")
    if "m_small" in present:
        d.arrow("gateway", "m_small", color=TEAL)
        d.arrow("gateway", "m_big", color=TEAL)
        d.arrow("out_guard", "scorer", None, color=ROSE, style="dashed", both=True)
    elif "scorer" in present:
        d.arrow("out_guard", "scorer", None, color=ROSE, style="dashed", both=True)
    if "cache" in present:
        d.arrow("in_guard", "cache", "lecture", color=AMBER, style="dashed")
        d.line([(3.0, 6.25), (3.0, 6.75), (12.5, 6.75), (12.5, 4.0)], color=AMBER, style="dashed", lw=1.4)
        d.note(9.0, 6.95, "succès du cache : réponse immédiate", color=AMBER, fontsize=8.5)
        d.line([(10.0, 3.05), (10.0, 2.68), (4.2, 2.68), (3.6, 5.1)], color=AMBER, style="dotted", lw=1.4)
        d.note(4.75, 2.85, "écriture en cache", color=AMBER, fontsize=8)
    if step >= 5:
        d.arrow("out_guard", "context", "boucle : réponse incomplète", color=AMBER, rad=0.25, lw=2.2,
                label_offset=(0.9, -0.15))
    if "actions" in present:
        d.arrow("out_guard", "actions", "action validée", color=GREEN, lw=2.0, label_offset=(0.55, 0.1))
    if show:
        d.show()
    return d


# =====================================================================================
# 2. Garde-fous d'entrée : données personnelles et injection de prompt
# =====================================================================================

#: motifs de données personnelles (l'ordre compte : IBAN avant téléphone)
PII_PATTERNS: dict[str, str] = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "IBAN": r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b",
    "TEL": r"(?<![\w+])(?:(?:\+|00)33[ .-]?[1-9]|0[1-9])(?:[ .-]?\d{2}){4}(?!\d)",
}


def iban_valide(iban: str) -> bool:
    """Vérifie la clé de contrôle d'un IBAN (norme ISO 13616, calcul modulo 97).

    On déplace les 4 premiers caractères à la fin, on remplace chaque lettre par un nombre
    (A = 10, ..., Z = 35) et l'IBAN est valide si l'entier obtenu vaut 1 modulo 97.
    """
    s = re.sub(r"\s", "", iban).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", s):
        return False
    r = s[4:] + s[:4]
    digits = "".join(str(int(c, 36)) for c in r)
    return int(digits) % 97 == 1


def iban_avec_cle(pays: str, bban: str) -> str:
    """Calcule la clé de contrôle et renvoie l'IBAN complet (utile pour fabriquer des exemples)."""
    s = bban.upper() + pays.upper() + "00"
    cle = 98 - int("".join(str(int(c, 36)) for c in s)) % 97
    return f"{pays.upper()}{cle:02d}{bban.upper()}"


@dataclass
class PIIMatch:
    kind: str
    start: int
    end: int
    value: str


def detect_pii(text: str, patterns: dict[str, str] | None = None, check_iban: bool = True) -> list[PIIMatch]:
    """Détecte les données personnelles d'un texte. Les zones déjà reconnues ne sont pas
    réexaminées (un IBAN n'est donc pas aussi compté comme téléphone)."""
    patterns = patterns or PII_PATTERNS
    found: list[PIIMatch] = []
    taken = np.zeros(len(text) + 1, dtype=bool)
    for kind, pat in patterns.items():
        for m in re.finditer(pat, text):
            if taken[m.start():m.end()].any():
                continue
            if kind == "IBAN" and check_iban and not iban_valide(m.group()):
                continue
            found.append(PIIMatch(kind, m.start(), m.end(), m.group()))
            taken[m.start():m.end()] = True
    return sorted(found, key=lambda m: m.start)


def mask_pii(text: str, patterns: dict[str, str] | None = None) -> tuple[str, dict[str, str]]:
    """Remplace chaque donnée personnelle par un jeton ``[EMAIL_1]``, ``[TEL_1]``...

    Renvoie (texte masqué, dictionnaire inverse jeton -> valeur) : ce dictionnaire reste
    dans l'organisation et permet de démasquer la réponse du modèle (``unmask``).
    """
    matches = detect_pii(text, patterns)
    reverse: dict[str, str] = {}
    by_value: dict[str, str] = {}
    counters: dict[str, int] = {}
    out, last = [], 0
    for m in matches:
        if m.value not in by_value:
            counters[m.kind] = counters.get(m.kind, 0) + 1
            tag = f"[{m.kind}_{counters[m.kind]}]"
            by_value[m.value] = tag
            reverse[tag] = m.value
        out.append(text[last:m.start])
        out.append(by_value[m.value])
        last = m.end
    out.append(text[last:])
    return "".join(out), reverse


def unmask(text: str, reverse: dict[str, str]) -> str:
    """Réinjecte les valeurs d'origine à la place des jetons dans une réponse."""
    for tag, value in reverse.items():
        text = text.replace(tag, value)
    return text


#: motifs (sur texte normalisé, sans accents) typiques des tentatives d'injection
INJECTION_PATTERNS: list[str] = [
    r"ignore[rsz]? (toutes |tous )?(les |tes |vos )?(instructions|consignes|regles)",
    r"oublie[rsz]? (tout|toutes|tes|vos|les) ?(instructions|consignes|regles)?",
    r"ignore (all |the )?(previous |prior )?instructions",
    r"(revele|affiche|donne|montre)[rsz]?( moi)? (ton|le|tes|ta) (prompt|message|instructions?|consigne)s? (systeme|initial)",
    r"system prompt",
    r"tu es (maintenant|desormais) ",
    r"(mode|role) (developpeur|admin|administrateur|sans limite|sans restriction)",
    r"sans (aucune )?(restriction|limite|filtre|censure)",
    r"\bdan\b|jailbreak",
    r"fais comme si tu (n avais|navais) (pas|aucune)",
]


def detect_injection(text: str, patterns: list[str] | None = None) -> list[str]:
    """Renvoie la liste des motifs d'injection reconnus (vide si rien de suspect)."""
    t = normalize(text)
    return [p for p in (patterns or INJECTION_PATTERNS) if re.search(p, t)]


# =====================================================================================
# 3. Garde-fous de sortie
# =====================================================================================


def validate_json(text: str, required_keys=()) -> tuple[bool, str, dict | None]:
    """Vérifie qu'une sortie contient un objet JSON valide avec les clés attendues.

    Renvoie (ok, raison, objet). Tolère du texte autour de l'objet (fréquent avec les LLM).
    """
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return False, "aucun objet JSON", None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return False, "JSON invalide", None
    if not isinstance(obj, dict):
        return False, "pas un objet", None
    missing = [k for k in required_keys if k not in obj]
    if missing:
        return False, f"clés manquantes : {missing}", obj
    return True, "ok", obj


#: petit lexique de toxicité (volontairement modéré) pour la démonstration
TOXIC_WORDS = ["idiot", "idiote", "imbecile", "cretin", "debile", "stupide", "abruti", "abrutie",
               "nul", "nulle", "ferme la", "la ferme", "degage", "minable", "pathetique", "con", "conne"]


def toxicity_score(text: str, lexicon=None) -> float:
    """Part des mots du texte appartenant au lexique toxique (0 = propre)."""
    t = f" {normalize(text)} "
    words = max(1, len(t.split()))
    hits = sum(t.count(f" {w} ") for w in (lexicon or TOXIC_WORDS))
    return hits / words


REFUSAL_PATTERNS = [
    r"je ne (peux|pourrai|suis pas en mesure)",
    r"je suis desole",
    r"en tant qu (ia|intelligence artificielle|assistant|modele)",
    r"i (cannot|can t|am unable)", r"i m sorry", r"as an ai",
    r"je n ai pas (acces|la possibilite)",
]


def is_refusal(text: str) -> bool:
    """Détecte un refus (« je ne peux pas », « en tant qu'IA »...)."""
    t = normalize(text)
    return any(re.search(p, t) for p in REFUSAL_PATTERNS)


@dataclass
class OutputGuard:
    """Contrôle de sortie : vide, format JSON, toxicité, refus, fuite de données personnelles."""

    json_keys: tuple | None = None
    max_toxicity: float = 0.0
    block_refusal: bool = True
    block_pii: bool = True

    def check(self, text: str) -> list[str]:
        fails = []
        if not text.strip():
            return ["vide"]
        if self.json_keys is not None:
            ok, why, _ = validate_json(text, self.json_keys)
            if not ok:
                fails.append(f"format ({why})")
        if toxicity_score(text) > self.max_toxicity:
            fails.append("toxicité")
        if self.block_refusal and is_refusal(text):
            fails.append("refus")
        if self.block_pii and detect_pii(text):
            fails.append("donnée personnelle")
        return fails


def retry_until_valid(generate: Callable[[int], str], check: Callable[[str], list[str]],
                      max_tries: int = 3) -> tuple[str | None, int, list[tuple[str, list[str]]]]:
    """Politique de relance : rappelle ``generate(tentative)`` tant que ``check`` signale un échec.

    Renvoie (sortie valide ou None, nombre d'appels, historique [(sortie, échecs)]).
    """
    history = []
    for i in range(max_tries):
        out = generate(i)
        fails = check(out)
        history.append((out, fails))
        if not fails:
            return out, i + 1, history
    return None, max_tries, history


def retry_stats(p: float, n_max: int, latency: float = 1.0, parallel: bool = False) -> dict:
    """Probabilité de succès, nombre d'appels et latence attendus d'une politique de relance.

    Chaque appel réussit indépendamment avec probabilité ``p``.
    - séquentiel : on relance après chaque échec, jusqu'à ``n_max`` appels ;
    - parallèle : on lance les ``n_max`` appels en même temps et on garde un succès.
    """
    q = 1 - p
    success = 1 - q ** n_max
    if parallel:
        calls, lat = float(n_max), latency
    else:
        # E[appels] = somme_{k=0}^{n-1} q^k (on fait l'appel k+1 si les k premiers ont échoué)
        calls = sum(q ** k for k in range(n_max))
        lat = latency * calls
    return {"p_succes": success, "appels_moyens": calls, "latence_moyenne": lat}


# =====================================================================================
# 4. Routeur d'intention
# =====================================================================================

#: intentions d'un assistant de service client (boutique fictive « ÉlectroNova »)
INTENTS_TRAIN: dict[str, list[str]] = {
    "mot_de_passe": [
        "J'ai oublié mon mot de passe", "Comment réinitialiser mon mot de passe ?",
        "Je n'arrive plus à me connecter à mon compte", "Mon mot de passe ne fonctionne plus",
        "Je voudrais changer mon mot de passe", "Impossible de me connecter, identifiants refusés",
        "Le lien de réinitialisation ne marche pas", "Mon compte est bloqué après plusieurs essais",
        "Comment récupérer l'accès à mon compte ?", "Je ne reçois pas l'email pour changer le mot de passe",
    ],
    "facturation": [
        "J'ai été débité deux fois", "Ma facture est incorrecte", "Je veux un remboursement",
        "Le montant prélevé ne correspond pas à ma commande", "Où trouver ma facture ?",
        "On m'a facturé des frais de livraison alors qu'elle était gratuite",
        "Je conteste un prélèvement sur ma carte", "Pouvez-vous corriger l'adresse sur ma facture ?",
        "Le remboursement n'est toujours pas arrivé", "Erreur de prix sur mon ticket de caisse",
    ],
    "technique": [
        "Mon casque Bluetooth ne s'appaire pas", "L'ordinateur portable ne s'allume plus",
        "La télévision n'a pas de son", "Mon téléphone surchauffe pendant la charge",
        "L'imprimante affiche une erreur de bourrage", "Le Wi-Fi de la box se coupe sans arrêt",
        "La batterie se décharge très vite", "L'écran clignote quand je lance un jeu",
        "La mise à jour du logiciel a échoué", "Ma montre connectée ne synchronise plus les données",
    ],
    "commande": [
        "Où en est ma commande ?", "Je veux annuler ma commande", "Ma livraison est en retard",
        "Puis-je modifier l'adresse de livraison ?", "Quand vais-je recevoir mon colis ?",
        "Le colis indiqué livré n'est jamais arrivé", "Je souhaite retourner un article",
        "Comment suivre mon colis ?", "Il manque un article dans mon colis",
        "Puis-je ajouter un produit à une commande déjà passée ?",
    ],
    "hors_perimetre": [
        "Pour qui vas-tu voter aux prochaines élections ?", "Quel temps fera-t-il demain ?",
        "Raconte-moi une blague", "Quelle est la capitale de l'Australie ?",
        "Écris-moi un poème sur la mer", "Que penses-tu du dernier film de science-fiction ?",
        "Donne-moi une recette de gâteau au chocolat", "Qui a gagné le match hier soir ?",
        "Quel est le sens de la vie ?", "Aide-moi à faire mes devoirs de maths",
    ],
}

INTENTS_TEST: list[tuple[str, str]] = [
    ("Je ne me souviens plus de mes identifiants", "mot_de_passe"),
    ("Le code de connexion est refusé à chaque fois", "mot_de_passe"),
    ("Comment modifier mon mot de passe sur l'application ?", "mot_de_passe"),
    ("Mon accès client est verrouillé", "mot_de_passe"),
    ("Vous m'avez prélevé deux fois la même somme", "facturation"),
    ("Je n'ai pas reçu mon avoir", "facturation"),
    ("La TVA sur ma facture est fausse", "facturation"),
    ("Je voudrais être remboursé de ma commande annulée", "facturation"),
    ("Mes écouteurs ne se connectent pas au téléphone", "technique"),
    ("Le lave-linge affiche le code erreur E21", "technique"),
    ("Ma tablette redémarre toute seule", "technique"),
    ("Le chargeur ne charge plus l'ordinateur", "technique"),
    ("Mon colis n'est toujours pas là", "commande"),
    ("Je souhaite annuler l'achat que j'ai fait hier", "commande"),
    ("Pouvez-vous livrer chez mon voisin ?", "commande"),
    ("Je veux renvoyer la cafetière, elle ne me plaît pas", "commande"),
    ("Tu préfères les chats ou les chiens ?", "hors_perimetre"),
    ("Combien mesure la tour Eiffel ?", "hors_perimetre"),
    ("Traduis cette phrase en espagnol", "hors_perimetre"),
    ("Quel est ton avis sur le gouvernement ?", "hors_perimetre"),
]

#: solution (ou destination) associée à chaque intention (section *Router*)
ROUTES = {
    "mot_de_passe": "page FAQ",
    "facturation": "opérateur humain",
    "technique": "chatbot dépannage (LLM + RAG)",
    "commande": "agent commandes (outils)",
    "hors_perimetre": "réponse type polie",
    "ambigu": "demande de clarification",
}


class IntentRouter:
    """Classifieur d'intention par embeddings : centroïde ou k plus proches voisins.

    - ``method="centroid"`` : score = similarité cosinus au centroïde de chaque intention ;
    - ``method="knn"`` : score = similarité moyenne des k exemples les plus proches de l'intention.
    Si le meilleur score est sous ``threshold`` ou trop proche du second (``margin``),
    la requête est déclarée « ambigu » (on demande une clarification).
    """

    def __init__(self, embed_fn: Callable, method: str = "centroid", k: int = 3,
                 threshold: float = 0.0, margin: float = 0.0):
        self.embed_fn, self.method, self.k = embed_fn, method, k
        self.threshold, self.margin = threshold, margin

    def fit(self, data: dict[str, list[str]]):
        self.labels = list(data)
        texts = [t for lab in self.labels for t in data[lab]]
        self.y = np.array([lab for lab in self.labels for _ in data[lab]])
        self.X = np.asarray(self.embed_fn(texts), dtype=np.float32)
        C = np.stack([self.X[self.y == lab].mean(0) for lab in self.labels])
        self.C = C / np.linalg.norm(C, axis=1, keepdims=True)
        return self

    def scores(self, texts) -> np.ndarray:
        """Matrice (n_textes, n_intentions) des scores de similarité."""
        if isinstance(texts, str):
            texts = [texts]
        E = np.asarray(self.embed_fn(list(texts)), dtype=np.float32)
        if self.method == "centroid":
            return E @ self.C.T
        S = E @ self.X.T
        out = np.zeros((len(E), len(self.labels)))
        for j, lab in enumerate(self.labels):
            s = np.sort(S[:, self.y == lab], axis=1)[:, ::-1][:, : self.k]
            out[:, j] = s.mean(1)
        return out

    def predict(self, texts) -> list[str]:
        S = self.scores(texts)
        order = np.argsort(-S, axis=1)
        res = []
        for i in range(len(S)):
            best, second = S[i, order[i, 0]], S[i, order[i, 1]]
            if best < self.threshold or best - second < self.margin:
                res.append("ambigu")
            else:
                res.append(self.labels[order[i, 0]])
        return res


# =====================================================================================
# 5. Passerelle de modèles
# =====================================================================================


class TokenBucket:
    """Limiteur de débit « seau à jetons » : ``capacity`` jetons, recharge de ``rate`` jetons/s.

    Chaque requête consomme un jeton ; sans jeton disponible, elle est refusée (HTTP 429).
    Le débit soutenu est ``rate`` requêtes/s, les rafales peuvent atteindre ``capacity``.
    """

    def __init__(self, capacity: float, rate: float):
        self.capacity, self.rate = capacity, rate
        self.tokens, self.t = float(capacity), 0.0

    def allow(self, now: float, cost: float = 1.0) -> bool:
        self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
        self.t = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


class ProviderError(RuntimeError):
    pass


@dataclass
class Provider:
    """Un fournisseur de modèle derrière la passerelle (API externe ou modèle auto-hébergé).

    ``fn(messages) -> str`` produit le texte ; ``fail_rate`` simule des pannes ;
    la latence simulée vaut ``base_ms + ms_per_token * tokens_sortie``.
    """

    name: str
    fn: Callable
    price_in: float = 0.0     # $ par million de tokens d'entrée
    price_out: float = 0.0    # $ par million de tokens de sortie
    fail_rate: float = 0.0
    base_ms: float = 300.0
    ms_per_token: float = 20.0
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, messages):
        if self.rng.random() < self.fail_rate:
            raise ProviderError(f"{self.name} : erreur 503 (service indisponible)")
        return self.fn(messages)


@dataclass
class GatewayResult:
    text: str | None
    provider: str | None
    attempts: list
    cost: float
    latency_ms: float
    status: str


class ModelGateway:
    """Passerelle minimale : interface unifiée, contrôle d'accès, quotas, repli, journalisation."""

    def __init__(self, providers: list[Provider], rate_limit: tuple[float, float] | None = None,
                 access: dict[str, set] | None = None, retry_wait_ms: float = 200.0):
        self.providers = {p.name: p for p in providers}
        self.order = [p.name for p in providers]
        self.rate_limit = rate_limit
        self.buckets: dict[str, TokenBucket] = {}
        self.access = access
        self.retry_wait_ms = retry_wait_ms
        self.logs: list[dict] = []

    def call(self, messages, app: str = "default", now: float = 0.0, models: list[str] | None = None) -> GatewayResult:
        models = models or self.order
        if self.access is not None:
            models = [m for m in models if m in self.access.get(app, set())]
            if not models:
                return self._log(app, now, messages, GatewayResult(None, None, [], 0, 0, "403 accès refusé"))
        if self.rate_limit is not None:
            b = self.buckets.setdefault(app, TokenBucket(*self.rate_limit))
            if not b.allow(now):
                return self._log(app, now, messages, GatewayResult(None, None, [], 0, 1.0, "429 quota dépassé"))
        attempts, lat = [], 0.0
        prompt = " ".join(m["content"] for m in messages) if not isinstance(messages, str) else messages
        for name in models:
            p = self.providers[name]
            try:
                text = p(messages)
            except ProviderError as e:
                attempts.append((name, str(e)))
                lat += p.base_ms + self.retry_wait_ms
                continue
            t_in, t_out = n_tokens(prompt), n_tokens(text)
            lat += p.base_ms + p.ms_per_token * t_out
            cost = (t_in * p.price_in + t_out * p.price_out) / 1e6
            attempts.append((name, "ok"))
            return self._log(app, now, messages, GatewayResult(text, name, attempts, cost, lat, "200 ok"))
        return self._log(app, now, messages, GatewayResult(None, None, attempts, 0.0, lat, "502 tous en échec"))

    def _log(self, app, now, messages, res: GatewayResult) -> GatewayResult:
        self.logs.append({"t": now, "app": app, "statut": res.status, "fournisseur": res.provider,
                          "tentatives": len(res.attempts), "latence_ms": round(res.latency_ms, 1),
                          "cout_$": res.cost})
        return res


# =====================================================================================
# 6. Caches
# =====================================================================================


class ExactCache:
    """Cache exact avec clé normalisée et politique d'éviction ``"lru"`` (moins récemment
    utilisé) ou ``"fifo"`` (premier entré, premier sorti)."""

    def __init__(self, capacity: int = 1000, normalize_key: bool = True, policy: str = "lru"):
        self.capacity, self.normalize_key, self.policy = capacity, normalize_key, policy
        self.store: OrderedDict = OrderedDict()
        self.hits = self.misses = self.evictions = 0

    def key(self, query: str, scope: str = "") -> str:
        q = normalize(query) if self.normalize_key else query
        return f"{scope}::{q}"

    def get(self, query: str, scope: str = ""):
        k = self.key(query, scope)
        if k in self.store:
            if self.policy == "lru":
                self.store.move_to_end(k)
            self.hits += 1
            return self.store[k]
        self.misses += 1
        return None

    def put(self, query: str, value, scope: str = "") -> None:
        k = self.key(query, scope)
        self.store[k] = value
        if self.policy == "lru":
            self.store.move_to_end(k)
        if len(self.store) > self.capacity:
            self.store.popitem(last=False)
            self.evictions += 1

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0


class SemanticCache:
    """Cache sémantique : renvoie la réponse d'une requête passée si la similarité cosinus
    de leurs embeddings dépasse ``threshold`` (recherche exhaustive, suffisante ici)."""

    def __init__(self, embed_fn: Callable, threshold: float = 0.85):
        self.embed_fn, self.threshold = embed_fn, threshold
        self.queries: list[str] = []
        self.values: list = []
        self.E = None
        self.hits = self.misses = 0

    def lookup(self, query: str):
        """Renvoie (valeur, similarité, requête en cache) ou (None, meilleure similarité, None)."""
        if self.E is None:
            self.misses += 1
            return None, 0.0, None
        e = np.asarray(self.embed_fn([query]), dtype=np.float32)[0]
        sims = self.E @ e
        j = int(np.argmax(sims))
        if sims[j] >= self.threshold:
            self.hits += 1
            return self.values[j], float(sims[j]), self.queries[j]
        self.misses += 1
        return None, float(sims[j]), None

    def put(self, query: str, value) -> None:
        e = np.asarray(self.embed_fn([query]), dtype=np.float32)
        self.E = e if self.E is None else np.vstack([self.E, e])
        self.queries.append(query)
        self.values.append(value)


#: requêtes groupées par « réponse attendue » : même groupe = même réponse.
#: Certains groupes sont des pièges : très proches lexicalement, mais réponse différente.
CACHE_QUERIES: list[tuple[str, str]] = [
    ("Quel est le délai de livraison en France ?", "livr_fr"),
    ("Combien de temps pour être livré en France ?", "livr_fr"),
    ("En combien de jours arrive un colis en France métropolitaine ?", "livr_fr"),
    ("Quel est le délai de livraison en Belgique ?", "livr_be"),
    ("Combien de temps pour être livré en Belgique ?", "livr_be"),
    ("Comment retourner un produit ?", "retour"),
    ("Quelle est la procédure de retour d'un article ?", "retour"),
    ("Je veux renvoyer un objet, comment faire ?", "retour"),
    ("Combien de temps ai-je pour retourner un article ?", "retour_delai"),
    ("Quel est le délai de rétractation ?", "retour_delai"),
    ("Quels moyens de paiement acceptez-vous ?", "paiement"),
    ("Puis-je payer par PayPal ou carte ?", "paiement"),
    ("Comment puis-je régler ma commande ?", "paiement"),
    ("Puis-je payer en trois fois ?", "paiement_3x"),
    ("Proposez-vous le paiement en plusieurs fois sans frais ?", "paiement_3x"),
    ("Quelle est la durée de garantie d'un téléviseur ?", "garantie_tv"),
    ("Combien de temps une télé est-elle garantie ?", "garantie_tv"),
    ("Quelle est la durée de garantie d'un smartphone ?", "garantie_tel"),
    ("Combien de temps un téléphone est-il garanti ?", "garantie_tel"),
    ("Comment réinitialiser mon mot de passe ?", "mdp"),
    ("J'ai oublié mon mot de passe, que faire ?", "mdp"),
    ("Comment changer mon mot de passe ?", "mdp"),
    ("Où se trouve votre magasin de Lyon ?", "mag_lyon"),
    ("Quelle est l'adresse de la boutique à Lyon ?", "mag_lyon"),
    ("Où se trouve votre magasin de Lille ?", "mag_lille"),
    ("Quels sont les horaires d'ouverture du magasin ?", "horaires"),
    ("À quelle heure ouvre la boutique ?", "horaires"),
    ("La livraison est-elle gratuite ?", "livr_gratuite"),
    ("Y a-t-il des frais de port ?", "livr_gratuite"),
    ("À partir de quel montant la livraison est-elle offerte ?", "livr_gratuite"),
    ("Livrez-vous le dimanche ?", "livr_dimanche"),
    ("Puis-je être livré un dimanche ?", "livr_dimanche"),
    ("Comment contacter le service client ?", "contact"),
    ("Quel est le numéro du service client ?", "contact"),
    ("Proposez-vous une extension de garantie ?", "ext_garantie"),
    ("Puis-je acheter une garantie supplémentaire ?", "ext_garantie"),
]


def pairwise_cache_stats(E: np.ndarray, groups, thresholds) -> dict[str, np.ndarray]:
    """Rejoue un flux de requêtes dans un cache sémantique pour chaque seuil.

    Les requêtes arrivent dans l'ordre ; la requête i interroge le cache formé des
    requêtes 0..i-1. Succès si la meilleure similarité >= seuil ; le succès est une
    erreur si la requête trouvée n'a pas la même réponse attendue (groupe différent).
    Renvoie hit_rate, err_rate (erreurs / requêtes), precision (succès corrects / succès),
    et le taux maximal atteignable ``oracle`` (requêtes dont le groupe a déjà été vu).
    """
    E = np.asarray(E)
    groups = np.asarray(groups)
    S = E @ E.T
    n = len(E)
    best_sim = np.full(n, -1.0)
    best_ok = np.zeros(n, dtype=bool)
    for i in range(1, n):
        j = int(np.argmax(S[i, :i]))
        best_sim[i], best_ok[i] = S[i, j], groups[j] == groups[i]
    oracle = np.mean([groups[i] in set(groups[:i]) for i in range(n)])
    hit, err, prec = [], [], []
    for t in thresholds:
        h = best_sim >= t
        hit.append(h.mean())
        err.append((h & ~best_ok).mean())
        prec.append((h & best_ok).sum() / h.sum() if h.sum() else 1.0)
    return {"thresholds": np.asarray(thresholds), "hit_rate": np.array(hit), "err_rate": np.array(err),
            "precision": np.array(prec), "oracle": oracle}


#: indices de requêtes personnelles ou dépendantes du temps : à ne pas mettre en cache
NON_CACHEABLE = [r"\b(ma|mon|mes)\b", r"\bje\b.*\b(commande|compte|colis)\b", r"aujourd hui",
                 r"maintenant", r"\bmeteo\b", r"en ce moment", r"\bdemain\b"]


def is_cacheable(query: str) -> bool:
    """Heuristique simple : ne pas mettre en cache les requêtes personnelles ou datées."""
    t = normalize(query)
    return not any(re.search(p, t) for p in NON_CACHEABLE)


def cache_economics(hit_rate: float, err_rate: float, cost_call: float, lat_call: float,
                    lat_lookup: float, cost_lookup: float = 0.0, cost_error: float = 0.0) -> dict:
    """Coût et latence moyens par requête avec un cache.

    coût = c_lookup + (1 - h) c_appel + e c_erreur ; latence = l_lookup + (1 - h) l_appel.
    """
    cost = cost_lookup + (1 - hit_rate) * cost_call + err_rate * cost_error
    lat = lat_lookup + (1 - hit_rate) * lat_call
    return {"cout": cost, "latence": lat, "gain_cout": 1 - cost / cost_call, "gain_latence": 1 - lat / lat_call}


# =====================================================================================
# 7. Mini-application qui grandit avec l'architecture
# =====================================================================================

#: base documentaire de la boutique fictive (contexte pour le RAG)
KB: list[str] = [
    "Livraison en France métropolitaine : 2 à 3 jours ouvrés. Gratuite dès 50 euros d'achat.",
    "Livraison en Belgique : 4 à 5 jours ouvrés, frais de 7,90 euros.",
    "Retours : vous disposez de 30 jours après réception pour retourner un article non utilisé, "
    "via le formulaire en ligne. Le remboursement intervient sous 10 jours.",
    "Garantie : 2 ans sur tous les produits, 3 ans sur les téléviseurs. Extension possible à 5 ans.",
    "Paiement : carte bancaire, PayPal, et paiement en 3 fois sans frais dès 150 euros.",
    "Mot de passe oublié : cliquez sur « Mot de passe oublié » sur la page de connexion, "
    "un lien valable 1 heure est envoyé par email.",
    "Casque Bluetooth qui ne s'appaire pas : éteignez-le, maintenez le bouton 10 secondes pour le "
    "réinitialiser, puis supprimez l'appareil de la liste Bluetooth du téléphone avant de réessayer.",
    "Batterie qui se décharge vite : désactivez la luminosité automatique maximale, mettez à jour le "
    "système et vérifiez les applications gourmandes dans les réglages.",
    "Service client : du lundi au samedi de 9 h à 19 h, par chat ou au 01 23 45 67 89.",
    "Magasin de Lyon : 12 rue de la République, ouvert du mardi au samedi de 10 h à 19 h.",
    "Annulation : une commande peut être annulée tant qu'elle n'est pas expédiée, depuis l'espace client.",
]

STOCK_REPLIES = {
    "hors_perimetre": "Je suis l'assistant d'ÉlectroNova : je ne peux répondre qu'aux questions sur nos "
                      "produits, commandes et services. Comment puis-je vous aider ?",
    "ambigu": "Je ne suis pas sûr de bien comprendre : pouvez-vous préciser votre demande ?",
    "facturation": "Votre demande concerne la facturation : je vous transfère à un conseiller humain.",
    "mot_de_passe": "Pour réinitialiser votre mot de passe, cliquez sur « Mot de passe oublié » sur la "
                    "page de connexion : un lien valable 1 heure vous sera envoyé par email.",
    "injection": "Désolé, je ne peux pas traiter cette demande.",
}

SYSTEM_PROMPT = ("Tu es l'assistant du service client d'ÉlectroNova, une boutique d'électronique. "
                 "Réponds en français, en deux phrases maximum, uniquement à partir du contexte fourni. "
                 "Si le contexte ne contient pas la réponse, dis que tu ne sais pas.")


@dataclass
class Reply:
    text: str
    trace: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        return sum(s.dur_ms for s in self.trace)

    @property
    def cost(self) -> float:
        return sum(s.attrs.get("cout_$", 0.0) for s in self.trace)


@dataclass
class Span:
    """Un segment de trace : un composant, sa durée (simulée) et ses attributs."""

    name: str
    start_ms: float
    dur_ms: float
    attrs: dict = field(default_factory=dict)


#: latences simulées des composants (ms) : ordres de grandeur réalistes
LATENCY = {"guard_in": 5, "embed": 15, "retrieval": 25, "router": 15, "cache_exact": 1,
           "cache_sem": 20, "guard_out": 5, "action": 150}


class MiniApp:
    """Assistant de service client construit étape par étape (étapes 0 à 6 du chapitre).

    Paramètres
    ----------
    llm_fn : callable(messages) -> str, le modèle (par ex. ``llm.chat`` sur Qwen2.5-0.5B).
    step : niveau d'architecture (0 = requête -> modèle ; 1 = + contexte ; 2 = + garde-fous ;
        3 = + routeur et passerelle ; 4 = + caches ; 5 = + boucle agentique ; 6 = + écritures).
    embed_fn : embeddings (retrieval, routeur, cache sémantique).
    """

    def __init__(self, llm_fn: Callable, step: int = 0, embed_fn: Callable = hash_embed,
                 kb: list[str] | None = None, router: IntentRouter | None = None,
                 gateway: ModelGateway | None = None, sem_threshold: float = 0.9, k: int = 2,
                 loop_sim: float = 0.5):
        self.llm_fn, self.step, self.embed_fn = llm_fn, step, embed_fn
        self.loop_sim = loop_sim
        self.kb = kb or KB
        self.k = k
        self.kb_E = np.asarray(embed_fn(self.kb), dtype=np.float32) if step >= 1 else None
        self.router = router
        if step >= 3 and router is None:
            self.router = IntentRouter(embed_fn).fit(INTENTS_TRAIN)
        self.gateway = gateway
        if step >= 3 and gateway is None:
            self.gateway = ModelGateway([Provider("local", llm_fn)])
        self.exact = ExactCache() if step >= 4 else None
        self.semantic = SemanticCache(embed_fn, sem_threshold) if step >= 4 else None
        self.out_guard = OutputGuard(block_refusal=False)
        self.actions_log: list[dict] = []
        self.orders = {"1042": "en préparation", "1043": "expédiée"}

    # -- briques ----------------------------------------------------------------------
    def retrieve(self, query: str, k: int | None = None) -> list[tuple[float, str]]:
        e = np.asarray(self.embed_fn([query]), dtype=np.float32)[0]
        sims = self.kb_E @ e
        idx = np.argsort(-sims)[: (k or self.k)]
        return [(float(sims[i]), self.kb[i]) for i in idx]

    def uncovered(self, query: str, docs: list[str]) -> list[str]:
        """Parties de la question mal couvertes par le contexte (étape 5).

        La question est découpée en propositions (virgules, deux-points, « et ») ; une
        proposition d'au moins 3 mots est « non couverte » si sa similarité maximale avec
        les documents du contexte est inférieure à ``loop_sim``.
        """
        parts = [p.strip() for p in re.split(r"[,:;?]| et | puis ", query) if len(p.split()) >= 3]
        if len(parts) < 2 or not docs:
            return []
        P = np.asarray(self.embed_fn(parts), dtype=np.float32)
        D = np.asarray(self.embed_fn(docs), dtype=np.float32)
        best = (P @ D.T).max(1)
        return [p for p, b in zip(parts, best) if b < self.loop_sim]

    def _messages(self, query: str, docs: list[str] | None) -> list[dict]:
        if docs is None:
            return [{"role": "user", "content": query}]
        ctx = "\n".join(f"- {d}" for d in docs)
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Contexte :\n{ctx}\n\nQuestion : {query}"}]

    def _generate(self, messages, trace, t, now):
        if self.gateway is not None:
            res = self.gateway.call(messages, now=now)
            trace.append(Span("passerelle + modèle", t, res.latency_ms,
                              {"fournisseur": res.provider, "statut": res.status, "cout_$": res.cost}))
            return res.text or "Service momentanément indisponible, réessayez plus tard.", t + res.latency_ms
        text = self.llm_fn(messages)
        prompt = " ".join(m["content"] for m in messages)
        dur = 300 + 20 * n_tokens(text)
        trace.append(Span("modèle", t, dur, {"tokens_in": n_tokens(prompt), "tokens_out": n_tokens(text)}))
        return text, t + dur

    # -- point d'entrée -----------------------------------------------------------------
    def ask(self, query: str, user: str = "anonyme", confirm: bool = False, now: float = 0.0) -> Reply:
        trace: list[Span] = []
        t = 0.0
        meta: dict = {"etape": self.step}
        reverse: dict = {}
        q = query

        def span(name, dur, **attrs):
            nonlocal t
            trace.append(Span(name, t, dur, attrs))
            t += dur

        # étape 2 : garde-fous d'entrée
        if self.step >= 2:
            inj = detect_injection(q)
            q, reverse = mask_pii(q)
            span("garde-fous entrée", LATENCY["guard_in"], pii=len(reverse), injection=bool(inj))
            meta["pii_masquees"] = list(reverse)
            if inj:
                meta["bloque"] = "injection"
                return Reply(STOCK_REPLIES["injection"], trace, meta)

        # étape 4 : caches (avant tout calcul coûteux)
        cacheable = self.step >= 4 and is_cacheable(q)
        if cacheable:
            hit = self.exact.get(q)
            span("cache exact", LATENCY["cache_exact"], succes=hit is not None)
            if hit is not None:
                meta["cache"] = "exact"
                return Reply(unmask(hit, reverse), trace, meta)
            val, sim, src = self.semantic.lookup(q)
            span("cache sémantique", LATENCY["cache_sem"], succes=val is not None, similarite=round(sim, 3))
            if val is not None:
                meta["cache"] = f"sémantique (sim={sim:.2f} avec « {src} »)"
                return Reply(unmask(val, reverse), trace, meta)

        # étape 3 : routeur
        if self.step >= 3:
            intent = self.router.predict([q])[0]
            span("routeur", LATENCY["router"], intention=intent)
            meta["intention"] = intent
            if intent in ("hors_perimetre", "ambigu", "facturation", "mot_de_passe"):
                return Reply(STOCK_REPLIES[intent], trace, meta)
            # étape 6 : actions d'écriture
            if self.step >= 6 and intent == "commande":
                m = re.search(r"annul\w*.*?(\d{4})", normalize(q))
                if m:
                    return self._write_action(m.group(1), user, confirm, trace, t, meta)

        # étape 1 : contexte
        docs = None
        if self.step >= 1:
            hits = self.retrieve(q)
            docs = [d for _, d in hits]
            span("recherche contexte", LATENCY["embed"] + LATENCY["retrieval"], top_sim=round(hits[0][0], 3))
            meta["contexte"] = docs

        text, t = self._generate(self._messages(q, docs), trace, t, now)

        # étape 5 : boucle agentique. Le système vérifie sa propre sortie : si le modèle dit
        # ne pas savoir, ou si une partie de la question n'est pas couverte par le contexte,
        # il relance une recherche ciblée sur ces parties puis régénère.
        if self.step >= 5:
            missing = self.uncovered(q, docs)
            unsure = bool(re.search(r"ne sais pas|pas d information|ne dispose pas", normalize(text)))
            if missing or unsure:
                extra = []
                for part in (missing or [q]):
                    extra += [d for _, d in self.retrieve(part, k=1 if missing else self.k + 2)]
                docs = list(dict.fromkeys(docs + extra))
                span("boucle : recherche ciblée", LATENCY["embed"] + LATENCY["retrieval"],
                     parties=len(missing), k=len(docs))
                text, t = self._generate(self._messages(q, docs), trace, t, now)
                meta["boucle"] = missing or ["le modèle ne savait pas"]
                meta["contexte"] = docs

        # étape 2 : garde-fous de sortie (une relance si échec)
        if self.step >= 2:
            fails = self.out_guard.check(text)
            span("garde-fous sortie", LATENCY["guard_out"], echecs=fails)
            if fails:
                text, t = self._generate(self._messages(q, docs), trace, t, now)
                if self.out_guard.check(text):
                    text = "Je préfère vous mettre en relation avec un conseiller."
                meta["relance"] = fails
            text = unmask(text, reverse)

        if cacheable:
            masked_text, _ = mask_pii(text)
            self.exact.put(q, masked_text)
            self.semantic.put(q, masked_text)
        return Reply(text, trace, meta)

    def _write_action(self, order_id, user, confirm, trace, t, meta) -> Reply:
        meta["action"] = f"annuler_commande({order_id})"
        if order_id not in self.orders:
            return Reply(f"Je ne trouve pas la commande {order_id}.", trace, meta)
        if self.orders[order_id] == "expédiée":
            return Reply(f"La commande {order_id} est déjà expédiée : annulation impossible, "
                         "mais vous pourrez la retourner sous 30 jours.", trace, meta)
        if not confirm:
            meta["en_attente_confirmation"] = True
            return Reply(f"Je peux annuler la commande {order_id}. Confirmez-vous ? (action irréversible)",
                         trace, meta)
        self.orders[order_id] = "annulée"
        self.actions_log.append({"utilisateur": user, "action": "annuler_commande", "commande": order_id})
        trace.append(Span("action d'écriture", t, LATENCY["action"], {"commande": order_id}))
        return Reply(f"C'est fait : la commande {order_id} est annulée.", trace, meta)


def trace_table(reply: Reply):
    """Tableau pandas d'une trace (un segment par ligne)."""
    import pandas as pd

    return pd.DataFrame([{"composant": s.name, "début_ms": round(s.start_ms, 1), "durée_ms": round(s.dur_ms, 1),
                          "attributs": ", ".join(f"{k}={v}" for k, v in s.attrs.items())} for s in reply.trace])


def plot_trace(trace: list[Span], ax=None, title: str = "Trace d'une requête"):
    """Diagramme de Gantt d'une trace (comme les vues de LangSmith ou Langfuse)."""
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(8, 0.45 * len(trace) + 1.2))[1]
    palette = {"garde": ROSE, "cache": AMBER, "routeur": VIOLET, "recherche": SKY, "boucle": AMBER,
               "modèle": TEAL, "passerelle": TEAL, "action": GREEN}
    for i, s in enumerate(trace):
        c = next((v for k, v in palette.items() if k in s.name), SLATE)
        ax.barh(i, s.dur_ms, left=s.start_ms, color=c, height=0.55)
        ax.text(s.start_ms + s.dur_ms + 5, i, f"{s.dur_ms:.0f} ms", va="center", fontsize=8.5)
    ax.set_yticks(range(len(trace)), [s.name for s in trace])
    ax.invert_yaxis()
    ax.set_xlabel("temps depuis l'arrivée de la requête (ms)")
    ax.set_title(title)
    end = max(s.start_ms + s.dur_ms for s in trace)
    ax.set_xlim(0, end * 1.18)
    return ax


# =====================================================================================
# 8. Monitoring : simulation de production, dérive, MTTD / MTTR
# =====================================================================================


def simulate_production(n_hours: int = 24 * 14, seed: int = 0, model_update_at: int | None = 24 * 9,
                        prompt_change_at: int | None = None, user_drift: float = 0.0,
                        outage: tuple[int, int] | None = None):
    """Simule les métriques horaires d'une application en production.

    - trafic avec cycle jour/nuit ; latence (p50, p95) ; tokens de sortie moyens ;
    - taux de refus, de déclenchement des garde-fous, de succès du cache, coût horaire ;
    - ``model_update_at`` : mise à jour silencieuse du modèle chez le fournisseur
      (réponses plus courtes, plus de refus) ;
    - ``prompt_change_at`` : modification du prompt système (réponses plus longues) ;
    - ``user_drift`` : baisse progressive de la longueur des requêtes (les utilisateurs apprennent) ;
    - ``outage`` : (début, fin) d'une panne partielle du fournisseur (latence et erreurs).
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    h = np.arange(n_hours)
    daily = 0.55 + 0.45 * np.sin(2 * np.pi * ((h % 24) - 8) / 24).clip(-0.6, 1)
    req = rng.poisson(400 * daily)
    tok_in = 120 * (1 - user_drift * h / n_hours) + rng.normal(0, 4, n_hours)
    tok_out = 180 + rng.normal(0, 8, n_hours)
    refusal = 0.03 + rng.normal(0, 0.006, n_hours)
    err = 0.005 + np.abs(rng.normal(0, 0.002, n_hours))
    if model_update_at is not None:
        tok_out[model_update_at:] -= 25
        refusal[model_update_at:] += 0.02
    if prompt_change_at is not None:
        tok_out[prompt_change_at:] += 60
    p50 = 350 + 2.2 * tok_out + 0.25 * req + rng.normal(0, 15, n_hours)
    p95 = p50 * (1.9 + rng.normal(0, 0.05, n_hours))
    if outage is not None:
        a, b = outage
        p95[a:b] *= 3.0
        p50[a:b] *= 1.6
        err[a:b] += 0.12
    cache = 0.28 + 0.03 * np.sin(2 * np.pi * h / 24) + rng.normal(0, 0.015, n_hours)
    guard = 0.02 + np.abs(rng.normal(0, 0.005, n_hours))
    cost = req * (1 - cache) * (tok_in * 0.15 + tok_out * 0.6) / 1e6 * 1000  # $ (échelle ×1000 req)
    return pd.DataFrame({"heure": h, "requetes": req, "tokens_entree": tok_in, "tokens_sortie": tok_out,
                         "latence_p50_ms": p50, "latence_p95_ms": p95, "taux_refus": refusal.clip(0, 1),
                         "taux_erreur": err.clip(0, 1), "taux_cache": cache.clip(0, 1),
                         "taux_garde_fous": guard, "cout_$": cost})


def cusum(x, target: float | None = None, sigma: float | None = None, k: float = 0.5, h: float = 5.0,
          n_ref: int = 48, direction: str = "both") -> tuple[np.ndarray, np.ndarray, list[int]]:
    """CUSUM (somme cumulée) de Page pour détecter un changement de moyenne.

    Les valeurs sont centrées réduites sur une période de référence (``n_ref`` premiers points)
    ; ``k`` est la marge (en écarts-types) et ``h`` le seuil d'alarme.
    S+_t = max(0, S+_{t-1} + z_t - k), S-_t = max(0, S-_{t-1} - z_t - k) ; alarme si S > h,
    puis remise à zéro. Renvoie (S+, S-, indices d'alarme).
    """
    x = np.asarray(x, dtype=float)
    mu = x[:n_ref].mean() if target is None else target
    sd = x[:n_ref].std(ddof=1) if sigma is None else sigma
    z = (x - mu) / sd
    sp, sm = np.zeros(len(x)), np.zeros(len(x))
    alarms = []
    a = b = 0.0
    for t, zt in enumerate(z):
        a = max(0.0, a + zt - k) if direction in ("both", "up") else 0.0
        b = max(0.0, b - zt - k) if direction in ("both", "down") else 0.0
        sp[t], sm[t] = a, b
        if a > h or b > h:
            alarms.append(t)
            a = b = 0.0
    return sp, sm, alarms


def first_alarm_after(alarms, change_point: int) -> int | None:
    """Premier indice d'alarme à partir du point de changement (None si aucun)."""
    after = [a for a in alarms if a >= change_point]
    return after[0] if after else None


def devops_metrics(incidents, n_deploys: int, failed_deploys: int) -> dict:
    """MTTD, MTTR et CFR (section *Monitoring and Observability*).

    ``incidents`` : liste de dicts avec les instants ``debut``, ``detection``, ``resolution``.
    MTTD = moyenne(detection - debut) ; MTTR = moyenne(resolution - detection) ;
    CFR = déploiements fautifs / déploiements.
    """
    d = np.array([i["detection"] - i["debut"] for i in incidents], dtype=float)
    r = np.array([i["resolution"] - i["detection"] for i in incidents], dtype=float)
    return {"MTTD": d.mean(), "MTTR": r.mean(), "CFR": failed_deploys / n_deploys,
            "indisponibilite": (d + r).sum()}


# =====================================================================================
# 9. Orchestration
# =====================================================================================


@dataclass
class Step:
    name: str
    fn: Callable
    inputs: tuple
    output: str
    latency_ms: float = 0.0


class Pipeline:
    """Orchestrateur minimal : définition de composants + chaînage en graphe orienté acyclique.

    Chaque étape lit des clés d'un état partagé (``inputs``) et écrit une clé (``output``).
    L'orchestrateur vérifie le câblage, exécute dans l'ordre topologique et calcule la
    latence séquentielle (somme) et parallèle (chemin critique).
    """

    def __init__(self, name: str = "pipeline"):
        self.name = name
        self.steps: dict[str, Step] = {}

    def add(self, name: str, fn: Callable, inputs=(), output: str | None = None, latency_ms: float = 0.0):
        self.steps[name] = Step(name, fn, tuple(inputs), output or name, latency_ms)
        return self

    def graph(self):
        import networkx as nx

        g = nx.DiGraph()
        producers = {s.output: s.name for s in self.steps.values()}
        for s in self.steps.values():
            g.add_node(s.name, latency=s.latency_ms)
            for k in s.inputs:
                if k in producers:
                    g.add_edge(producers[k], s.name)
        return g

    def validate(self, initial_keys=("query",)) -> list[str]:
        """Liste des problèmes de câblage (entrée jamais produite, cycle)."""
        import networkx as nx

        problems = []
        produced = set(initial_keys) | {s.output for s in self.steps.values()}
        for s in self.steps.values():
            for k in s.inputs:
                if k not in produced:
                    problems.append(f"{s.name} attend « {k} » que personne ne produit")
        if not nx.is_directed_acyclic_graph(self.graph()):
            problems.append("le graphe contient un cycle")
        return problems

    def run(self, **state) -> dict:
        import networkx as nx

        problems = self.validate(tuple(state))
        if problems:
            raise ValueError("; ".join(problems))
        for name in nx.topological_sort(self.graph()):
            s = self.steps[name]
            state[s.output] = s.fn(*[state[k] for k in s.inputs])
        return state

    def latency(self) -> dict:
        """Latence séquentielle (somme) et parallèle (plus long chemin pondéré du graphe)."""
        import networkx as nx

        g = self.graph()
        finish = {}
        for n in nx.topological_sort(g):
            finish[n] = g.nodes[n]["latency"] + max((finish[p] for p in g.predecessors(n)), default=0.0)
        end = max(finish, key=finish.get)
        path = [end]
        while list(g.predecessors(path[-1])):
            path.append(max(g.predecessors(path[-1]), key=finish.get))
        return {"sequentielle": sum(s.latency_ms for s in self.steps.values()),
                "parallele": finish[end], "chemin_critique": path[::-1], "fin": finish}


# =====================================================================================
# 10. Retours utilisateurs : signaux conversationnels
# =====================================================================================

CORRECTION_PATTERNS = [r"^non\b", r"^pas (ca|du tout)", r"je voulais dire", r"^en fait", r"j ai dit",
                       r"ce n est pas (ce que|ca)", r"tu n as pas compris", r"^je parlais de"]
CONFIRM_PATTERNS = [r"tu es sur", r"vous etes sur", r"verifie", r"tes sources", r"quelles sources",
                    r"c est vrai", r"vraiment \?"]
COMPLAINT_PATTERNS = [r"trop long", r"pas clair", r"ne repond pas", r"c est faux", r"inutile", r"hors sujet",
                      r"tu te repetes", r"pas assez (precis|detaille)", r"n importe quoi"]
STOP_PATTERNS = [r"^stop\b", r"^arrete", r"\blaisse tomber\b", r"\btant pis\b"]
POS_WORDS = ["merci", "parfait", "super", "genial", "top", "excellent", "nickel", "bravo", "impeccable",
             "exactement", "ca marche", "ca fonctionne", "resolu", "cool"]
NEG_WORDS = ["nul", "pff", "pfff", "ugh", "agacant", "enervant", "n importe quoi", "inutile", "decu", "decue",
             "horrible", "ras le bol", "marre", "toujours pas", "ca ne marche pas", "ca ne fonctionne pas", "grr"]


def sentiment_score(text: str) -> float:
    """Score de sentiment lexical dans [-1, 1] : (positifs - négatifs) / (positifs + négatifs)."""
    t = f" {normalize(text)} "
    pos = sum(t.count(f" {w} ") for w in POS_WORDS)
    neg = sum(t.count(f" {w} ") for w in NEG_WORDS) + text.count("!!") + (1 if "?!" in text else 0)
    return 0.0 if pos + neg == 0 else (pos - neg) / (pos + neg)


def _match_any(patterns, text):
    t = normalize(text)
    return any(re.search(p, t) for p in patterns)


def extract_signals(conversation: list[dict], sim_fn: Callable | None = None, rephrase_threshold: float = 0.6) -> dict:
    """Extrait les signaux implicites d'une conversation (section *Extracting Conversational Feedback*).

    ``conversation`` : liste de messages {"role", "content"} ; un message assistant peut porter
    ``"interrupted": True`` (génération arrêtée) et un message utilisateur ``"action"``
    ("regenerer", "copier", "partager"...).
    ``sim_fn(a, b)`` : similarité entre deux messages utilisateur (par défaut : cosinus de
    ``hash_embed``) ; au-delà de ``rephrase_threshold`` on compte une reformulation.
    """
    if sim_fn is None:
        def sim_fn(a, b):
            E = hash_embed([a, b])
            return float(E[0] @ E[1])
    users = [m for m in conversation if m["role"] == "user"]
    bots = [m for m in conversation if m["role"] == "assistant"]
    rephr = sum(sim_fn(a["content"], b["content"]) >= rephrase_threshold for a, b in zip(users[:-1], users[1:]))
    sents = [sentiment_score(m["content"]) for m in users]
    return {
        "tours": len(users),
        "reformulations": int(rephr),
        "corrections": sum(_match_any(CORRECTION_PATTERNS, m["content"]) for m in users),
        "demandes_confirmation": sum(_match_any(CONFIRM_PATTERNS, m["content"]) for m in users),
        "plaintes": sum(_match_any(COMPLAINT_PATTERNS, m["content"]) for m in users),
        "interruptions": sum(bool(m.get("interrupted")) for m in bots)
        + sum(_match_any(STOP_PATTERNS, m["content"]) for m in users),
        "regenerations": sum(m.get("action") == "regenerer" for m in users),
        "refus_modele": sum(is_refusal(m["content"]) for m in bots),
        "sentiment_moyen": float(np.mean(sents)) if sents else 0.0,
        "sentiment_final": sents[-1] if sents else 0.0,
        "diversite_bot": _distinct_ratio([m["content"] for m in bots]),
    }


def _distinct_ratio(texts: list[str]) -> float:
    """Part de bigrammes distincts dans les réponses (faible = le bot se répète)."""
    grams = []
    for t in texts:
        w = normalize(t).split()
        grams += list(zip(w[:-1], w[1:]))
    return len(set(grams)) / len(grams) if grams else 1.0


#: poids par défaut d'un score de satisfaction implicite (à calibrer sur des données annotées)
SIGNAL_WEIGHTS = {"reformulations": -1.0, "corrections": -1.0, "demandes_confirmation": -0.5, "plaintes": -1.5,
                  "interruptions": -1.0, "regenerations": -0.7, "refus_modele": -1.0, "sentiment_final": 1.5,
                  "sentiment_moyen": 0.5}


def implicit_score(signals: dict, weights: dict | None = None) -> float:
    """Combinaison linéaire des signaux : > 0 suggère une conversation réussie."""
    w = weights or SIGNAL_WEIGHTS
    return float(sum(w[k] * signals[k] for k in w))


def _c(*turns):
    out = []
    for i, t in enumerate(turns):
        role = "user" if i % 2 == 0 else "assistant"
        if isinstance(t, dict):
            out.append({"role": role, **t})
        else:
            out.append({"role": role, "content": t})
    return out


#: petites conversations annotées (satisfait = 1, insatisfait = 0)
CONVERSATIONS: list[tuple[list[dict], int]] = [
    (_c("Quel est le délai de livraison en France ?",
        "La livraison en France métropolitaine prend 2 à 3 jours ouvrés.",
        "Parfait, merci !"), 1),
    (_c("Mon casque Bluetooth ne s'appaire pas",
        "Réinitialisez-le en maintenant le bouton 10 secondes, puis supprimez-le de la liste Bluetooth.",
        "Super, ça marche, merci beaucoup"), 1),
    (_c("Je veux retourner un article",
        {"content": "Pour retourner un article, vous devez d'abord vérifier les conditions générales de vente, "
                    "qui précisent que", "interrupted": True},
        "Non, je veux juste savoir comment retourner un article",
        "Utilisez le formulaire en ligne, vous avez 30 jours.",
        "Ok"), 0),
    (_c("La batterie de mon téléphone se vide vite",
        "Je suis désolé, je ne peux pas vous aider sur ce sujet.",
        "Pff, la batterie de mon téléphone se décharge très vite, que faire ?",
        "Je suis désolé, je ne peux pas vous aider.",
        "Inutile. Laisse tomber"), 0),
    (_c("Puis-je payer en plusieurs fois ?",
        "Oui, le paiement en 3 fois sans frais est possible dès 150 euros.",
        "Tu es sûr ? Même pour une télé à 140 euros ?",
        "Non, le seuil est de 150 euros.",
        "D'accord, merci"), 1),
    (_c("Quelle est la garantie d'une télévision ?",
        "La garantie est de 2 ans sur tous les produits.",
        "En fait je parlais des téléviseurs, c'est pas 3 ans ?",
        "Vous avez raison : 3 ans pour les téléviseurs.",
        "Ok merci"), 1),
    (_c("Comment annuler ma commande 1043 ?",
        "Votre commande peut être annulée depuis votre espace client.",
        "Ça ne marche pas, le bouton est grisé !!",
        "Votre commande peut être annulée depuis votre espace client.",
        "Tu te répètes, c'est n'importe quoi"), 0),
    (_c("Où est le magasin de Lyon ?",
        "Au 12 rue de la République, ouvert du mardi au samedi.",
        "Et le lundi c'est fermé ?",
        "Oui, le magasin est fermé le lundi.",
        "Top, merci"), 1),
    (_c("J'ai oublié mon mot de passe",
        "Cliquez sur « Mot de passe oublié » : un lien vous sera envoyé.",
        {"content": "J'ai oublié mon mot de passe, je ne reçois pas le lien", "action": "regenerer"},
        "Vérifiez vos courriers indésirables, le lien est valable 1 heure.",
        "Toujours pas reçu, ras le bol"), 0),
    (_c("Livrez-vous en Belgique ?",
        "Oui, en 4 à 5 jours ouvrés, pour 7,90 euros.",
        "Génial"), 1),
    (_c("L'écran de mon PC clignote",
        "Mettez à jour le pilote graphique et vérifiez le câble.",
        "C'est pas clair, quel pilote ?",
        "Le pilote de la carte graphique, dans le gestionnaire de périphériques.",
        "Ok je vais essayer"), 1),
    (_c("Combien coûte la livraison ?",
        {"content": "La livraison dépend de nombreux facteurs, notamment la destination, le poids, le mode de "
                    "transport choisi, la période de l'année et", "interrupted": True},
        "Stop. Combien coûte la livraison en France ?",
        "Elle est gratuite dès 50 euros d'achat.",
        "Trop long pour une réponse aussi simple"), 0),
]


# =====================================================================================
# 11. Biais des retours et boucles de rétroaction dégénérées
# =====================================================================================


def simulate_position_bias(n: int, p_a_better: float, p_first: float, randomize: bool, seed: int = 0) -> dict:
    """Comparaisons A/B par des utilisateurs soumis au biais de position.

    Avec probabilité ``p_first`` l'utilisateur clique sur l'option affichée en premier
    (sans lire) ; sinon il choisit A avec probabilité ``p_a_better``.
    Sans randomisation, A est toujours affichée en premier.
    Renvoie le taux de victoire observé de A et l'estimation corrigée.
    """
    rng = np.random.default_rng(seed)
    a_first = rng.random(n) < 0.5 if randomize else np.ones(n, dtype=bool)
    lazy = rng.random(n) < p_first
    honest_a = rng.random(n) < p_a_better
    choose_a = np.where(lazy, a_first, honest_a)
    out = {"observe": choose_a.mean()}
    if randomize:
        # P(A | A en 1er) = p_f + (1-p_f) q ; P(A | A en 2e) = (1-p_f) q ; leur moyenne = p_f/2 + (1-p_f) q
        pa1, pa2 = choose_a[a_first].mean(), choose_a[~a_first].mean()
        pf_hat = pa1 - pa2
        out["p_first_estime"] = pf_hat
        out["corrige"] = pa2 / (1 - pf_hat) if pf_hat < 1 else np.nan
    return out


def simulate_selection_bias(true_sat: float, p_fb_happy: float, p_fb_unhappy: float, n: int = 10000,
                            seed: int = 0) -> dict:
    """Seuls certains utilisateurs donnent un avis : satisfaction observée vs réelle."""
    rng = np.random.default_rng(seed)
    happy = rng.random(n) < true_sat
    gives = rng.random(n) < np.where(happy, p_fb_happy, p_fb_unhappy)
    return {"reelle": happy.mean(), "observee": happy[gives].mean(), "taux_reponse": gives.mean()}


def simulate_feedback_loop(qualities, n_rounds: int = 60, users_per_round: int = 500, policy: str = "greedy",
                           eps: float = 0.0, position_ctr=None, prior: float = 1.0, seed: int = 0) -> dict:
    """Boucle de rétroaction dans un système de recommandation (section *Degenerate feedback loop*).

    À chaque tour, le système classe les éléments selon leur taux de clic estimé et les
    affiche dans cet ordre ; le clic dépend de la qualité réelle × un facteur de position
    (biais d'exposition). Les estimations sont mises à jour avec les clics observés.
    - ``greedy`` : on classe selon le taux de clic brut estimé (boucle dégénérée) ;
    - ``epsilon`` : avec probabilité ``eps`` l'ordre est tiré au hasard (exploration) ;
    - ``ipw`` : on corrige chaque clic par l'inverse du facteur de position (pondération
      par l'inverse de la propension) avant d'estimer la qualité (avec exploration ``eps``
      éventuelle, comme ``epsilon``).
    Renvoie l'historique des parts d'exposition en tête et des qualités estimées.
    """
    rng = np.random.default_rng(seed)
    q = np.asarray(qualities, dtype=float)
    K = len(q)
    pos = np.asarray(position_ctr if position_ctr is not None else 1.0 / (1 + np.arange(K)) ** 1.0)
    clicks = np.full(K, prior * q.mean())
    views = np.full(K, prior)
    top_share, est_hist, cum_clicks = [], [], np.zeros(K)
    for _ in range(n_rounds):
        est = clicks / views
        top_counts = np.zeros(K)
        for _u in range(users_per_round // 10):  # paquets de 10 utilisateurs par classement
            if policy in ("epsilon", "ipw") and eps > 0 and rng.random() < eps:
                order = rng.permutation(K)
            else:
                order = np.argsort(-(est + 1e-9 * rng.random(K)))
            top_counts[order[0]] += 10
            p_click = q[order] * pos
            c = rng.binomial(10, p_click)
            cum_clicks[order] += c
            if policy == "ipw":
                clicks[order] += c / pos
                views[order] += 10
            else:
                clicks[order] += c
                views[order] += 10
        top_share.append(top_counts / top_counts.sum())
        est_hist.append(clicks / views)
    return {"top_share": np.array(top_share), "estimates": np.array(est_hist), "clicks": cum_clicks}


def leniency_ratings(true_quality, n: int = 5000, leniency: float = 1.2, seed: int = 0) -> np.ndarray:
    """Notes de 1 à 5 étoiles avec biais de complaisance : la note latente est décalée vers le haut."""
    rng = np.random.default_rng(seed)
    latent = 1 + 4 * np.asarray(true_quality) + rng.normal(0, 0.8, n) + leniency
    return np.clip(np.round(latent), 1, 5).astype(int)


# =====================================================================================
# 12. Aide pour les LLM : classification de retours par un petit modèle
# =====================================================================================

FEEDBACK_LABELS = ["satisfait", "correction", "plainte", "neutre"]


def llm_feedback_label(user_message: str, previous_answer: str, chat_fn: Callable) -> str:
    """Demande à un LLM (via ``chat_fn(messages) -> str``) de classer un message de retour.

    La sortie est ramenée à l'une des étiquettes de ``FEEDBACK_LABELS`` (« neutre » par défaut).
    """
    prompt = ("Voici la réponse d'un assistant puis le message suivant de l'utilisateur.\n"
              f"Réponse de l'assistant : « {previous_answer} »\n"
              f"Message de l'utilisateur : « {user_message} »\n\n"
              "Classe le message de l'utilisateur dans UNE catégorie : satisfait, correction, plainte, neutre.\n"
              "- satisfait : remerciement, approbation\n- correction : l'utilisateur corrige ou reformule\n"
              "- plainte : mécontentement, frustration\n- neutre : autre\n"
              "Réponds par un seul mot.")
    out = normalize(chat_fn([{"role": "user", "content": prompt}]))
    for lab in FEEDBACK_LABELS:
        if lab in out:
            return lab
    return "neutre"
