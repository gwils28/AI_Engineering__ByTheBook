"""Chapitre 7 : finetuning.

Fonctions réutilisables des notebooks du chapitre 7 :

1. style et petits utilitaires graphiques ;
2. jeu de données « tickets clients » en français (tâche de format de sortie) ;
3. calculs de mémoire (inférence, entraînement, activations, gradient checkpointing) et
   mesure réelle de la mémoire GPU d'un pas d'entraînement ;
4. représentations numériques (bits d'un flottant, plages, arrondi simulé dans un format
   quelconque) ;
5. quantification écrite à la main (absmax, zéro-point, par groupe, NF4) et application
   « simulée » aux poids d'un vrai modèle ;
6. LoRA écrite à la main (couche `LoRALinear`, comptage de paramètres) ;
7. finetuning supervisé minimal (masque de perte, poids de la perte du prompt, boucle
   d'entraînement transparente, génération par lots) ;
8. fusion de modèles (moyenne, task arithmetic, SLERP, TIES, DARE) et petits réseaux de
   démonstration sur le jeu `digits` de scikit-learn.

Les fonctions qui demandent torch l'importent localement : le module s'importe sans GPU.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

GB = 1e9  # on compte en Go décimaux (10^9 octets), comme le livre


# =====================================================================================
# 1. Style
# =====================================================================================

def setup_style() -> None:
    """`set_style()` du socle + repli de police (flèches, symboles) vers DejaVu Sans."""
    import matplotlib as mpl

    from .plotting import set_style

    set_style()
    mpl.rcParams["font.family"] = ["Noto Sans", "DejaVu Sans"]
    quiet_hf()


def quiet_hf() -> None:
    """Coupe les barres de progression et les avertissements de Hugging Face (sorties de notebook propres)."""
    import os

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # modèles déjà en cache : pas d'appel réseau
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        from huggingface_hub.utils import disable_progress_bars
        from transformers.utils import logging as hf_logging

        import logging

        logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
        disable_progress_bars()
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:  # pragma: no cover
        pass


def log_vram(tag: str) -> None:
    """Journal discret du pic de VRAM (fichier désigné par la variable AIE_VRAM_LOG, sinon rien)."""
    import os

    path = os.environ.get("AIE_VRAM_LOG")
    if not path:
        return
    import torch

    if torch.cuda.is_available():
        with open(path, "a") as f:
            f.write(f"{tag}\t{torch.cuda.max_memory_reserved() / 1e9:.2f}\n")
        torch.cuda.reset_peak_memory_stats()


def title_sub(ax, title: str, sub: str | None = None, pad: float = 20) -> None:
    """Titre d'axe + sous-titre gris sans chevauchement."""
    from .plotting import GRAY

    ax.set_title(title, pad=pad if sub else 6)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9, color=GRAY, va="bottom", ha="left")


def plain_log(ax, which: str = "y") -> None:
    """Graduations lisibles sur un axe logarithmique, sans mathtext."""
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


def sci_log(ax, which: str = "y") -> None:
    """Graduations d'un axe logarithmique en texte simple (0,1 ; 1 ; 1e-5 ; 1e+38), sans mathtext."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    def fmt(v, _):
        if v <= 0:
            return ""
        e = int(np.round(np.log10(v)))
        if abs(np.log10(v) - e) > 1e-6:
            return ""
        if -2 <= e <= 4:
            return f"{v:g}".replace(".", ",")
        return f"1e{e}"

    for axis in ([ax.xaxis] if which == "x" else [ax.yaxis] if which == "y" else [ax.xaxis, ax.yaxis]):
        axis.set_major_locator(LogLocator(base=10, numticks=12))
        axis.set_major_formatter(FuncFormatter(fmt))
        axis.set_minor_formatter(NullFormatter())


def fmt_bytes(n: float) -> str:
    """Octets -> chaîne lisible (o, Ko, Mo, Go, To ; puissances de 10)."""
    for unit, f in [("To", 1e12), ("Go", 1e9), ("Mo", 1e6), ("Ko", 1e3)]:
        if abs(n) >= f:
            return f"{n / f:.2f} {unit}".replace(".", ",")
    return f"{n:.0f} o"


# =====================================================================================
# 2. Jeu de données : tickets clients (tâche de format)
# =====================================================================================

CATEGORIES = ["livraison", "facturation", "technique", "compte"]
URGENCES = ["haute", "moyenne", "basse"]

#: pour chaque catégorie : (formulation du problème, résumé court attendu)
PROBLEMES = {
    "livraison": [
        ("ma commande {num} n'est toujours pas arrivée", "colis non reçu"),
        ("le colis {num} est arrivé complètement abîmé", "colis endommagé"),
        ("le livreur a déposé mon colis à la mauvaise adresse", "erreur d'adresse de livraison"),
        ("le suivi de ma commande {num} ne bouge plus depuis cinq jours", "suivi bloqué"),
        ("il manque un article dans ma commande {num}", "article manquant"),
    ],
    "facturation": [
        ("j'ai été débité deux fois pour la commande {num}", "double prélèvement"),
        ("je ne trouve pas ma facture du mois de {mois}", "facture introuvable"),
        ("le montant facturé ne correspond pas au devis", "écart de montant"),
        ("mon remboursement pour la commande {num} n'est jamais arrivé", "remboursement non reçu"),
        ("je voudrais changer la carte bancaire de mon abonnement", "changement de moyen de paiement"),
    ],
    "technique": [
        ("l'application plante dès que j'ouvre le panier", "plantage de l'application"),
        ("le site affiche une erreur 500 au moment du paiement", "erreur au paiement"),
        ("je ne reçois plus aucune notification", "notifications absentes"),
        ("la page de connexion tourne en boucle", "connexion en boucle"),
        ("le bouton de téléchargement des factures ne fait rien", "bouton inactif"),
    ],
    "compte": [
        ("je n'arrive pas à réinitialiser mon mot de passe", "mot de passe bloqué"),
        ("je veux supprimer mon compte et toutes mes données", "suppression de compte"),
        ("mon adresse e-mail a changé, comment la mettre à jour ?", "mise à jour de l'e-mail"),
        ("quelqu'un s'est connecté à mon compte sans mon accord", "connexion suspecte"),
        ("je voudrais fusionner mes deux comptes", "fusion de comptes"),
    ],
}

INDICES_URGENCE = {
    "haute": ["C'est urgent !", "Je suis bloqué depuis ce matin, merci de faire vite.",
              "J'ai besoin d'une solution aujourd'hui.", "Situation critique, aidez-moi au plus vite."],
    "moyenne": ["Merci de regarder quand vous pourrez.", "J'aimerais une réponse cette semaine.",
                "Pouvez-vous vous en occuper dans les prochains jours ?", "Merci de me tenir au courant."],
    "basse": ["Rien de pressé.", "Simple question, aucune urgence.", "Quand vous aurez un moment, sans urgence.",
              "Ce n'est pas grave, c'est juste pour information."],
}
OUVERTURES = ["Bonjour,", "Bonsoir,", "Madame, Monsieur,", "Salut,", ""]
SIGNATURES = ["Cordialement, Julie", "Merci, Marc", "Bien à vous, Amina", "Léo", "Bonne journée, Sophie", ""]
MOIS = ["janvier", "mars", "mai", "juillet", "septembre", "novembre"]

#: phrases variées rédigées à la main (14 par catégorie) pour l'expérience de transfert
PHRASES_CLASSIF = {
    "livraison": [
        "Mon paquet n'est jamais arrivé.", "Le transporteur dit avoir livré mais je n'ai rien reçu.",
        "Quand vais-je recevoir mes achats ?", "La boîte était écrasée et le vase cassé.",
        "Le coursier est passé alors que j'étais absent, que faire ?", "Il y a du retard sur l'expédition ?",
        "Je n'ai reçu que deux chaussures sur trois paires.", "Le point relais a renvoyé mon colis.",
        "Pouvez-vous livrer chez mon voisin ?", "Le numéro de suivi n'existe pas.",
        "Mes courses ont été laissées sous la pluie devant la porte.", "Ça fait trois semaines que j'attends le carton.",
        "On m'a envoyé la mauvaise taille de pull.", "Est-il possible de changer le créneau de passage du livreur ?",
    ],
    "facturation": [
        "On m'a prélevé deux fois le même montant.", "Je veux une copie de mon reçu.",
        "Le prix payé est plus élevé que celui affiché.", "Quand serai-je remboursé ?",
        "Ma carte a été refusée alors qu'elle est valide.", "Pourquoi cette ligne de frais supplémentaire ?",
        "Je souhaite payer en trois fois.", "Le code promo n'a pas été déduit du total.",
        "J'ai besoin d'une facture au nom de mon entreprise.", "L'abonnement continue d'être débité après résiliation.",
        "La TVA est mal calculée sur mon relevé.", "Mon avoir n'apparaît pas sur le compte.",
        "Je conteste ce prélèvement inconnu.", "Puis-je régler par virement plutôt que par carte ?",
    ],
    "technique": [
        "L'appli se ferme toute seule sur mon téléphone.", "Le site est très lent ce soir.",
        "Impossible de valider le panier, la page reste blanche.", "Les images des produits ne s'affichent pas.",
        "J'ai un message d'erreur quand je clique sur commander.", "La recherche ne donne aucun résultat.",
        "La mise à jour a cassé le mode sombre.", "Le lecteur vidéo ne démarre pas.",
        "Le formulaire de contact ne s'envoie pas.", "Le chargement tourne indéfiniment sur tablette.",
        "Les filtres de prix ne fonctionnent plus.", "Mon navigateur affiche un avertissement de sécurité sur votre page.",
        "Le scanner de code-barres de l'application est bloqué.", "Le bouton ajouter au panier ne réagit pas.",
    ],
    "compte": [
        "J'ai oublié mon identifiant.", "Comment changer mon numéro de téléphone dans mon profil ?",
        "Je souhaite fermer définitivement mon espace client.", "Mon compte a été suspendu sans explication.",
        "Je ne reçois pas l'e-mail de confirmation d'inscription.", "Quelqu'un a modifié mon adresse de connexion.",
        "Puis-je avoir deux profils avec la même adresse ?", "Je veux activer la double authentification.",
        "Mon mot de passe n'est plus accepté.", "Comment télécharger toutes mes données personnelles ?",
        "Je voudrais changer le nom affiché sur mon profil.", "Mon compte affiche les commandes d'une autre personne.",
        "Le lien de vérification a expiré.", "Je n'arrive plus à me connecter avec mon compte Google.",
    ],
}

#: consigne minimale (sans description du format) utilisée pour le finetuning
INSTRUCTION = "Analyse ce message client."

#: consigne détaillée (prompt engineering) pour la comparaison avec le finetuning
INSTRUCTION_DETAILLEE = (
    "Analyse ce message client. Réponds sur une seule ligne, exactement au format :\n"
    "Catégorie : <livraison|facturation|technique|compte> | Urgence : <haute|moyenne|basse> | "
    "Résumé : <3 à 5 mots>\n"
    "Exemple : Catégorie : compte | Urgence : basse | Résumé : changement de pseudo"
)


@dataclass
class Ticket:
    message: str
    categorie: str
    urgence: str
    resume: str

    @property
    def cible(self) -> str:
        return format_ticket(self.categorie, self.urgence, self.resume)


def format_ticket(categorie: str, urgence: str, resume: str) -> str:
    """La sortie attendue, sur une ligne."""
    return f"Catégorie : {categorie} | Urgence : {urgence} | Résumé : {resume}"


_TICKET_RE = re.compile(
    r"^\s*Catégorie\s*:\s*(?P<c>[\w\-]+)\s*\|\s*Urgence\s*:\s*(?P<u>[\w\-]+)\s*\|\s*Résumé\s*:\s*(?P<r>.+?)\s*$")


def parse_ticket(text: str) -> dict | None:
    """Analyse une sortie de modèle ; None si le format n'est pas respecté à la lettre.

    Seule la première ligne compte, et les valeurs doivent appartenir aux listes autorisées.
    """
    line = text.strip().split("\n")[0] if text.strip() else ""
    m = _TICKET_RE.match(line)
    if not m:
        return None
    c, u = m["c"].lower(), m["u"].lower()
    if c not in CATEGORIES or u not in URGENCES:
        return None
    return {"categorie": c, "urgence": u, "resume": m["r"]}


def make_tickets(n: int, seed: int = 0, split: str = "train") -> list[Ticket]:
    """Génère `n` tickets. split="train" utilise les 4 premiers problèmes de chaque
    catégorie, split="test" le 5e (formulation jamais vue à l'entraînement)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        cat = CATEGORIES[i % len(CATEGORIES)]
        probs = PROBLEMES[cat][:4] if split == "train" else PROBLEMES[cat][4:]
        tpl, resume = probs[rng.integers(len(probs))]
        urg = URGENCES[rng.integers(3)]
        pb = tpl.format(num=f"{rng.integers(10000, 99999)}", mois=MOIS[rng.integers(len(MOIS))])
        pb = pb[0].upper() + pb[1:]
        if not pb.endswith(("?", "!", ".")):
            pb += "."
        parts = [OUVERTURES[rng.integers(len(OUVERTURES))], pb,
                 INDICES_URGENCE[urg][rng.integers(4)], SIGNATURES[rng.integers(len(SIGNATURES))]]
        msg = " ".join(p for p in parts if p)
        out.append(Ticket(msg, cat, urg, resume))
    return out


def score_tickets(outputs: list[str], tickets: list[Ticket]) -> dict:
    """Taux de format valide, exactitude de la catégorie et de l'urgence (sur tout le lot)."""
    parsed = [parse_ticket(o) for o in outputs]
    n = len(tickets)
    fmt = sum(p is not None for p in parsed) / n
    cat = sum(p is not None and p["categorie"] == t.categorie for p, t in zip(parsed, tickets)) / n
    urg = sum(p is not None and p["urgence"] == t.urgence for p, t in zip(parsed, tickets)) / n
    return {"format valide": fmt, "catégorie exacte": cat, "urgence exacte": urg}


# =====================================================================================
# 3. Quand finetuner : données du livre et petits modèles de décision
# =====================================================================================

#: Tableau 7-2 du livre (Ovadia et al., 2024), questions sur l'actualité
OVADIA = {
    "Mistral-7B": {"base": 0.481, "base + RAG": 0.875, "FT-reg": 0.504, "FT-par": 0.588,
                   "FT-reg + RAG": 0.810, "FT-par + RAG": 0.830},
    "Llama 2-7B": {"base": 0.353, "base + RAG": 0.585, "FT-reg": 0.219, "FT-par": 0.392,
                   "FT-reg + RAG": 0.326, "FT-par + RAG": 0.520},
    "Orca 2-7B": {"base": 0.456, "base + RAG": 0.876, "FT-reg": 0.511, "FT-par": 0.566,
                  "FT-reg + RAG": 0.820, "FT-par + RAG": 0.826},
}


def recommend_adaptation(prompting_tried: bool, failure: str, has_data: bool = True,
                         rag_tried: bool = False) -> str:
    """Petit arbre de décision inspiré de la section *Finetuning and RAG*.

    failure : "aucune", "information", "comportement" ou "les deux".
    """
    if not prompting_tried:
        return "prompt engineering systématique (avec exemples)"
    if failure == "aucune":
        return "rester sur le prompt"
    if failure in ("information", "les deux") and not rag_tried:
        return "RAG simple (BM25)"
    if failure == "information":
        return "RAG avancé (embeddings, hybride)"
    if not has_data:
        return "collecter des données (chapitre 8) avant de finetuner"
    if failure == "comportement":
        return "finetuning (PEFT / LoRA)"
    return "RAG + finetuning"


def fewshot_vs_finetune_cost(n_requests: float, shot_tokens: int, base_tokens: int, out_tokens: int,
                             price_in: float, price_out: float, ft_price_in: float | None = None,
                             ft_price_out: float | None = None, ft_fixed_cost: float = 0.0) -> dict:
    """Coût (en $) de n requêtes : prompt avec exemples contre modèle finetuné sans exemples.

    Prix par million de tokens. Le modèle finetuné peut avoir des prix différents et un
    coût fixe (entraînement, données).
    """
    ft_price_in = price_in if ft_price_in is None else ft_price_in
    ft_price_out = price_out if ft_price_out is None else ft_price_out
    few = n_requests * ((base_tokens + shot_tokens) * price_in + out_tokens * price_out) / 1e6
    ft = ft_fixed_cost + n_requests * (base_tokens * ft_price_in + out_tokens * ft_price_out) / 1e6
    return {"few-shot": few, "finetuné": ft}


# =====================================================================================
# 4. Mémoire
# =====================================================================================

@dataclass
class ModelSpec:
    """Dimensions d'un transformer décodeur (pour les calculs de mémoire)."""
    name: str
    n_params: float
    layers: int
    hidden: int
    heads: int
    vocab: int = 32000


MODEL_SPECS = {
    "GPT-2 (124M)": ModelSpec("GPT-2 (124M)", 124e6, 12, 768, 12, 50257),
    "Qwen2.5-0.5B": ModelSpec("Qwen2.5-0.5B", 494e6, 24, 896, 14, 151936),
    "Qwen2.5-1.5B": ModelSpec("Qwen2.5-1.5B", 1.54e9, 28, 1536, 12, 151936),
    "Llama 2-7B": ModelSpec("Llama 2-7B", 6.74e9, 32, 4096, 32, 32000),
    "Llama 2-13B": ModelSpec("Llama 2-13B", 13.0e9, 40, 5120, 40, 32000),
    "Llama 2-70B": ModelSpec("Llama 2-70B", 69.0e9, 80, 8192, 64, 32000),
    "GPT-3 (175B)": ModelSpec("GPT-3 (175B)", 175e9, 96, 12288, 96, 50257),
}

#: octets par valeur pour quelques formats
BYTES = {"fp64": 8, "fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}


def inference_memory(n_params: float, bytes_per_param: float = 2, overhead: float = 0.2) -> float:
    """Mémoire d'inférence approchée du livre : N x M x (1 + overhead), en octets."""
    return n_params * bytes_per_param * (1 + overhead)


def training_memory(n_params: float, n_trainable: float | None = None, recipe: str = "livre",
                    optimizer: str = "adam", activations: float = 0.0) -> dict:
    """Décomposition de la mémoire d'entraînement (octets) : poids, gradients, états, activations.

    recipe :
      - "livre" : tout en 16 bits (2 octets) ; gradients + états = (1 + k) x 2 octets / param
        entraînable, k = 0 (SGD), 1 (momentum) ou 2 (Adam) ;
      - "fp32" : tout en 32 bits ;
      - "mixte" : recette de précision mixte classique (poids et gradients 16 bits, copie
        maîtresse fp32 et états Adam fp32), soit 16 octets par paramètre entraînable avec Adam.
    """
    n_trainable = n_params if n_trainable is None else n_trainable
    k = {"sgd": 0, "momentum": 1, "adam": 2}[optimizer]
    if recipe == "livre":
        w, g, s, master = 2 * n_params, 2 * n_trainable, 2 * k * n_trainable, 0
    elif recipe == "fp32":
        w, g, s, master = 4 * n_params, 4 * n_trainable, 4 * k * n_trainable, 0
    elif recipe == "mixte":
        w, g, s, master = 2 * n_params, 2 * n_trainable, 4 * k * n_trainable, 4 * n_trainable
    else:
        raise ValueError(recipe)
    d = {"poids": w, "gradients": g, "états de l'optimiseur": s + master, "activations": activations}
    d["total"] = sum(d.values())
    return d


def activation_memory(spec: ModelSpec, seq: int, batch: int, bytes_per_value: int = 2,
                      checkpointing: str = "aucun") -> float:
    """Mémoire des activations stockées pour la rétropropagation (octets).

    Formule de Korthikanti et al. (2022) pour un bloc transformer en 16 bits, sans
    parallélisme : s.b.h.(34 + 5.a.s/h) octets par couche. On la redimensionne pour
    d'autres tailles de valeurs (bytes_per_value / 2).
    checkpointing="complet" : on ne garde que l'entrée de chaque bloc, 2.s.b.h octets
    (en 16 bits), et on recalcule le reste pendant la passe arrière.
    """
    s, b, h, a, L = seq, batch, spec.hidden, spec.heads, spec.layers
    if checkpointing == "complet":
        per_layer = 2 * s * b * h
    else:
        per_layer = s * b * h * (34 + 5 * a * s / h)
    return L * per_layer * bytes_per_value / 2


def lora_trainable_fraction(n_params: float, n_lora: float) -> float:
    return n_lora / n_params


def measure_train_step(model, batch: dict, optimizer_factory, n_steps: int = 2) -> dict:
    """Mesure la mémoire GPU (octets) au fil d'un pas d'entraînement.

    Renvoie un dict : poids (avant tout), après la passe avant (activations stockées),
    pic pendant la passe arrière, gradients (après backward), états de l'optimiseur
    (après le pas), pic global. Plusieurs pas sont faits pour que les états d'Adam existent.
    """
    import torch

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optimizer_factory(params)
    base = torch.cuda.memory_allocated()
    res = {}
    for step in range(n_steps):
        torch.cuda.reset_peak_memory_stats()
        out = model(**batch)
        torch.cuda.synchronize()
        after_fwd = torch.cuda.memory_allocated()
        loss = out.loss
        del out  # libère les logits avant la passe arrière (sinon comptés avec les gradients)
        loss.backward()
        torch.cuda.synchronize()
        after_bwd = torch.cuda.memory_allocated()
        peak_bwd = torch.cuda.max_memory_allocated()
        opt.step()
        torch.cuda.synchronize()
        after_step = torch.cuda.memory_allocated()
        opt.zero_grad(set_to_none=True)
        del loss
        torch.cuda.synchronize()
        if step == n_steps - 1:
            # au dernier pas, les états de l'optimiseur existent déjà (créés au 1er pas)
            states = torch.cuda.memory_allocated() - base
            res = {"poids": base, "activations": after_fwd - base - states,
                   "gradients": after_bwd - base - states, "états de l'optimiseur": states,
                   "pic": peak_bwd}
    del opt
    torch.cuda.empty_cache()
    return res


# =====================================================================================
# 5. Représentations numériques
# =====================================================================================

#: (bits d'exposant, bits de mantisse) ; 1 bit de signe en plus
FLOAT_FORMATS = {
    "FP32": (8, 23),
    "TF32": (8, 10),
    "FP16": (5, 10),
    "BF16": (8, 7),
    "FP8 E5M2": (5, 2),
    "FP8 E4M3": (4, 3),
}


@dataclass
class FloatInfo:
    name: str
    exp_bits: int
    man_bits: int
    max: float
    min_normal: float
    min_subnormal: float
    eps: float

    @property
    def bits(self) -> int:
        return 1 + self.exp_bits + self.man_bits

    @property
    def decimal_digits(self) -> float:
        """Chiffres décimaux significatifs : log10(2^(m+1))."""
        return (self.man_bits + 1) * math.log10(2)


def float_info(name: str) -> FloatInfo:
    """Caractéristiques d'un format IEEE-like (exposant biaisé, valeurs spéciales réservées).

    Pour FP8 E4M3, on utilise la convention OCP (pas d'infini, max = 448).
    """
    e, m = FLOAT_FORMATS[name]
    bias = 2 ** (e - 1) - 1
    if name == "FP8 E4M3":
        mx = 448.0
    else:
        mx = (2 - 2 ** -m) * 2.0 ** (2 ** e - 2 - bias)
    return FloatInfo(name, e, m, mx, 2.0 ** (1 - bias), 2.0 ** (1 - bias - m), 2.0 ** -m)


def round_to_float(x, name: str) -> np.ndarray:
    """Arrondi au plus proche (pair en cas d'égalité) dans le format `name`, simulé en float64.

    Gère les sous-normaux et le dépassement (-> inf, ou saturation pour FP8 E4M3).
    """
    x = np.asarray(x, dtype=np.float64)
    info = float_info(name)
    e, m = FLOAT_FORMATS[name]
    bias = 2 ** (e - 1) - 1
    out = np.zeros_like(x)
    nz = (x != 0) & np.isfinite(x)
    ax = np.abs(x[nz])
    ex = np.floor(np.log2(ax))
    ex = np.maximum(ex, 1 - bias)  # sous-normaux : pas fixe
    q = 2.0 ** (ex - m)
    r = np.round(ax / q) * q  # np.round : arrondi au pair
    if name == "FP8 E4M3":
        r = np.minimum(r, info.max)
    else:
        r = np.where(r > info.max, np.inf, r)
    out[nz] = np.sign(x[nz]) * r
    out[~np.isfinite(x)] = x[~np.isfinite(x)]
    return out


def float_bits(x: float, name: str = "FP32") -> tuple[str, str, str]:
    """(signe, exposant, mantisse) en chaînes de bits, pour FP32, FP16 ou BF16 (via torch)."""
    import torch

    dt = {"FP32": (torch.float32, torch.int32, 32), "FP16": (torch.float16, torch.int16, 16),
          "BF16": (torch.bfloat16, torch.int16, 16)}[name]
    v = torch.tensor([x], dtype=dt[0]).view(dt[1]).item()
    bits = format(v & ((1 << dt[2]) - 1), f"0{dt[2]}b")
    e, _ = FLOAT_FORMATS[name]
    return bits[0], bits[1:1 + e], bits[1 + e:]


def int_range(bits: int, signed: bool = True) -> tuple[int, int]:
    if signed:
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2 ** bits - 1


def relative_error(x, y) -> np.ndarray:
    x, y = np.asarray(x, float), np.asarray(y, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(y - x) / np.abs(x)


# =====================================================================================
# 6. Quantification
# =====================================================================================

def quantize_absmax(x, bits: int = 8):
    """Quantification symétrique (absmax) : q = round(x / s), s = max|x| / (2^(b-1) - 1).

    Renvoie (q entiers, échelle s). Le zéro est représenté exactement.
    """
    x = np.asarray(x, dtype=np.float64)
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(x).max() / qmax if np.abs(x).max() > 0 else 1.0
    q = np.clip(np.round(x / s), -qmax, qmax).astype(np.int32)
    return q, s


def dequantize_absmax(q, s) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) * s


def quantize_zeropoint(x, bits: int = 8):
    """Quantification asymétrique (zéro-point) sur [0, 2^b - 1].

    s = (max - min) / (2^b - 1), z = round(-min / s), q = clip(round(x / s) + z).
    Renvoie (q, s, z).
    """
    x = np.asarray(x, dtype=np.float64)
    qmax = 2 ** bits - 1
    lo, hi = min(x.min(), 0.0), max(x.max(), 0.0)
    s = (hi - lo) / qmax if hi > lo else 1.0
    z = int(np.round(-lo / s))
    q = np.clip(np.round(x / s) + z, 0, qmax).astype(np.int32)
    return q, s, z


def dequantize_zeropoint(q, s, z) -> np.ndarray:
    return (np.asarray(q, dtype=np.float64) - z) * s


#: les 16 niveaux NF4 de QLoRA (Dettmers et al., 2023), normalisés dans [-1, 1]
NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0])


def nf4_levels_from_normal(n_neg: int = 7, n_pos: int = 8, offset: float = 0.9677) -> np.ndarray:
    """Reconstruit (approximativement) les niveaux NF4 à partir des quantiles de la loi normale."""
    from scipy.stats import norm

    pos = norm.ppf(np.linspace(offset, 0.5, n_pos + 1)[:-1])
    neg = -norm.ppf(np.linspace(offset, 0.5, n_neg + 1)[:-1])
    v = np.sort(np.concatenate([neg, [0.0], pos]))
    return v / np.abs(v).max()


def quantize_groupwise(x, bits: int = 4, group_size: int = 64, method: str = "absmax") -> np.ndarray:
    """Quantifie puis déquantifie `x` (aplati) par groupes de `group_size` valeurs.

    method : "absmax", "zeropoint" ou "nf4" (bits ignoré pour nf4). Une échelle (et un
    zéro-point) par groupe. Renvoie la reconstruction, de même forme que x.
    """
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(-1)
    n = len(flat)
    pad = (-n) % group_size
    g = np.concatenate([flat, np.zeros(pad)]).reshape(-1, group_size)
    if method == "absmax":
        qmax = 2 ** (bits - 1) - 1
        s = np.abs(g).max(1, keepdims=True) / qmax
        s[s == 0] = 1
        rec = np.clip(np.round(g / s), -qmax, qmax) * s
    elif method == "zeropoint":
        qmax = 2 ** bits - 1
        lo, hi = np.minimum(g.min(1, keepdims=True), 0), np.maximum(g.max(1, keepdims=True), 0)
        s = (hi - lo) / qmax
        s[s == 0] = 1
        z = np.round(-lo / s)
        rec = (np.clip(np.round(g / s) + z, 0, qmax) - z) * s
    elif method == "nf4":
        s = np.abs(g).max(1, keepdims=True)
        s[s == 0] = 1
        idx = np.abs((g / s)[..., None] - NF4_LEVELS).argmin(-1)
        rec = NF4_LEVELS[idx] * s
    else:
        raise ValueError(method)
    return rec.reshape(-1)[:n].reshape(x.shape)


def effective_bits(bits: float, group_size: int | None, scale_bits: int = 16, zero_bits: int = 0) -> float:
    """Bits moyens par poids, en comptant l'échelle (et le zéro-point) stockés par groupe."""
    if not group_size:
        return float(bits)
    return bits + (scale_bits + zero_bits) / group_size


def quant_error(x, rec) -> dict:
    """Erreurs de reconstruction : RMSE relative et SQNR (dB)."""
    x, rec = np.asarray(x, float), np.asarray(rec, float)
    err = rec - x
    rmse = float(np.sqrt((err ** 2).mean()))
    sig = float(np.sqrt((x ** 2).mean()))
    return {"rmse": rmse, "rmse relative": rmse / sig, "sqnr (dB)": 20 * math.log10(sig / max(rmse, 1e-30))}


def fake_quantize_weight(w, bits: int = 8, group_size: int | None = None, method: str = "absmax"):
    """Quantification simulée d'une matrice de poids torch (out x in), par ligne ou par groupe.

    group_size=None : une échelle par ligne (« par canal ») ; group_size=-1 : une seule échelle
    pour tout le tenseur. Le calcul se fait en float32, le résultat garde le dtype d'origine.
    """
    import torch

    dt = w.dtype
    x = w.detach().float()
    out_f, in_f = x.shape
    if group_size == -1:
        g = x.reshape(1, -1)
    elif group_size is None:
        g = x
    else:
        assert in_f % group_size == 0
        g = x.reshape(-1, group_size)
    if method == "absmax":
        qmax = 2 ** (bits - 1) - 1
        s = g.abs().amax(1, keepdim=True).clamp(min=1e-12) / qmax
        rec = torch.clamp(torch.round(g / s), -qmax, qmax) * s
    elif method == "zeropoint":
        qmax = 2 ** bits - 1
        lo = g.amin(1, keepdim=True).clamp(max=0)
        hi = g.amax(1, keepdim=True).clamp(min=0)
        s = ((hi - lo) / qmax).clamp(min=1e-12)
        z = torch.round(-lo / s)
        rec = (torch.clamp(torch.round(g / s) + z, 0, qmax) - z) * s
    elif method == "nf4":
        lv = torch.tensor(NF4_LEVELS, dtype=torch.float32, device=x.device)
        s = g.abs().amax(1, keepdim=True).clamp(min=1e-12)
        idx = ((g / s).unsqueeze(-1) - lv).abs().argmin(-1)
        rec = lv[idx] * s
    else:
        raise ValueError(method)
    return rec.reshape(out_f, in_f).to(dt)


def linear_layers(model, skip=("lm_head",)):
    """Liste (nom, module) des couches nn.Linear d'un modèle (hors lm_head par défaut)."""
    import torch

    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and not any(s in n for s in skip)]


def quantize_model_(model, bits: int, group_size: int | None = 128, method: str = "absmax") -> dict:
    """Quantifie en place (simulation) les poids des couches linéaires. Renvoie une sauvegarde
    CPU des poids d'origine, à passer à `restore_weights_`."""
    import torch

    backup = {}
    with torch.no_grad():
        for n, m in linear_layers(model):
            backup[n] = m.weight.detach().cpu().clone()
            m.weight.copy_(fake_quantize_weight(m.weight, bits, group_size, method))
    return backup


def restore_weights_(model, backup: dict) -> None:
    import torch

    mods = dict(model.named_modules())
    with torch.no_grad():
        for n, w in backup.items():
            mods[n].weight.copy_(w.to(mods[n].weight.device))


#: petit corpus français (rédigé pour ce dépôt) pour mesurer la perplexité
TEXTES_EVAL = [
    "Le finetuning consiste à poursuivre l'entraînement d'un modèle déjà pré-entraîné sur des "
    "données propres à une tâche. On part de poids qui savent déjà beaucoup de choses sur la "
    "langue, et l'on ne fait que les ajuster pour obtenir le comportement souhaité.",
    "La ville s'éveillait doucement. Les boulangers sortaient les premières baguettes du four, "
    "tandis que les cyclistes traversaient le pont sous une lumière encore pâle. Au marché, "
    "les marchands installaient leurs étals de fruits et de légumes de saison.",
    "Pour réduire la mémoire nécessaire, on peut représenter chaque poids avec moins de bits. "
    "Passer de seize à huit bits divise par deux la taille du modèle, mais introduit une petite "
    "erreur d'arrondi sur chaque valeur.",
    "Le comité a voté le budget de l'année prochaine après une longue discussion. La majorité "
    "des membres a soutenu l'augmentation des crédits consacrés à la rénovation des écoles et "
    "à l'entretien des routes communales.",
    "Une bonne recette de soupe demande peu d'ingrédients : quelques poireaux, des pommes de "
    "terre, un oignon, de l'eau et une pincée de sel. On laisse cuire à feu doux une demi-heure "
    "avant de mixer le tout.",
]


def perplexity_texts(model, tok, texts=TEXTES_EVAL) -> float:
    """Perplexité moyenne (par token, pondérée) d'un modèle causal sur une liste de textes."""
    import torch

    nll, n = 0.0, 0
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt")["input_ids"].to(model.device)
            out = model(ids, labels=ids)
            k = ids.shape[1] - 1
            nll += out.loss.float().item() * k
            n += k
    return math.exp(nll / n)


# =====================================================================================
# 7. LoRA
# =====================================================================================

def lora_param_count(d_in: int, d_out: int, r: int) -> int:
    """Paramètres d'un adaptateur LoRA de rang r sur une matrice d_out x d_in."""
    return r * (d_in + d_out)


def low_rank_approx(W, r: int) -> np.ndarray:
    """Meilleure approximation de rang r (au sens de Frobenius) par SVD tronquée."""
    U, S, Vt = np.linalg.svd(np.asarray(W, float), full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vt[:r]


def module_shapes(cfg) -> dict:
    """Formes (d_out, d_in) des matrices d'un bloc décodeur type Llama/Qwen (GQA prise en compte),
    à partir d'une config Hugging Face ou d'un dict."""
    g = cfg if isinstance(cfg, dict) else cfg.to_dict()
    h = g["hidden_size"]
    nh = g["num_attention_heads"]
    nkv = g.get("num_key_value_heads", nh) or nh
    hd = g.get("head_dim") or h // nh
    ff = g["intermediate_size"]
    return {"q_proj": (nh * hd, h), "k_proj": (nkv * hd, h), "v_proj": (nkv * hd, h), "o_proj": (h, nh * hd),
            "gate_proj": (ff, h), "up_proj": (ff, h), "down_proj": (h, ff)}


def lora_params_for_model(cfg, targets, r: int) -> int:
    """Nombre total de paramètres LoRA pour un modèle, LoRA appliquée à `targets` dans chaque couche."""
    g = cfg if isinstance(cfg, dict) else cfg.to_dict()
    shapes = module_shapes(g)
    per_layer = sum(lora_param_count(shapes[t][1], shapes[t][0], r) for t in targets)
    return per_layer * g["num_hidden_layers"]


def make_lora_linear_class():
    """Construit la classe `LoRALinear` (torch importé paresseusement)."""
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        """y = W x + (alpha / r) B A x, avec W gelé, A (r x d_in) aléatoire, B (d_out x r) nul.

        B = 0 à l'initialisation : le modèle adapté commence exactement égal au modèle de base.
        """

        def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.r, self.alpha = r, alpha
            self.scaling = alpha / r
            d_out, d_in = base.weight.shape
            self.A = nn.Parameter(torch.empty(r, d_in, dtype=base.weight.dtype, device=base.weight.device))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.B = nn.Parameter(torch.zeros(d_out, r, dtype=base.weight.dtype, device=base.weight.device))

        def forward(self, x):
            return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling

        def delta_weight(self):
            return self.scaling * (self.B @ self.A)

        def merged(self) -> nn.Linear:
            """Couche fusionnée W' = W + (alpha/r) B A : plus aucun coût à l'inférence."""
            lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
            with torch.no_grad():
                lin.weight.copy_(self.base.weight + self.delta_weight())
                if self.base.bias is not None:
                    lin.bias.copy_(self.base.bias)
            return lin.to(self.base.weight.device)

    return LoRALinear


def train_lora_toy(W0, W_target, r: int, alpha: float | None = None, steps: int = 300, lr: float = 1e-2,
                   n: int = 512, seed: int = 0) -> list[float]:
    """Apprend W_target à partir de W0 avec une LoRA de rang r (régression y = W x).

    Renvoie la courbe de perte (MSE relative à la variance de la cible).
    """
    import torch

    torch.manual_seed(seed)
    LoRALinear = make_lora_linear_class()
    W0 = torch.as_tensor(np.asarray(W0), dtype=torch.float32)
    Wt = torch.as_tensor(np.asarray(W_target), dtype=torch.float32)
    base = torch.nn.Linear(W0.shape[1], W0.shape[0], bias=False)
    with torch.no_grad():
        base.weight.copy_(W0)
    lay = LoRALinear(base, r=r, alpha=alpha if alpha is not None else r)
    X = torch.randn(n, W0.shape[1])
    Y = X @ Wt.T
    var = Y.var().item()
    opt = torch.optim.Adam([lay.A, lay.B], lr=lr)
    hist = []
    for _ in range(steps):
        loss = ((lay(X) - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(loss.item() / var)
    return hist


# =====================================================================================
# 8. Finetuning supervisé minimal
# =====================================================================================

def chat_prompt(tok, instruction: str, message: str) -> str:
    """Prompt au format conversation du modèle, prêt pour la génération."""
    msgs = [{"role": "user", "content": f"{instruction}\n\nMessage : {message}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_sft_example(tok, prompt_text: str, response: str, prompt_loss_weight: float = 0.0) -> dict:
    """Tokenise (prompt, réponse) et renvoie input_ids et poids de perte par token.

    Les tokens de la réponse ont un poids 1, ceux du prompt `prompt_loss_weight`
    (0 : on n'apprend que la réponse ; 1 : prompt et réponse comptent autant).
    """
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    r_ids = tok(response + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = p_ids + r_ids
    w = [prompt_loss_weight] * len(p_ids) + [1.0] * len(r_ids)
    return {"input_ids": ids, "weights": w, "n_prompt": len(p_ids)}


def collate(examples: list[dict], pad_id: int, device: str = "cuda") -> dict:
    """Assemble des exemples de longueurs variables (remplissage à droite)."""
    import torch

    L = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), L), pad_id, dtype=torch.long)
    w = torch.zeros((len(examples), L), dtype=torch.float32)
    att = torch.zeros((len(examples), L), dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e["input_ids"])
        ids[i, :n] = torch.tensor(e["input_ids"])
        w[i, :n] = torch.tensor(e["weights"])
        att[i, :n] = 1
    return {"input_ids": ids.to(device), "attention_mask": att.to(device), "weights": w.to(device)}


def weighted_lm_loss(logits, input_ids, weights):
    """Perte de modélisation pondérée : sum_t w_t * NLL_t / sum_t w_t (décalage d'un token)."""
    import torch.nn.functional as F

    lg = logits[:, :-1].float()
    tgt = input_ids[:, 1:]
    w = weights[:, 1:]
    nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").view_as(tgt)
    return (nll * w).sum() / w.sum().clamp(min=1e-8)


@dataclass
class TrainHistory:
    step: list = field(default_factory=list)
    loss: list = field(default_factory=list)
    val_step: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    lr: list = field(default_factory=list)
    seconds: float = 0.0


def evaluate_loss(model, data: list[dict], pad_id: int, batch_size: int = 4) -> float:
    """Perte moyenne sur la réponse seule (poids du prompt forcé à 0)."""
    import torch

    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            chunk = [dict(e, weights=[0.0] * e["n_prompt"] + [1.0] * (len(e["input_ids"]) - e["n_prompt"]))
                     for e in data[i:i + batch_size]]
            b = collate(chunk, pad_id, model.device)
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            k = b["weights"][:, 1:].sum().item()
            tot += weighted_lm_loss(logits, b["input_ids"], b["weights"]).item() * k
            n += k
    model.train()
    return tot / n


def train_sft(model, train: list[dict], pad_id: int, epochs: int = 2, batch_size: int = 8, lr: float = 2e-4,
              grad_accum: int = 1, val: list[dict] | None = None, eval_every: int = 10, warmup: int = 5,
              schedule: str = "cosine", seed: int = 0, max_seconds: float = 170.0) -> TrainHistory:
    """Boucle d'entraînement minimale et transparente (AdamW, warmup + décroissance cosinus).

    Seuls les paramètres `requires_grad` sont optimisés (LoRA : les adaptateurs). La boucle
    s'arrête d'elle-même après `max_seconds` (garde-fou pour un GPU partagé).
    """
    import time

    import torch

    g = torch.Generator().manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    n_batches = math.ceil(len(train) / batch_size)
    total = max(1, epochs * n_batches // grad_accum)

    def lr_at(s):
        if s < warmup:
            return lr * (s + 1) / warmup
        if schedule == "constant":
            return lr
        p = (s - warmup) / max(1, total - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    hist = TrainHistory()
    t0 = time.time()
    step, micro = 0, 0
    model.train()
    if val is not None:
        hist.val_step.append(0)
        hist.val_loss.append(evaluate_loss(model, val, pad_id))
    for ep in range(epochs):
        perm = torch.randperm(len(train), generator=g).tolist()
        for bi in range(n_batches):
            chunk = [train[j] for j in perm[bi * batch_size:(bi + 1) * batch_size]]
            b = collate(chunk, pad_id, model.device)
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            loss = weighted_lm_loss(logits, b["input_ids"], b["weights"]) / grad_accum
            loss.backward()
            micro += 1
            if micro % grad_accum:
                continue
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            hist.step.append(step)
            hist.loss.append(loss.item() * grad_accum)
            hist.lr.append(lr_at(step - 1))
            if val is not None and (step % eval_every == 0 or step == total):
                hist.val_step.append(step)
                hist.val_loss.append(evaluate_loss(model, val, pad_id))
            if time.time() - t0 > max_seconds:
                break
        if time.time() - t0 > max_seconds:
            break
    hist.seconds = time.time() - t0
    model.eval()
    return hist


def generate_batch(model, tok, prompts: list[str], max_new_tokens: int = 40, batch_size: int = 16) -> list[str]:
    """Génération gloutonne par lots (remplissage à gauche), sorties décodées sans le prompt."""
    import torch

    old_side = tok.padding_side
    tok.padding_side = "left"
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                 top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    tok.padding_side = old_side
    return [o.strip() for o in outs]


def load_fresh(model_id: str, dtype: str = "bfloat16", **kw):
    """Charge un modèle neuf (hors du cache de `aie.llm`, car on va modifier ses poids)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype), **kw)
    if "device_map" not in kw and "quantization_config" not in kw:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return tok, model


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def lora_ticket_run(train: list, val: list | None = None, eval_sets: dict | None = None, lr: float = 2e-4,
                    epochs: int = 2, batch_size: int = 2, grad_accum: int = 4, prompt_loss_weight: float = 0.0,
                    r: int = 8, alpha: float | None = None, model_id: str | None = None, eval_every: int = 5,
                    schedule: str = "cosine", seed: int = 0, instruction: str = INSTRUCTION) -> dict:
    """Expérience complète de finetuning LoRA sur la tâche des tickets.

    Charge un modèle neuf, ajoute une LoRA (7 matrices par bloc), entraîne avec `train_sft`,
    évalue les jeux `eval_sets` ({nom: liste de Ticket}) et libère le GPU. Renvoie un dict
    {"history", "scores", "outputs", "peak"}.
    """
    import gc

    import torch
    from peft import LoraConfig, get_peft_model

    from . import llm

    model_id = model_id or llm.SMALL
    llm.seed_everything(seed)
    tok, model = load_fresh(model_id)
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha if alpha is not None else 2 * r, lora_dropout=0.0,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    to_ex = lambda ts: [build_sft_example(tok, chat_prompt(tok, instruction, t.message), t.cible, prompt_loss_weight)
                        for t in ts]
    torch.cuda.reset_peak_memory_stats()
    hist = train_sft(model, to_ex(train), tok.pad_token_id, epochs=epochs, batch_size=batch_size, lr=lr,
                     grad_accum=grad_accum, val=to_ex(val) if val else None, eval_every=eval_every,
                     schedule=schedule, seed=seed)
    peak = torch.cuda.max_memory_allocated()
    scores, outputs = {}, {}
    for name, ts in (eval_sets or {}).items():
        outs = generate_batch(model, tok, [chat_prompt(tok, instruction, t.message) for t in ts], 40)
        scores[name], outputs[name] = score_tickets(outs, ts), outs
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"history": hist, "scores": scores, "outputs": outputs, "peak": peak}


def gpu_retry(fn, *args, tries: int = 8, wait: float = 45.0, **kw):
    """Appelle fn ; en cas d'OOM CUDA (GPU partagé), libère la mémoire, attend et réessaie."""
    import gc
    import time

    import torch

    for i in range(tries):
        try:
            return fn(*args, **kw)
        except torch.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if i == tries - 1:
                raise
            time.sleep(wait)


# =====================================================================================
# 9. Fusion de modèles
# =====================================================================================

def linear_merge(vectors, weights=None) -> np.ndarray:
    """Moyenne pondérée sum_i w_i v_i / sum_i w_i."""
    V = np.stack([np.asarray(v, float) for v in vectors])
    w = np.ones(len(V)) if weights is None else np.asarray(weights, float)
    return (w[:, None] * V.reshape(len(V), -1)).sum(0).reshape(V.shape[1:]) / w.sum()


def task_vector(finetuned, base) -> np.ndarray:
    """Vecteur de tâche : poids finetunés moins poids de base."""
    return np.asarray(finetuned, float) - np.asarray(base, float)


def task_arithmetic(base, task_vectors, lambdas) -> np.ndarray:
    """base + sum_i lambda_i tau_i (un lambda négatif retire une capacité)."""
    out = np.asarray(base, float).copy()
    if np.isscalar(lambdas):
        lambdas = [lambdas] * len(task_vectors)
    for tv, lam in zip(task_vectors, lambdas):
        out = out + lam * np.asarray(tv, float)
    return out


def slerp(v0, v1, t: float, eps: float = 1e-8) -> np.ndarray:
    """Interpolation linéaire sphérique entre deux vecteurs (formule de Shoemake).

    On interpole sur le grand cercle des directions, la norme est interpolée linéairement.
    Si les vecteurs sont presque colinéaires, on revient à l'interpolation linéaire.
    """
    v0, v1 = np.asarray(v0, float), np.asarray(v1, float)
    n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
    u0, u1 = v0 / max(n0, eps), v1 / max(n1, eps)
    dot = float(np.clip((u0 * u1).sum(), -1.0, 1.0))
    theta = math.acos(dot)
    if theta < 1e-4:
        return (1 - t) * v0 + t * v1
    s = math.sin(theta)
    u = (math.sin((1 - t) * theta) / s) * u0 + (math.sin(t * theta) / s) * u1
    return u * ((1 - t) * n0 + t * n1)


def trim(tv, density: float) -> np.ndarray:
    """Garde les `density` x 100 % plus grandes composantes (en valeur absolue), met le reste à 0."""
    tv = np.asarray(tv, float)
    k = int(round(density * tv.size))
    if k <= 0:
        return np.zeros_like(tv)
    thr = np.sort(np.abs(tv).ravel())[-k]
    return np.where(np.abs(tv) >= thr, tv, 0.0)


def ties_merge(task_vectors, density: float = 0.2, lam: float = 1.0) -> np.ndarray:
    """TIES (Yadav et al., 2023) : Trim, Elect sign, disjoint merge. Renvoie le vecteur fusionné.

    1. on élague chaque vecteur de tâche (top-k en magnitude) ;
    2. on élit, pour chaque paramètre, le signe de la somme des valeurs élaguées ;
    3. on moyenne seulement les valeurs non nulles qui ont ce signe.
    """
    T = np.stack([trim(tv, density) for tv in task_vectors])
    sign = np.sign(T.sum(0))
    agree = (np.sign(T) == sign) & (T != 0)
    num = (T * agree).sum(0)
    cnt = agree.sum(0)
    merged = np.where(cnt > 0, num / np.maximum(cnt, 1), 0.0)
    return lam * merged


def dare(tv, p: float = 0.9, seed: int = 0) -> np.ndarray:
    """DARE (Yu et al., 2023) : on supprime au hasard une fraction p des composantes et on
    multiplie les survivantes par 1 / (1 - p) (l'espérance du vecteur est conservée)."""
    rng = np.random.default_rng(seed)
    tv = np.asarray(tv, float)
    keep = rng.random(tv.shape) >= p
    return np.where(keep, tv / (1 - p), 0.0)


# --- petits réseaux sur le jeu digits (scikit-learn, embarqué, sans téléchargement) -------

def digits_tasks(seed: int = 0) -> dict:
    """Jeu digits (8x8) et trois « domaines » dérivés, perturbations modérées des images :
    bruit gaussien, occultation d'un carré 4x4, décalage d'un pixel (vers le bas et la droite).

    Renvoie un dict {domaine: (X_train, y_train, X_test, y_test)}, valeurs dans [0, 1].
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    d = load_digits()
    X = d.images / 16.0
    y = d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

    def noisy(im, r):
        return np.clip(im + r.normal(0, 0.4, im.shape), 0, 1)

    def occlude(im, r):
        out = im.copy()
        for k in range(len(out)):
            i, j = r.integers(1, 4), r.integers(1, 4)
            out[k, i:i + 4, j:j + 4] = 0
        return out

    def shift(im, r):
        out = np.zeros_like(im)
        out[:, 1:, 1:] = im[:, :-1, :-1]
        return out

    doms = {"original": lambda im, r: im, "bruité": noisy, "occulté": occlude, "décalé": shift}
    res = {}
    for i, (k, f) in enumerate(doms.items()):
        r = np.random.default_rng(seed + 100 + i)
        res[k] = (f(Xtr, r).reshape(len(Xtr), -1), ytr, f(Xte, r).reshape(len(Xte), -1), yte)
    return res


def make_mlp(seed: int = 0, hidden: int = 64):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(64, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 10))


def fit_mlp(net, X, y, epochs: int = 30, lr: float = 1e-2, batch: int = 64, seed: int = 0):
    """Entraînement simple (Adam, CPU). Modifie `net` en place et le renvoie."""
    import torch

    g = torch.Generator().manual_seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            loss = torch.nn.functional.cross_entropy(net(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


def accuracy(net, X, y) -> float:
    import torch

    with torch.no_grad():
        return float((net(torch.as_tensor(X, dtype=torch.float32)).argmax(1).numpy() == np.asarray(y)).mean())


def get_flat(net) -> np.ndarray:
    import torch

    return torch.nn.utils.parameters_to_vector(net.parameters()).detach().numpy().astype(np.float64)


def set_flat(net, vec) -> None:
    import torch

    torch.nn.utils.vector_to_parameters(torch.as_tensor(vec, dtype=torch.float32), net.parameters())


@dataclass
class DigitsMergeSetup:
    """Base « pré-entraînée » et réseaux finetunés par domaine, pour les expériences de fusion."""
    tasks: dict
    hidden: int
    base: np.ndarray
    finetuned: dict

    @property
    def task_vectors(self) -> dict:
        return {k: v - self.base for k, v in self.finetuned.items()}

    def evaluate(self, vec) -> dict:
        net = make_mlp(0, self.hidden)
        set_flat(net, vec)
        return {k: accuracy(net, t[2], t[3]) for k, t in self.tasks.items()}


def digits_merge_setup(hidden: int = 256, frac: float = 0.15, ft_lr: float = 5e-4, ft_epochs: int = 10,
                       seed: int = 0) -> DigitsMergeSetup:
    """Pré-entraîne un MLP sur les images originales + une petite fraction (`frac`) de chaque
    domaine perturbé, puis le finetune séparément sur chaque domaine (bruité, occulté, décalé).

    Tout se fait sur CPU en environ une seconde.
    """
    tasks = digits_tasks(seed)
    rng = np.random.default_rng(seed)
    Xs, ys = [tasks["original"][0]], [tasks["original"][1]]
    for k in ["bruité", "occulté", "décalé"]:
        idx = rng.choice(len(tasks[k][0]), int(frac * len(tasks[k][0])), replace=False)
        Xs.append(tasks[k][0][idx]); ys.append(tasks[k][1][idx])
    base = fit_mlp(make_mlp(seed, hidden), np.concatenate(Xs), np.concatenate(ys), epochs=20, lr=2e-3, seed=seed)
    vb = get_flat(base)
    fts = {}
    for k in ["bruité", "occulté", "décalé"]:
        net = make_mlp(seed, hidden)
        set_flat(net, vb)
        fit_mlp(net, tasks[k][0], tasks[k][1], epochs=ft_epochs, lr=ft_lr, seed=seed)
        fts[k] = get_flat(net)
    return DigitsMergeSetup(tasks, hidden, vb, fts)
