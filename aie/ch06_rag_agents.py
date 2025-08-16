"""Chapitre 6 : RAG et agents.

Fonctions réutilisables pour les notebooks du chapitre :

- un petit **corpus français** rédigé à la main (10 documents, 64 passages) et un jeu de
  **42 questions annotées** (passages pertinents, extrait-clé, mots attendus dans la réponse) ;
- la **recherche par termes** (tokenisation, index inversé, TF-IDF, BM25 implémentés à la main),
  la **recherche dense** (embeddings + FAISS) et la **fusion RRF** ;
- les **métriques de recherche** (précision@k, rappel@k, MRR, MAP, NDCG) ;
- les **stratégies de découpage** (caractères, mots, phrases, paragraphes, récursif) ;
- le **reranking** (cross-encoder, LLM) et la génération par lots avec Qwen ;
- un **agent** minimal : outils sûrs (calculatrice AST, recherche, date), analyse et validation
  des appels de fonctions JSON, boucle ReAct ;
- la **mémoire** : conversation synthétique et stratégies d'éviction sous budget de tokens ;
- un mini **RAG tabulaire** (SQLite) pour le text-to-SQL.

Tout ce qui ne demande pas de modèle est testable sans GPU (voir tests/test_ch06_rag_agents.py).
"""

from __future__ import annotations

import ast
import functools
import json
import math
import operator
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import numpy as np

# =====================================================================================
# 1. Corpus et questions annotées
# =====================================================================================

#: Chaque document : identifiant court, titre, liste de paragraphes (= passages).
DOCUMENTS: list[dict] = [
    {"doc": "thermo", "title": "Manuel du thermostat connecté Thermo-X200", "paragraphs": [
        "Le Thermo-X200 est un thermostat connecté conçu par la société Domotik pour les maisons équipées d'une chaudière ou d'une pompe à chaleur. Il mesure la température toutes les trente secondes et ajuste le chauffage pièce par pièce. Son écran tactile de 3,5 pouces affiche la consigne et la consommation du jour.",
        "Avant l'installation, coupez l'alimentation électrique au disjoncteur. Le boîtier se fixe au mur avec deux vis, à 1,50 mètre du sol, loin des fenêtres et des radiateurs. Le câblage nécessite un fil neutre ; sans fil neutre, utilisez l'adaptateur AD-3 vendu séparément.",
        "Pour relier l'appareil au réseau sans fil, maintenez le bouton central appuyé pendant cinq secondes jusqu'à ce que le voyant clignote en bleu. Ouvrez ensuite l'application Domotik Home et choisissez votre réseau. Seuls les réseaux 2,4 GHz sont pris en charge ; la bande 5 GHz n'est pas compatible.",
        "Le code E-42 signale une défaillance de la sonde de température interne. Dans ce cas, l'appareil passe en mode sécurité et maintient 16 °C. Si le code persiste après un redémarrage, la sonde doit être remplacée par le service après-vente.",
        "Le code E-17 indique que les piles de secours sont presque vides. Ces piles alimentent l'horloge en cas de coupure de courant. Remplacez-les par deux piles AAA ; vos programmes ne seront pas effacés pendant l'opération.",
        "Le mode vacances abaisse la consigne à 12 °C pendant votre absence et relance le chauffage la veille de votre retour. Il s'active depuis l'application en indiquant les dates de départ et d'arrivée. Pendant ce mode, l'écran s'éteint pour économiser l'énergie.",
        "Pour revenir aux réglages d'usine, maintenez simultanément les deux boutons latéraux pendant quinze secondes. Toutes les programmations et les identifiants Wi-Fi sont alors effacés. Cette opération est conseillée avant de revendre l'appareil.",
        "Le Thermo-X200 est garanti trois ans à compter de la date d'achat, sur présentation de la facture. La garantie ne couvre pas les dommages causés par une installation non conforme ou par une surtension. Une extension de garantie de deux années supplémentaires peut être souscrite en ligne.",
    ]},
    {"doc": "rh", "title": "Guide des ressources humaines de la société Calypso", "paragraphs": [
        "Chaque salarié de Calypso bénéficie de 25 jours ouvrés de congés payés par an, auxquels s'ajoutent 10 jours de RTT pour les cadres au forfait. Les congés se posent dans l'outil Temps & Absences au moins quinze jours à l'avance. Au maximum cinq jours non pris peuvent être reportés sur l'année suivante.",
        "Le télétravail est autorisé jusqu'à trois jours par semaine après la fin de la période d'essai. Les jours de présence obligatoire sont fixés par chaque équipe. Une indemnité forfaitaire de 30 euros par mois couvre les frais de connexion et d'électricité.",
        "Les notes de frais se déclarent dans l'application Dépenses avant le 5 du mois suivant. Un repas d'affaires est remboursé dans la limite de 20 euros par personne, sur justificatif. Les trajets en train se réservent en seconde classe, sauf pour les voyages de plus de quatre heures.",
        "La période d'essai dure deux mois pour les employés et quatre mois pour les cadres. Elle peut être renouvelée une fois, avec l'accord écrit du salarié. Pendant cette période, chaque partie peut mettre fin au contrat en respectant un court délai de prévenance.",
        "En cas d'arrêt maladie, le salarié transmet son avis d'arrêt au service du personnel sous 48 heures. Calypso maintient l'intégralité du salaire dès le premier jour d'absence, sans délai de carence, pendant trois mois.",
        "La mutuelle d'entreprise est obligatoire et prise en charge à 60 % par l'employeur. Les conjoints et les enfants peuvent être ajoutés au contrat moyennant une cotisation complémentaire. Les remboursements d'optique sont plafonnés à 400 euros tous les deux ans.",
        "Chaque collaborateur dispose d'un budget formation de 1 500 euros par an, utilisable pour des cours en ligne, des conférences ou des certifications. Les demandes sont validées par le manager puis par l'équipe Talents. Le temps de formation est compté comme du temps de travail.",
    ]},
    {"doc": "transfo", "title": "Le transformateur électrique", "paragraphs": [
        "Un transformateur électrique modifie la tension d'un courant alternatif sans changer sa fréquence. Il se compose de deux bobines de cuivre enroulées autour d'un noyau de fer feuilleté. Le courant variable dans la bobine primaire crée un champ magnétique qui induit une tension dans la bobine secondaire.",
        "Le rapport de transformation est égal au rapport du nombre de spires des deux enroulements. Si le secondaire compte dix fois moins de spires que le primaire, la tension de sortie est divisée par dix. À puissance constante, le courant est alors multiplié par dix.",
        "Un transformateur réel subit deux familles de pertes : les pertes fer, dues à l'hystérésis et aux courants de Foucault dans le noyau, et les pertes cuivre, dues à l'effet Joule dans les enroulements. Le rendement des gros appareils dépasse néanmoins 99 %.",
        "Les transformateurs de forte puissance sont plongés dans une huile minérale qui isole les enroulements et évacue la chaleur. Des radiateurs extérieurs et parfois des ventilateurs complètent le refroidissement. Une surchauffe prolongée dégrade l'isolant et réduit fortement la durée de vie.",
        "Sur le réseau, l'électricité produite par les centrales est élevée à 400 000 volts pour limiter les pertes en ligne lors du transport. Des postes de transformation abaissent ensuite la tension par étapes, jusqu'aux 230 volts distribués dans les logements.",
        "Les chargeurs de téléphone contenaient autrefois un petit transformateur, aujourd'hui souvent remplacé par une alimentation à découpage. Celle-ci fonctionne à haute fréquence, ce qui permet de réduire la taille du noyau magnétique et le poids du chargeur.",
    ]},
    {"doc": "transformer", "title": "L'architecture Transformer en apprentissage profond", "paragraphs": [
        "L'architecture Transformer a été proposée en 2017 dans l'article « Attention Is All You Need ». Elle remplace les réseaux récurrents par un mécanisme d'attention qui traite tous les tokens d'une séquence en parallèle. Elle est à la base de la plupart des grands modèles de langage actuels.",
        "Le mécanisme d'attention calcule, pour chaque token, une moyenne pondérée des représentations des autres tokens. Les poids sont obtenus par un softmax des produits scalaires entre requêtes et clés, divisés par la racine carrée de la dimension. Chaque token peut ainsi puiser l'information utile n'importe où dans le contexte.",
        "L'attention multi-têtes exécute plusieurs attentions en parallèle, chacune avec ses propres projections. Une tête peut suivre les relations syntaxiques pendant qu'une autre repère les coréférences. Les sorties des têtes sont concaténées puis projetées.",
        "Comme l'attention ignore l'ordre des mots, on ajoute aux embeddings un encodage de position. Le modèle original utilisait des sinusoïdes ; de nombreux modèles récents emploient des encodages rotatifs (RoPE), qui se prolongent mieux vers des contextes longs.",
        "Le coût de l'attention croît avec le carré de la longueur de la séquence, car chaque token est comparé à tous les autres. Doubler la longueur du contexte multiplie donc ce calcul par quatre. Cette contrainte explique les recherches sur l'attention creuse et les fenêtres glissantes.",
        "Lors de la génération, le modèle produit un token à la fois. Pour éviter de recalculer les clés et valeurs des tokens déjà traités, on les conserve dans un cache KV, dont la taille augmente linéairement avec le contexte et devient vite le premier poste de mémoire GPU.",
    ]},
    {"doc": "cinema", "title": "Fiches de films de science-fiction", "paragraphs": [
        "Le film Transformers, réalisé par Michael Bay en 2007, met en scène des robots extraterrestres capables de se changer en véhicules. Il est adapté d'une gamme de jouets lancée dans les années 1980.",
        "Interstellar, réalisé par Christopher Nolan en 2014, suit des astronautes qui traversent un trou de ver près de Saturne pour trouver une planète habitable. Le physicien Kip Thorne a conseillé l'équipe pour représenter le trou noir Gargantua.",
        "Matrix, des Wachowski, sorti en 1999, imagine une humanité prisonnière d'une simulation informatique contrôlée par des machines. Le film a popularisé l'effet « bullet time », qui fige l'action pendant que la caméra tourne autour des personnages.",
        "Dune, adapté du roman de Frank Herbert, se déroule sur la planète désertique Arrakis, seule source de l'épice. La version de Denis Villeneuve est sortie en deux parties, en 2021 et en 2024.",
    ]},
    {"doc": "cuisine", "title": "Carnet de cuisine vietnamienne", "paragraphs": [
        "Le phở est une soupe de nouilles de riz servie dans un bouillon de bœuf parfumé. Le bouillon mijote au moins six heures avec des os, de l'oignon et du gingembre grillés, de l'anis étoilé, de la cannelle et des clous de girofle. On ajoute au moment de servir des herbes fraîches, du citron vert et des pousses de soja.",
        "Le bánh mì est un sandwich né de la rencontre entre la baguette française et les saveurs vietnamiennes. Il contient souvent du porc grillé ou du pâté, des carottes et du radis marinés, de la coriandre et du piment. Le pain doit rester très croustillant et léger.",
        "Les nems, appelés chả giò dans le sud du Vietnam, sont des rouleaux frits garnis de porc haché, de vermicelles, de champignons noirs et de carottes. On les enveloppe dans une feuille de laitue avec de la menthe avant de les tremper dans la sauce.",
        "Les rouleaux de printemps (gỏi cuốn) ne sont pas frits : la galette de riz est simplement réhydratée puis garnie de crevettes, de vermicelles et d'herbes. Ils s'accompagnent d'une sauce aux cacahuètes et au hoisin.",
        "Le nước chấm, sauce de base de la cuisine vietnamienne, mélange de la sauce de poisson, du sucre, du jus de citron vert, de l'eau, de l'ail et du piment. Son équilibre entre salé, sucré et acide se règle au goût.",
        "Pour une version végétarienne du phở, remplacez les os de bœuf par un bouillon de légumes enrichi de champignons shiitakés séchés et de sauce soja. Le tofu frit remplace la viande et absorbe bien les épices.",
        "Le bún chả, spécialité de Hanoï, associe des boulettes et des tranches de porc grillées au charbon de bois, servies dans un bouillon tiède légèrement sucré avec des vermicelles de riz et des herbes.",
    ]},
    {"doc": "solaire", "title": "Petit guide du système solaire", "paragraphs": [
        "Mercure est la planète la plus proche du Soleil et la plus petite du système solaire. Dépourvue d'atmosphère épaisse, elle connaît des écarts de température extrêmes, de plus de 400 °C le jour à environ -180 °C la nuit.",
        "Vénus est la planète la plus chaude, avec une température de surface proche de 470 °C. Son atmosphère très dense de dioxyde de carbone provoque un effet de serre emballé. Elle tourne sur elle-même dans le sens inverse de la plupart des planètes.",
        "Mars abrite Olympus Mons, le plus grand volcan connu du système solaire, qui culmine à environ 22 kilomètres au-dessus de la plaine environnante. Sa couleur rouge vient de l'oxyde de fer présent dans ses poussières.",
        "Jupiter est la plus grosse planète : elle pourrait contenir plus de 1 300 Terres. Sa Grande Tache rouge est une tempête anticyclonique observée depuis plus de trois siècles. On lui connaît aujourd'hui plus de 90 lunes, dont Ganymède, le plus grand satellite du système solaire.",
        "Les anneaux de Saturne sont constitués de milliards de fragments de glace et de roche, dont la taille va du grain de poussière au bloc de plusieurs mètres. Malgré leur largeur de plusieurs centaines de milliers de kilomètres, leur épaisseur ne dépasse souvent pas une dizaine de mètres.",
        "Neptune, la planète la plus éloignée du Soleil, est balayée par les vents les plus rapides du système solaire, qui dépassent 2 000 km/h. Elle a été découverte en 1846 grâce aux calculs d'Urbain Le Verrier, avant même d'être observée.",
        "La lumière du Soleil met environ huit minutes et vingt secondes pour atteindre la Terre. Une unité astronomique, distance moyenne Terre-Soleil, vaut environ 150 millions de kilomètres.",
    ]},
    {"doc": "minou", "title": "Boutique en ligne Minou Chic : mode pour chats", "paragraphs": [
        "Minou Chic est une boutique en ligne spécialisée dans les accessoires de mode pour chats. Tous les articles sont fabriqués en petites séries dans un atelier de Lyon, avec des tissus doux testés pour ne pas irriter la peau des animaux.",
        "Le Fedora Fruité est notre chapeau le plus vendu : un petit feutre orné de cerises en tissu, vendu 18 euros. Il existe en trois coloris (framboise, pistache et myrtille) et s'attache par un élastique réglable.",
        "Le Nœud Papillon Gala, en satin noir, coûte 12 euros et convient aux mariages comme aux photos de famille. L'Imperméable Pluie Douce, doublé de polaire, est proposé à 35 euros pour les sorties sur le balcon.",
        "Pour choisir la bonne taille, mesurez le tour de cou de votre chat avec un mètre ruban souple. Taille S : jusqu'à 22 cm ; taille M : de 22 à 28 cm ; taille L : au-delà de 28 cm. En cas de doute, prenez la taille au-dessus.",
        "La livraison en France métropolitaine est offerte dès 50 euros d'achat ; en dessous, elle coûte 4,90 euros. Les colis sont expédiés sous 48 heures ouvrées et livrés en point relais ou à domicile.",
        "Vous disposez de 30 jours après réception pour retourner un article non porté, dans son emballage d'origine. Le remboursement est effectué sous dix jours sur le moyen de paiement initial. Les articles personnalisés ne sont ni repris ni échangés.",
        "Le programme de fidélité Club Moustache donne un point par euro dépensé. À partir de 100 points, vous recevez un bon de réduction de 10 euros valable sur tout le site.",
    ]},
    {"doc": "reseau", "title": "Diagnostiquer les erreurs réseau sous Linux", "paragraphs": [
        "L'erreur EADDRINUSE (code 98) apparaît lorsqu'un programme tente d'écouter sur un port déjà occupé par un autre processus. La commande ss -ltnp permet d'identifier le processus qui détient le port.",
        "L'erreur EADDRNOTAVAIL (code 99) signifie que l'adresse demandée n'est pas disponible sur la machine : par exemple, un serveur essaie de se lier à une adresse IP qui n'appartient à aucune interface. Elle survient aussi quand les ports éphémères sont épuisés lors de très nombreuses connexions sortantes.",
        "L'erreur ECONNREFUSED (code 111) indique que la machine distante est joignable mais qu'aucun service n'écoute sur le port visé. Vérifiez que le serveur est démarré et que le pare-feu autorise la connexion.",
        "L'erreur ETIMEDOUT (code 110) survient quand une connexion ne reçoit aucune réponse dans le délai imparti. Les causes fréquentes sont un pare-feu qui ignore silencieusement les paquets, une route absente ou une machine éteinte.",
        "L'option SO_REUSEADDR autorise un serveur à réutiliser immédiatement un port encore dans l'état TIME_WAIT après un redémarrage. Sans elle, il faut parfois attendre une à deux minutes avant de pouvoir relancer le service.",
        "Les ports éphémères sont attribués automatiquement aux connexions sortantes. Sous Linux, leur plage se consulte dans le fichier /proc/sys/net/ipv4/ip_local_port_range et va par défaut de 32768 à 60999.",
    ]},
    {"doc": "jardin", "title": "Conseils de jardinage au potager", "paragraphs": [
        "Les tomates se plantent en pleine terre après les dernières gelées, généralement à la mi-mai. Espacez les plants d'environ 60 centimètres et installez un tuteur dès la plantation.",
        "Un compost équilibré alterne déchets verts, riches en azote (épluchures, tontes), et déchets bruns, riches en carbone (feuilles mortes, carton). Il doit rester humide comme une éponge essorée et être retourné tous les mois.",
        "Arrosez de préférence le matin ou le soir, au pied des plantes, pour limiter l'évaporation. Un paillage de paille ou de tontes séchées garde la fraîcheur du sol et réduit les arrosages de moitié.",
        "Le mildiou est une maladie causée par un micro-organisme qui forme des taches brunes sur les feuilles de tomates et de pommes de terre par temps chaud et humide. Supprimez les feuilles atteintes et évitez de mouiller le feuillage.",
        "Les associations de cultures aident le potager : le basilic éloigne certains insectes des tomates, et les œillets d'Inde repoussent les nématodes. À l'inverse, on évite de planter les haricots à côté de l'ail.",
        "La rotation des cultures consiste à ne pas cultiver la même famille de légumes au même endroit deux années de suite. Elle limite l'épuisement du sol et la propagation des maladies.",
    ]},
]


@dataclass
class Passage:
    pid: str
    doc: str
    title: str
    text: str


def passages() -> list[Passage]:
    """Les 64 passages du corpus (un paragraphe = un passage), identifiants « doc-numéro »."""
    out = []
    for d in DOCUMENTS:
        for i, p in enumerate(d["paragraphs"], start=1):
            out.append(Passage(f"{d['doc']}-{i}", d["doc"], d["title"], p))
    return out


def document_text(doc: dict) -> str:
    """Texte intégral d'un document (paragraphes séparés par une ligne vide)."""
    return "\n\n".join(doc["paragraphs"])


def _q(qid, q, rel, key, ans, kind):
    rel = {rel: 1} if isinstance(rel, str) else dict(rel)
    return {"qid": qid, "question": q, "relevant": rel, "key": key, "answers": ans, "type": kind}


#: 42 questions annotées. relevant : {passage: note} (2 = très pertinent, 1 = pertinent) ;
#: key : extrait qui contient la réponse (sert à juger un découpage quelconque) ;
#: answers : variantes acceptées dans une réponse générée ; type : lexical / semantique / ambigu.
QUESTIONS: list[dict] = [
    _q("q01", "Que signifie le code E-42 sur le Thermo-X200 ?", "thermo-4", "sonde de température interne", ["sonde"], "lexical"),
    _q("q02", "Mon thermostat affiche que les piles sont faibles, que faire ?", "thermo-5", "deux piles AAA", ["AAA"], "semantique"),
    _q("q03", "Comment connecter le thermostat au Wi-Fi ?", "thermo-3", "bouton central appuyé pendant cinq secondes", ["cinq secondes", "5 secondes", "bouton central"], "semantique"),
    _q("q04", "Le thermostat fonctionne-t-il avec un réseau 5 GHz ?", "thermo-3", "la bande 5 GHz n'est pas compatible", ["pas compatible", "non", "2,4"], "lexical"),
    _q("q05", "Quelle est la durée de la garantie du Thermo-X200 ?", "thermo-8", "garanti trois ans", ["trois ans", "3 ans"], "lexical"),
    _q("q06", "Comment remettre l'appareil Domotik dans son état d'origine avant de le vendre ?", "thermo-7", "réglages d'usine", ["quinze secondes", "15 secondes", "boutons latéraux"], "semantique"),
    _q("q07", "Quelle température le thermostat maintient-il quand je pars en congé ?", "thermo-6", "abaisse la consigne à 12 °C", ["12"], "semantique"),
    _q("q08", "Combien de jours de vacances ai-je par an chez Calypso ?", "rh-1", "25 jours ouvrés de congés payés", ["25"], "semantique"),
    _q("q09", "Combien de jours par semaine puis-je travailler depuis chez moi ?", "rh-2", "jusqu'à trois jours par semaine", ["trois", "3"], "semantique"),
    _q("q10", "Quel est le plafond de remboursement d'un repas d'affaires ?", "rh-3", "dans la limite de 20 euros", ["20"], "lexical"),
    _q("q11", "Combien de temps dure la période d'essai d'un cadre ?", "rh-4", "quatre mois pour les cadres", ["quatre mois", "4 mois"], "lexical"),
    _q("q12", "Suis-je payé si je tombe malade dès le premier jour ?", "rh-5", "sans délai de carence", ["carence", "intégralité", "premier jour"], "semantique"),
    _q("q13", "Quel montant puis-je dépenser pour me former chaque année ?", "rh-7", "budget formation de 1 500 euros", ["1 500", "1500"], "semantique"),
    _q("q14", "Quelle part de la mutuelle est payée par l'employeur ?", "rh-6", "prise en charge à 60 %", ["60"], "lexical"),
    _q("q15", "Comment fonctionne un transformateur électrique ?", {"transfo-1": 2, "transfo-2": 1}, "induit une tension dans la bobine secondaire", ["bobine", "champ magnétique", "induction", "induit"], "ambigu"),
    _q("q16", "Comment l'architecture transformer traite-t-elle les mots d'une phrase ?", {"transformer-2": 2, "transformer-1": 1}, "moyenne pondérée des représentations", ["attention"], "ambigu"),
    _q("q17", "Pourquoi le coût de calcul explose-t-il avec des contextes longs dans les modèles de langage ?", "transformer-5", "carré de la longueur de la séquence", ["carré", "quadratique"], "semantique"),
    _q("q18", "À quoi sert le cache KV ?", "transformer-6", "recalculer les clés et valeurs", ["recalcul"], "lexical"),
    _q("q19", "Qui a réalisé le film Transformers ?", "cinema-1", "Michael Bay", ["Michael Bay"], "ambigu"),
    _q("q20", "Pourquoi place-t-on les gros transformateurs dans de l'huile ?", "transfo-4", "isole les enroulements et évacue la chaleur", ["isol", "chaleur", "refroid"], "lexical"),
    _q("q21", "À quelle tension l'électricité est-elle transportée sur les lignes ?", "transfo-5", "400 000 volts", ["400 000", "400000", "400 kV"], "semantique"),
    _q("q22", "Quelles épices parfument le bouillon du phở ?", "cuisine-1", "de l'anis étoilé, de la cannelle", ["anis", "cannelle"], "lexical"),
    _q("q23", "Comment préparer une soupe phở sans viande ?", "cuisine-6", "bouillon de légumes", ["légumes", "tofu"], "semantique"),
    _q("q24", "Quelle est la différence entre les nems et les rouleaux de printemps ?", {"cuisine-4": 2, "cuisine-3": 1}, "ne sont pas frits", ["frit"], "lexical"),
    _q("q25", "Quels ingrédients faut-il pour la sauce au poisson vietnamienne ?", "cuisine-5", "sauce de poisson, du sucre", ["sucre", "citron"], "semantique"),
    _q("q26", "Quelle est la spécialité de porc grillé de Hanoï ?", "cuisine-7", "Le bún chả", ["bún chả", "bun cha"], "lexical"),
    _q("q27", "Quelle est la planète la plus chaude ?", "solaire-2", "la planète la plus chaude", ["Vénus", "Venus"], "lexical"),
    _q("q28", "Où se trouve la plus haute montagne du système solaire ?", "solaire-3", "Olympus Mons", ["Mars", "Olympus"], "semantique"),
    _q("q29", "Combien de temps la lumière solaire met-elle à nous parvenir ?", "solaire-7", "huit minutes et vingt secondes", ["huit minutes", "8 minutes"], "semantique"),
    _q("q30", "Qui a prédit l'existence de Neptune par le calcul ?", "solaire-6", "Urbain Le Verrier", ["Le Verrier"], "lexical"),
    _q("q31", "De quoi sont faits les anneaux de Saturne ?", "solaire-5", "fragments de glace et de roche", ["glace"], "lexical"),
    _q("q32", "Combien coûte le Fedora Fruité ?", "minou-2", "vendu 18 euros", ["18"], "lexical"),
    _q("q33", "À partir de quel montant la livraison est-elle gratuite chez Minou Chic ?", "minou-5", "offerte dès 50 euros", ["50"], "semantique"),
    _q("q34", "Mon chat a un cou de 25 cm, quelle taille commander ?", "minou-4", "de 22 à 28 cm", ["taille M", "M"], "semantique"),
    _q("q35", "Puis-je renvoyer un article qui ne me plaît pas ?", "minou-6", "30 jours après réception", ["30 jours"], "semantique"),
    _q("q36", "Que signifie l'erreur EADDRNOTAVAIL ?", "reseau-2", "l'adresse demandée n'est pas disponible", ["adresse"], "lexical"),
    _q("q37", "Mon serveur ne peut pas redémarrer car le port est encore occupé, comment l'éviter ?", {"reseau-5": 2, "reseau-1": 1}, "SO_REUSEADDR", ["SO_REUSEADDR"], "semantique"),
    _q("q38", "Quel code d'erreur obtient-on quand aucun service n'écoute sur le port distant ?", "reseau-3", "ECONNREFUSED (code 111)", ["ECONNREFUSED", "111"], "semantique"),
    _q("q39", "Quand planter les tomates ?", "jardin-1", "à la mi-mai", ["mai"], "lexical"),
    _q("q40", "Comment économiser l'eau au potager ?", "jardin-3", "réduit les arrosages de moitié", ["paillage", "matin", "soir"], "semantique"),
    _q("q41", "Quelles taches brunes apparaissent sur les feuilles de tomate ?", "jardin-4", "Le mildiou", ["mildiou"], "lexical"),
    _q("q42", "Quelle est la plage par défaut des ports éphémères sous Linux ?", "reseau-6", "de 32768 à 60999", ["32768"], "lexical"),
]

#: Questions conversationnelles (suivi ambigu) pour la réécriture de requête.
CONVERSATIONS: list[dict] = [
    {"history": [("user", "Combien dure la période d'essai chez Calypso ?"), ("assistant", "Deux mois pour les employés, quatre mois pour les cadres.")],
     "followup": "Peut-on la renouveler ?", "gold": "Peut-on renouveler la période d'essai chez Calypso ?", "relevant": {"rh-4": 1}},
    {"history": [("user", "Quelle est la plus grosse planète du système solaire ?"), ("assistant", "C'est Jupiter.")],
     "followup": "Quelle est sa particularité la plus célèbre ?", "gold": "Quelle est la particularité la plus célèbre de Jupiter ?", "relevant": {"solaire-4": 1}},
    {"history": [("user", "Que veut dire l'erreur EADDRINUSE ?"), ("assistant", "Le port est déjà occupé par un autre processus.")],
     "followup": "Comment savoir quel programme est en cause ?", "gold": "Comment savoir quel programme occupe le port (erreur EADDRINUSE) ?", "relevant": {"reseau-1": 1}},
    {"history": [("user", "Comment se prépare le phở ?"), ("assistant", "C'est une soupe de nouilles de riz dans un bouillon de bœuf parfumé.")],
     "followup": "Et quelles herbes ajoute-t-on à la fin ?", "gold": "Quelles herbes ajoute-t-on au phở au moment de servir ?", "relevant": {"cuisine-1": 1}},
    {"history": [("user", "Le télétravail est-il autorisé chez Calypso ?"), ("assistant", "Oui, jusqu'à trois jours par semaine.")],
     "followup": "Et à partir de quand y a-t-on droit ?", "gold": "À partir de quand a-t-on droit au télétravail chez Calypso ?", "relevant": {"rh-2": 1}},
    {"history": [("user", "J'ai acheté un Fedora Fruité chez Minou Chic."), ("assistant", "Excellent choix, c'est le chapeau le plus vendu !")],
     "followup": "Je peux le rendre s'il ne va pas à mon chat ?", "gold": "Peut-on retourner un article Minou Chic qui ne convient pas ?", "relevant": {"minou-6": 1}},
    {"history": [("user", "Quand faut-il planter les tomates ?"), ("assistant", "À la mi-mai, après les dernières gelées.")],
     "followup": "À quelle distance faut-il les espacer ?", "gold": "À quelle distance faut-il espacer les plants de tomates ?", "relevant": {"jardin-1": 1}},
    {"history": [("user", "Parle-moi du thermostat Thermo-X200."), ("assistant", "C'est un thermostat connecté de la société Domotik.")],
     "followup": "Combien de temps est-il couvert en cas de panne ?", "gold": "Combien de temps le thermostat Thermo-X200 est-il garanti ?", "relevant": {"thermo-8": 1}},
]


# =====================================================================================
# 2. Tokenisation et recherche par termes
# =====================================================================================

STOPWORDS_FR = set("""
a à au aux avec ce ces cet cette dans de des du elle elles en et eux il ils je la le les leur
leurs lui ma mais me même mes moi mon ne nos notre nous on ou où par pas pour qu que qui sa se
ses son sur ta te tes toi ton tu un une vos votre vous c d j l m n s t y été être est sont ai as
avons avez ont avait était fait faire peut puis plus moins très tout tous toute toutes si sans
sous entre comme aussi alors donc car quel quelle quels quelles quoi comment pourquoi quand
combien dont ici là cela ça ceci celui celle ceux celles chaque autre autres afin lors
""".split())


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


_TOKEN_RE = re.compile(r"[a-z0-9_]+(?:-[a-z0-9]+)*")


def tokenize(text: str, stopwords: bool = True, stem: bool = True) -> list[str]:
    """Tokenisation simple pour la recherche par termes.

    minuscules, suppression des accents, découpe sur tout ce qui n'est pas alphanumérique
    (les codes comme « e-42 » ou « thermo-x200 » restent entiers), suppression des mots
    vides, « racinisation » minimale (pluriels en -s / -x retirés).
    """
    text = strip_accents(text.lower()).replace("'", " ").replace("’", " ")
    toks = _TOKEN_RE.findall(text)
    if stopwords:
        sw = {strip_accents(w) for w in STOPWORDS_FR}
        toks = [t for t in toks if t not in sw]
    if stem:
        toks = [t[:-1] if len(t) > 3 and t[-1] in "sx" and not t[-2].isdigit() else t for t in toks]
    return toks


def build_inverted_index(docs_tokens: list[list[str]]) -> dict[str, list[tuple[int, int]]]:
    """Index inversé : terme -> liste de (indice du document, fréquence du terme)."""
    index: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, toks in enumerate(docs_tokens):
        for t, c in Counter(toks).items():
            index[t].append((i, c))
    return dict(index)


class TfIdf:
    """TF-IDF « naïf » tel que décrit dans le livre : score(D, Q) = Σ IDF(t) · f(t, D),
    avec IDF(t) = log(N / C(t))."""

    def __init__(self, docs_tokens: list[list[str]]):
        self.N = len(docs_tokens)
        self.index = build_inverted_index(docs_tokens)
        self.idf = {t: math.log(self.N / len(post)) for t, post in self.index.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.N)
        for t in query_tokens:
            for i, f in self.index.get(t, []):
                s[i] += self.idf[t] * f
        return s


class BM25:
    """Okapi BM25 implémenté à la main.

    score(D, Q) = Σ_t IDF(t) · f(t,D)·(k1+1) / (f(t,D) + k1·(1 - b + b·|D|/avgdl))

    variant="okapi" : IDF(t) = log((N - n_t + 0.5) / (n_t + 0.5)), les IDF négatives étant
    remplacées par epsilon × (IDF moyenne), exactement comme ``rank_bm25.BM25Okapi`` ;
    variant="lucene" : IDF(t) = log(1 + (N - n_t + 0.5) / (n_t + 0.5)), toujours positive.
    """

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75,
                 epsilon: float = 0.25, variant: str = "okapi"):
        self.k1, self.b, self.variant = k1, b, variant
        self.N = len(docs_tokens)
        self.doc_len = np.array([len(d) for d in docs_tokens], dtype=float)
        self.avgdl = self.doc_len.mean()
        self.index = build_inverted_index(docs_tokens)
        self.idf: dict[str, float] = {}
        for t, post in self.index.items():
            n = len(post)
            if variant == "lucene":
                self.idf[t] = math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            else:
                self.idf[t] = math.log(self.N - n + 0.5) - math.log(n + 0.5)
        if variant == "okapi":
            avg = sum(self.idf.values()) / len(self.idf)
            for t, v in self.idf.items():
                if v < 0:
                    self.idf[t] = epsilon * avg

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.N)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        for t in query_tokens:
            for i, f in self.index.get(t, []):
                s[i] += self.idf[t] * f * (self.k1 + 1) / (f + norm[i])
        return s


def rank(scores: np.ndarray, ids: list[str], k: int | None = None) -> list[str]:
    """Identifiants triés par score décroissant (départage stable par ordre d'origine)."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    if k is not None:
        order = order[:k]
    return [ids[i] for i in order]


class TermRetriever:
    """Recherche par termes sur une liste de textes (BM25 ou TF-IDF)."""

    def __init__(self, texts: list[str], ids: list[str], method: str = "bm25", **kw):
        self.ids = list(ids)
        toks = [tokenize(t) for t in texts]
        self.model = BM25(toks, **kw) if method == "bm25" else TfIdf(toks)

    def scores(self, query: str) -> np.ndarray:
        return self.model.scores(tokenize(query))

    def search(self, query: str, k: int = 10) -> list[str]:
        return rank(self.scores(query), self.ids, k)


# =====================================================================================
# 3. Recherche dense
# =====================================================================================

class DenseRetriever:
    """Recherche par embeddings : vecteurs normalisés + produit scalaire (= cosinus) dans FAISS."""

    def __init__(self, texts: list[str], ids: list[str], model_id: str | None = None,
                 embed_fn: Callable | None = None):
        import faiss

        from . import llm

        self.ids = list(ids)
        self.model_id = model_id or llm.EMBED_MULTI
        self.embed_fn = embed_fn or (lambda xs: llm.embed(xs, self.model_id))
        self.emb = np.asarray(self.embed_fn(list(texts)), dtype="float32")
        self.index = faiss.IndexFlatIP(self.emb.shape[1])
        self.index.add(self.emb)

    def scores(self, query: str) -> np.ndarray:
        q = np.asarray(self.embed_fn([query]), dtype="float32")
        return (self.emb @ q[0]).astype(float)

    def search(self, query: str, k: int = 10) -> list[str]:
        q = np.asarray(self.embed_fn([query]), dtype="float32")
        _, I = self.index.search(q, k)
        return [self.ids[i] for i in I[0] if i >= 0]


def rrf(rankings: list[list[str]], k: int = 60, weights: list[float] | None = None) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion : score(D) = Σ_i w_i / (k + r_i(D)), rangs comptés à partir de 1."""
    weights = weights or [1.0] * len(rankings)
    s: dict[str, float] = defaultdict(float)
    for w, r in zip(weights, rankings):
        for pos, d in enumerate(r, start=1):
            s[d] += w / (k + pos)
    return sorted(s.items(), key=lambda kv: (-kv[1], kv[0]))


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    span = x.max() - x.min()
    return np.zeros_like(x) if span == 0 else (x - x.min()) / span


# =====================================================================================
# 4. Métriques de recherche
# =====================================================================================

def precision_at_k(retrieved: list[str], relevant, k: int) -> float:
    return sum(1 for d in retrieved[:k] if d in relevant) / k


def recall_at_k(retrieved: list[str], relevant, k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for d in retrieved[:k] if d in relevant) / len(relevant)


def hit_at_k(retrieved: list[str], relevant, k: int) -> float:
    """1 si au moins un document pertinent figure dans les k premiers."""
    return float(any(d in relevant for d in retrieved[:k]))


def reciprocal_rank(retrieved: list[str], relevant, k: int | None = None) -> float:
    for i, d in enumerate(retrieved[:k] if k else retrieved, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def average_precision(retrieved: list[str], relevant, k: int | None = None) -> float:
    hits, total = 0, 0.0
    for i, d in enumerate(retrieved[:k] if k else retrieved, start=1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant) if relevant else 0.0


def dcg(gains: list[float]) -> float:
    return float(sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1)))


def ndcg_at_k(retrieved: list[str], relevant, k: int) -> float:
    """NDCG@k à gains linéaires : relevant est un dict {doc: note} (ou un ensemble : note 1)."""
    grades = relevant if isinstance(relevant, dict) else {d: 1 for d in relevant}
    gains = [grades.get(d, 0) for d in retrieved[:k]]
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def evaluate(run: dict[str, list[str]], questions: list[dict] | None = None, k: int = 5) -> dict[str, float]:
    """Moyenne des métriques sur un « run » {qid: classement}."""
    questions = questions or QUESTIONS
    rows = []
    for q in questions:
        r, rel = run[q["qid"]], q["relevant"]
        rows.append([precision_at_k(r, rel, k), recall_at_k(r, rel, k), hit_at_k(r, rel, k),
                     reciprocal_rank(r, rel, k), ndcg_at_k(r, rel, k)])
    m = np.mean(rows, axis=0)
    return {f"P@{k}": m[0], f"R@{k}": m[1], f"Hit@{k}": m[2], "MRR": m[3], f"NDCG@{k}": m[4]}


def per_question(run: dict[str, list[str]], questions: list[dict] | None = None, k: int = 5):
    import pandas as pd

    questions = questions or QUESTIONS
    return pd.DataFrame([{"qid": q["qid"], "type": q["type"], "RR": reciprocal_rank(run[q["qid"]], q["relevant"], k),
                          f"R@{k}": recall_at_k(run[q["qid"]], q["relevant"], k),
                          f"NDCG@{k}": ndcg_at_k(run[q["qid"]], q["relevant"], k)} for q in questions])


# =====================================================================================
# 5. Découpage (chunking)
# =====================================================================================

def chunk_chars(text: str, size: int, overlap: int = 0) -> list[str]:
    """Morceaux de `size` caractères ; deux morceaux consécutifs partagent `overlap` caractères."""
    if overlap >= size:
        raise ValueError("overlap doit être < size")
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]


def chunk_words(text: str, size: int, overlap: int = 0) -> list[str]:
    words = text.split()
    if overlap >= size:
        raise ValueError("overlap doit être < size")
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Ý0-9«])")


def split_sentences(text: str) -> list[str]:
    out = []
    for para in text.split("\n\n"):
        out.extend(s.strip() for s in _SENT_RE.split(para) if s.strip())
    return out


def chunk_sentences(text: str, n: int = 2, overlap: int = 0) -> list[str]:
    sents = split_sentences(text)
    step = max(n - overlap, 1)
    return [" ".join(sents[i:i + n]) for i in range(0, max(len(sents) - overlap, 1), step)]


def chunk_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_recursive(text: str, max_chars: int, separators=("\n\n", ". ", " ")) -> list[str]:
    """Découpage récursif : on coupe sur le séparateur le plus « gros » ; un morceau trop long
    est redécoupé avec le séparateur suivant ; les petits morceaux voisins sont regroupés."""
    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []
    if not separators:
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    sep, rest = separators[0], separators[1:]
    parts = [p for p in text.split(sep) if p.strip()]
    if sep == ". ":
        parts = [p if p.endswith(".") else p + "." for p in parts]
        sep = " "
    pieces: list[str] = []
    for p in parts:
        pieces.extend(chunk_recursive(p, max_chars, rest) if len(p) > max_chars else [p.strip()])
    merged: list[str] = []
    for p in pieces:
        if merged and len(merged[-1]) + len(sep) + len(p) <= max_chars:
            merged[-1] = merged[-1] + sep + p
        else:
            merged.append(p)
    return merged


def make_chunks(strategy: Callable[[str], list[str]], docs: list[dict] | None = None):
    """Applique une stratégie de découpage à chaque document. Renvoie (textes, ids, docs)."""
    docs = docs or DOCUMENTS
    texts, ids, owners = [], [], []
    for d in docs:
        for j, c in enumerate(strategy(document_text(d))):
            texts.append(c)
            ids.append(f"{d['doc']}#{j}")
            owners.append(d["doc"])
    return texts, ids, owners


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def chunk_relevance(chunk_texts: list[str], chunk_ids: list[str], key: str) -> set[str]:
    """Morceaux « pertinents » pour une question : ceux qui contiennent l'extrait-clé en entier."""
    k = _norm_ws(key)
    return {cid for cid, t in zip(chunk_ids, chunk_texts) if k in _norm_ws(t)}


# =====================================================================================
# 6. Génération : Qwen par lots, reranking, RAG
# =====================================================================================

@functools.lru_cache(maxsize=1)
def load_cpu(model_id: str):
    """Charge un modèle causal sur CPU (float32), pour épargner le GPU partagé (ex. Qwen 1,5B)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).eval()
    return tok, model


def generate_batch(conversations: list, model_id: str | None = None, max_new_tokens: int = 64,
                   batch_size: int = 16, tools: list | None = None, device: str | None = None) -> list[str]:
    """Génère (glouton) une réponse pour chaque conversation, par lots avec remplissage à gauche.

    device="cpu" : charge le modèle sur CPU (utile pour un modèle de 1,5 B quand le GPU est partagé).
    """
    import torch

    from . import llm

    model_id = model_id or llm.SMALL
    tok, model = load_cpu(model_id) if device == "cpu" else llm.load(model_id)
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(conversations), batch_size):
        batch = conversations[i:i + batch_size]
        batch = [[{"role": "user", "content": c}] if isinstance(c, str) else c for c in batch]
        prompts = [tok.apply_chat_template(m, tools=tools, add_generation_prompt=True, tokenize=False) for m in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        for row in out[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True).strip())
    return outs


def yes_probability(prompts: list[str], model_id: str | None = None, batch_size: int = 16,
                    yes: str = "Oui", no: str = "Non") -> np.ndarray:
    """P(« Oui ») / (P(« Oui ») + P(« Non »)) pour le premier token de la réponse du modèle."""
    import torch

    from . import llm

    model_id = model_id or llm.SMALL
    tok, model = llm.load(model_id)
    tok.padding_side = "left"
    y_id, n_id = tok.encode(yes)[0], tok.encode(no)[0]
    res = []
    for i in range(0, len(prompts), batch_size):
        msgs = [[{"role": "user", "content": p}] for p in prompts[i:i + batch_size]]
        texts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1].float()
        two = torch.stack([logits[:, y_id], logits[:, n_id]], dim=1)
        res.extend(torch.softmax(two, dim=1)[:, 0].cpu().numpy().tolist())
    return np.array(res)


LLM_RERANK_PROMPT = ("Question : {q}\n\nPassage : {p}\n\n"
                     "Ce passage contient-il la réponse à la question ? Réponds uniquement par Oui ou Non.")


def llm_rerank_scores(question: str, texts: list[str], model_id: str | None = None) -> np.ndarray:
    """Reranking « point par point » par un LLM : probabilité qu'il réponde Oui."""
    return yes_probability([LLM_RERANK_PROMPT.format(q=question, p=t) for t in texts], model_id)


@functools.lru_cache(maxsize=1)
def load_cross_encoder(model_id: str = "BAAI/bge-reranker-v2-m3"):
    """Cross-encoder multilingue (≈ 570 M paramètres, ≈ 1,1 Go en float16). Repli sur CPU si
    le GPU partagé est plein."""
    import torch
    from sentence_transformers import CrossEncoder
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    try:
        return CrossEncoder(model_id, device="cuda" if torch.cuda.is_available() else "cpu",
                            model_kwargs={"torch_dtype": torch.float16})
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return CrossEncoder(model_id, device="cpu")


def cross_encoder_scores(question: str, texts: list[str]) -> np.ndarray:
    ce = load_cross_encoder()
    return np.asarray(ce.predict([(question, t) for t in texts], show_progress_bar=False), dtype=float)


def free_models() -> None:
    """Libère le cross-encoder et les modèles du socle (GPU partagé)."""
    from . import llm

    load_cross_encoder.cache_clear()
    load_cpu.cache_clear()
    llm.free_gpu()


RAG_SYSTEM = ("Tu es un assistant qui répond en français, en une ou deux phrases, "
              "uniquement à partir des documents fournis. Si la réponse n'y figure pas, "
              "réponds « Je ne sais pas ».")


def build_rag_prompt(question: str, contexts: list[str]) -> str:
    """Gabarit de prompt RAG : documents numérotés puis question."""
    docs = "\n\n".join(f"[Document {i}]\n{c}" for i, c in enumerate(contexts, start=1))
    return f"Documents :\n{docs}\n\nQuestion : {question}\nRéponse :"


def normalize_text(s: str) -> str:
    return " ".join(strip_accents(s.lower()).replace(" ", " ").replace("\xa0", " ").split())


def answer_ok(answer: str, expected: list[str]) -> bool:
    """Vrai si l'une des variantes attendues apparaît dans la réponse (sans casse ni accents).
    Une variante d'un ou deux caractères (« M », « 3 ») doit apparaître comme mot entier."""
    a = normalize_text(answer)
    for e in expected:
        e = normalize_text(e)
        if len(e) <= 2:
            if re.search(rf"(?<![\w]){re.escape(e)}(?![\w])", a):
                return True
        elif e in a:
            return True
    return False


def abstains(answer: str) -> bool:
    a = normalize_text(answer)
    return any(x in a for x in ["je ne sais pas", "ne figure pas", "pas d'information", "aucune information",
                                "ne mentionne pas", "pas mentionne", "ne contient pas"])


# =====================================================================================
# 7. Agents : outils, appels de fonctions, boucle ReAct
# =====================================================================================

_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "racine": math.sqrt, "abs": abs, "round": round, "arrondi": round,
          "min": min, "max": max, "log": math.log, "exp": math.exp}


def safe_eval(expr: str) -> float:
    """Calculatrice sûre : n'évalue que nombres, + - * / // % **, parenthèses et quelques
    fonctions (sqrt, round, abs, min, max, log, exp). Jamais `eval` : aucun code arbitraire."""
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/")
    expr = re.sub(r"(?<=\d)[  \xa0](?=\d{3}\b)", "", expr)  # « 2 480 » -> « 2480 »
    if not re.search(r"[a-z]\s*\(", expr):  # virgule décimale française hors appel de fonction
        expr = re.sub(r"(?<=\d),(?=\d)", ".", expr)

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exposant trop grand")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"expression non autorisée : {ast.dump(node)[:40]}")

    if len(expr) > 200:
        raise ValueError("expression trop longue")
    return ev(ast.parse(expr, mode="eval"))


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # schéma JSON (type object)
    fn: Callable
    write: bool = False  # action d'écriture (à soumettre à validation humaine)

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


#: Date « du jour » figée pour rendre les notebooks reproductibles.
TODAY = date(2026, 11, 25)


def make_tools(search_fn: Callable[[str], list[str]] | None = None, today: date = TODAY) -> dict[str, Tool]:
    """Inventaire d'outils sûrs (lecture seule) : calculatrice, recherche dans le corpus, date,
    écart en jours entre deux dates."""
    def calculatrice(expression: str):
        v = safe_eval(str(expression))
        return round(v, 6) if isinstance(v, float) else v

    def recherche_corpus(requete: str):
        if search_fn is None:
            return "Aucun moteur de recherche configuré."
        return "\n".join(search_fn(str(requete)))

    def date_du_jour():
        return today.isoformat()

    def jours_entre(date_debut: str, date_fin: str):
        return (date.fromisoformat(str(date_fin)) - date.fromisoformat(str(date_debut))).days

    S = lambda props, req: {"type": "object", "properties": props, "required": req}  # noqa: E731
    tools = [
        Tool("calculatrice", "Évalue une expression arithmétique (+, -, *, /, **, sqrt, round). Exemple : '3 * 18'.",
             S({"expression": {"type": "string", "description": "expression arithmétique à calculer"}}, ["expression"]), calculatrice),
        Tool("recherche_corpus", "Cherche dans la base documentaire interne (produits Minou Chic, thermostat Thermo-X200, RH de Calypso, cuisine, astronomie, réseau Linux, jardinage) et renvoie les passages pertinents.",
             S({"requete": {"type": "string", "description": "la question ou les mots-clés à chercher"}}, ["requete"]), recherche_corpus),
        Tool("date_du_jour", "Renvoie la date d'aujourd'hui au format AAAA-MM-JJ.", S({}, []), date_du_jour),
        Tool("jours_entre", "Nombre de jours entre deux dates au format AAAA-MM-JJ.",
             S({"date_debut": {"type": "string"}, "date_fin": {"type": "string"}}, ["date_debut", "date_fin"]), jours_entre),
    ]
    return {t.name: t for t in tools}


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)


def parse_tool_calls(text: str) -> tuple[str, list]:
    """Extrait les appels d'outils d'une sortie de modèle.

    Format Qwen : <tool_call>{"name": ..., "arguments": {...}}</tool_call>. On accepte aussi un
    objet JSON nu contenant "name". Renvoie (texte hors appels = « pensée » ou réponse, appels) ;
    un appel illisible est renvoyé sous la forme {"_raw": texte} (JSON invalide).
    """
    calls = []
    blocks = _TOOL_CALL_RE.findall(text)
    if not blocks:
        m = re.search(r"\{.*\"name\".*\}", text, re.S)
        if m:
            blocks = [m.group(0)]
            text = text.replace(m.group(0), "")
    else:
        text = _TOOL_CALL_RE.sub("", text)
    for b in blocks:
        try:
            obj = json.loads(b)
            if isinstance(obj, dict) and "arguments" in obj and isinstance(obj["arguments"], str):
                obj["arguments"] = json.loads(obj["arguments"])
            calls.append(obj)
        except (json.JSONDecodeError, TypeError):
            calls.append({"_raw": b})
    return text.strip(), calls


_JSON_TYPES = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "object": dict, "array": list}


def validate_call(call: dict, tools: dict[str, Tool]) -> str:
    """Classe un appel d'outil selon la typologie d'échecs du livre (section *Planning failures*) :
    "ok", "json_invalide", "outil_invalide", "parametres_invalides"."""
    if "_raw" in call or not isinstance(call, dict) or "name" not in call:
        return "json_invalide"
    if call["name"] not in tools:
        return "outil_invalide"
    args = call.get("arguments", {}) or {}
    if not isinstance(args, dict):
        return "parametres_invalides"
    schema = tools[call["name"]].parameters
    props, req = schema.get("properties", {}), schema.get("required", [])
    if any(r not in args for r in req) or any(a not in props for a in args):
        return "parametres_invalides"
    for a, v in args.items():
        t = props[a].get("type")
        if t and not isinstance(v, _JSON_TYPES[t]):
            if not (t == "string" and isinstance(v, (int, float))):  # tolérance : nombre pour une chaîne
                return "parametres_invalides"
    return "ok"


@dataclass
class Step:
    thought: str
    call: dict | None
    status: str
    observation: str


@dataclass
class AgentResult:
    question: str
    answer: str | None
    steps: list[Step] = field(default_factory=list)
    finished: bool = False

    @property
    def n_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.call is not None)

    @property
    def tools_used(self) -> list[str]:
        return [s.call.get("name", "?") for s in self.steps if s.call is not None]

    def trace(self, width: int = 110) -> str:
        lines = [f"Question : {self.question}"]
        for i, s in enumerate(self.steps, start=1):
            if s.thought:
                lines.append(f"  Pensée {i}      : {s.thought[:width]}")
            if s.call is not None:
                c = json.dumps({k: v for k, v in s.call.items()}, ensure_ascii=False)
                lines.append(f"  Action {i}      : {c[:width]}  [{s.status}]")
                lines.append(f"  Observation {i} : {str(s.observation).replace(chr(10), ' | ')[:width]}")
        lines.append(f"  Réponse finale : {(self.answer or '(aucune : budget d étapes épuisé)')[:width]}")
        return "\n".join(lines)


def run_agent(question: str, tools: dict[str, Tool], step_fn: Callable[[list, list], str],
              max_steps: int = 5, system: str | None = None,
              critic: Callable[[str, "AgentResult"], str | None] | None = None,
              max_reflections: int = 1) -> AgentResult:
    """Boucle d'agent (raisonnement + action, à la ReAct).

    À chaque itération : le modèle (step_fn) lit l'historique et produit soit un ou des appels
    d'outils (exécutés, le résultat étant ajouté comme message « tool » = observation), soit une
    réponse finale sans appel. Les erreurs (outil inconnu, mauvais paramètres, exception) sont
    renvoyées au modèle comme observation, ce qui lui permet de se corriger.

    critic (optionnel) : évaluateur appelé sur chaque réponse finale ; s'il renvoie un retour
    (texte), celui-ci est ajouté à la conversation et la boucle reprend (réflexion, au plus
    max_reflections fois), comme dans Reflexion.
    """
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    schemas = [t.schema() for t in tools.values()]
    res = AgentResult(question, None)
    reflections = 0
    for _ in range(max_steps):
        raw = step_fn(messages, schemas)
        thought, calls = parse_tool_calls(raw)
        if not calls:
            res.answer, res.finished = thought, True
            res.steps.append(Step(thought, None, "fin", ""))
            feedback = critic(question, res) if critic and reflections < max_reflections else None
            if not feedback:
                break
            reflections += 1
            res.finished, res.answer = False, None
            res.steps.append(Step("", None, "réflexion", feedback))
            messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": feedback}]
            continue
        messages.append({"role": "assistant", "content": raw})
        for c in calls:
            status = validate_call(c, tools)
            if status == "ok":
                try:
                    obs = str(tools[c["name"]].fn(**(c.get("arguments") or {})))
                except Exception as e:  # l'erreur devient une observation
                    status, obs = "erreur_execution", f"ERREUR : {type(e).__name__} : {e}"
            else:
                obs = f"ERREUR : appel refusé ({status}). Outils disponibles : {', '.join(tools)}."
            res.steps.append(Step(thought, c, status, obs))
            thought = ""
            messages.append({"role": "tool", "content": obs})
    return res


_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def grounding_critic(question: str, res: "AgentResult") -> str | None:
    """Évaluateur par règles (un « scorer » spécialisé) : chaque nombre de la réponse finale doit
    provenir de la question ou d'une observation d'outil. Sinon, renvoie un retour qui demande
    d'utiliser les outils. Les annonces sans action (« je vais chercher ») sont aussi signalées."""
    answer = res.answer or ""
    a_norm = normalize_text(answer)
    if any(x in a_norm for x in ["je vais chercher", "je vais utiliser", "je vais calculer", "je vais verifier"]):
        return "Tu as annoncé une action sans l'effectuer : appelle maintenant l'outil nécessaire."
    known = " ".join([question] + [str(s.observation) for s in res.steps])
    known = re.sub(r"(?<=\d)[ \u202f\xa0](?=\d{3}\b)", "", known)
    values = [float(n.replace(",", ".")) for n in _NUM_RE.findall(known)]

    def justified(n: str) -> bool:
        x = float(n.replace(",", "."))
        dec = len(n.replace(",", ".").split(".")[1]) if re.search(r"[.,]", n) else 0
        tol = 0.5 * 10 ** (-dec) + 1e-9  # un arrondi d'une valeur connue est accepté
        return any(abs(x - v) <= tol for v in values)

    answer_nums = _NUM_RE.findall(re.sub(r"(?<=\d)[ \u202f\xa0](?=\d{3}\b)", "", answer))
    bad = [n for n in answer_nums if not justified(n)]
    if bad:
        return (f"Vérification : le nombre {bad[0]} de ta réponse ne provient d'aucun résultat d'outil. "
                "Utilise les outils (recherche_corpus, calculatrice, date_du_jour, jours_entre) pour l'obtenir, puis réponds.")
    return None


def make_qwen_step(model_id: str | None = None, max_new_tokens: int = 96,
                   device: str | None = None) -> Callable[[list, list], str]:
    """Fonction d'étape pour run_agent : Qwen avec son gabarit natif d'appel d'outils.
    device="cpu" : modèle chargé sur CPU (voir load_cpu). Compte les tokens lus et produits."""
    import torch

    from . import llm

    model_id = model_id or llm.SMALL

    def step(messages, schemas):
        tok, model = load_cpu(model_id) if device == "cpu" else llm.load(model_id)
        text = tok.apply_chat_template(messages, tools=schemas or None, add_generation_prompt=True, tokenize=False)
        enc = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                 top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        step.tokens_in += enc["input_ids"].shape[1]
        step.tokens_out += out.shape[1] - enc["input_ids"].shape[1]
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=False).replace("<|im_end|>", "").strip()

    step.tokens_in = 0
    step.tokens_out = 0
    return step


#: Tâches d'évaluation de l'agent. expected : variantes acceptées dans la réponse finale ;
#: tools : outils dont on attend l'usage (vide = aucun outil nécessaire).
AGENT_TASKS: list[dict] = [
    {"task": "Combien coûtent 3 Fedoras Fruités chez Minou Chic ?", "expected": ["54"], "tools": ["recherche_corpus", "calculatrice"]},
    {"task": "Combien font 199999 divisé par 292 ? Arrondis à deux décimales.", "expected": ["684.93", "684,93"], "tools": ["calculatrice"]},
    {"task": "Quelle est la date d'aujourd'hui ?", "expected": ["2026-11-25", "25 novembre 2026"], "tools": ["date_du_jour"]},
    {"task": "Combien de jours reste-t-il entre aujourd'hui et le 31 décembre 2026 ?", "expected": ["36"], "tools": ["date_du_jour", "jours_entre"]},
    {"task": "Que signifie le code E-42 du thermostat Thermo-X200 ?", "expected": ["sonde"], "tools": ["recherche_corpus"]},
    {"task": "Quelle est la racine carrée de 1764 ?", "expected": ["42"], "tools": ["calculatrice"]},
    {"task": "Combien coûtent ensemble un Nœud Papillon Gala et un Imperméable Pluie Douce chez Minou Chic ?", "expected": ["47"], "tools": ["recherche_corpus", "calculatrice"]},
    {"task": "Combien vaut 17,5 % de 2 480 euros ?", "expected": ["434"], "tools": ["calculatrice"]},
    {"task": "Bonjour ! Peux-tu te présenter en une phrase ?", "expected": [], "tools": []},
    {"task": "Quel temps fera-t-il demain à Lyon ?", "expected": [], "tools": [], "refuse": True},
]

_REFUS = ["pas acces", "n'ai pas acces", "ne peux pas", "ne dispose pas", "impossible", "je ne sais pas",
          "pas d'outil", "aucun outil", "pas en mesure", "ne suis pas en mesure"]


def classify_run(task: dict, res: AgentResult) -> str:
    """Catégorie d'issue d'une exécution d'agent (typologie de la section *Agent Failure Modes*).

    Ordre de priorité : appel mal formé (JSON, outil inventé, paramètres) > budget épuisé >
    appel inutile > succès (réponse juste) > outil nécessaire non utilisé > réponse incorrecte.
    """
    statuses = [s.status for s in res.steps if s.call is not None]
    for st, lab in [("json_invalide", "JSON invalide"), ("outil_invalide", "outil inventé"),
                    ("parametres_invalides", "paramètres invalides")]:
        if st in statuses:
            return lab
    if not res.finished:
        return "budget d'étapes épuisé"
    used = set(res.tools_used)
    if task.get("refuse"):
        a = normalize_text(res.answer or "")
        return "succès" if any(normalize_text(r) in a for r in _REFUS) else "hallucination (outil manquant)"
    if not task["tools"]:
        return "appel inutile" if used else "succès"
    if answer_ok(res.answer or "", task["expected"]):
        return "succès"
    if not set(task["tools"]) <= used:
        return "outil nécessaire non utilisé"
    return "erreur d'exécution" if "erreur_execution" in statuses else "valeurs ou réponse incorrectes"


def compound_success(p: float, n) -> np.ndarray:
    """Probabilité de réussir n étapes indépendantes de réussite p chacune : p**n."""
    return np.asarray(p, dtype=float) ** np.asarray(n, dtype=float)


def validate_plan(plan: list[str], tools: set[str], max_steps: int = 6) -> tuple[bool, str]:
    """Heuristiques de validation d'un plan (section *Planning overview*) : plan non vide,
    actions toutes dans l'inventaire, longueur raisonnable."""
    if not plan:
        return False, "plan vide"
    bad = [a for a in plan if a not in tools]
    if bad:
        return False, f"action invalide : {bad[0]}"
    if len(plan) > max_steps:
        return False, "plan trop long"
    return True, "ok"


def parse_plan(text: str) -> list[str]:
    """Extrait une liste d'actions d'un texte du type « [a, b, c] » ou d'une liste numérotée."""
    m = re.search(r"\[(.*?)\]", text, re.S)
    body = m.group(1) if m else text
    items = re.split(r"[,\n]", body)
    out = []
    for it in items:
        it = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", it).strip().strip("'\"` ")
        it = re.sub(r"\(.*\)$", "", it).strip()
        if it:
            out.append(it)
    return out


# =====================================================================================
# 8. Mémoire : conversation synthétique et stratégies d'éviction
# =====================================================================================

#: Faits plantés dans la conversation : (énoncé de l'utilisateur, question test, réponses acceptées).
MEMORY_FACTS: list[tuple[str, str, list[str]]] = [
    ("Je m'appelle Léa et je prépare un voyage en Italie cet été.", "Comment est-ce que je m'appelle ?", ["Léa", "Lea"]),
    ("Attention, je suis allergique aux arachides.", "À quoi suis-je allergique ?", ["arachide"]),
    ("Mon budget total pour le voyage est de 1 200 euros.", "Quel est mon budget total ?", ["1 200", "1200"]),
    ("Je partirai de Lyon.", "De quelle ville est-ce que je pars ?", ["Lyon"]),
    ("Je voyage avec mon chien, qui s'appelle Pixel.", "Comment s'appelle mon chien ?", ["Pixel"]),
    ("Je préfère prendre le train plutôt que l'avion.", "Quel moyen de transport est-ce que je préfère ?", ["train"]),
    ("Ma sœur Camille me rejoindra à Rome.", "Qui me rejoindra à Rome ?", ["Camille"]),
    ("Je dois être rentrée chez moi le 14 août.", "Quelle est ma date de retour ?", ["14 août", "14 aout"]),
]

_FILLER_USER = [
    "Peux-tu me conseiller un musée à {v} ?", "Quel temps fait-il en général à {v} en été ?",
    "Y a-t-il un marché intéressant à {v} ?", "Combien de temps faut-il pour visiter {v} ?",
    "Quel plat typique goûter à {v} ?", "Est-ce que {v} est agréable à vélo ?",
    "Quel quartier choisir pour dormir à {v} ?", "Faut-il réserver les visites à {v} à l'avance ?",
]
_FILLER_ASSISTANT = [
    "À {v}, je te recommande de commencer par le centre historique, puis de flâner dans les ruelles en fin de journée.",
    "En été, il fait chaud à {v} : prévois de l'eau, un chapeau et des visites tôt le matin.",
    "Le marché couvert de {v} est très animé le samedi matin ; on y trouve fromages, fruits et spécialités locales.",
    "Compte au moins deux jours pour {v} si tu veux voir les principaux monuments sans te presser.",
    "À {v}, goûte les pâtes fraîches de la maison et termine par une glace artisanale.",
    "Oui, {v} se visite bien à vélo, surtout le long des quais et dans les parcs.",
    "Pour dormir à {v}, un quartier calme mais proche du centre est un bon compromis.",
    "Pour {v}, réserver en ligne quelques jours avant évite de longues files d'attente.",
]
_CITIES = ["Florence", "Venise", "Bologne", "Naples", "Turin", "Vérone", "Sienne", "Gênes", "Pise", "Milan"]


def make_conversation(n_turns: int = 40, facts=None, fact_turns: list[int] | None = None, seed: int = 0) -> list[dict]:
    """Conversation synthétique de n_turns échanges (utilisateur puis assistant).

    Les faits sont énoncés par l'utilisateur aux tours `fact_turns` (par défaut répartis, le
    premier au tour 0) ; les autres tours sont du « remplissage » sur des villes italiennes.
    Chaque message porte un champ "fact" (indice du fait ou None).
    """
    facts = facts or MEMORY_FACTS
    rng = np.random.default_rng(seed)
    if fact_turns is None:
        fact_turns = list(np.linspace(0, n_turns - 4, len(facts)).round().astype(int))
    by_turn = {t: i for i, t in enumerate(fact_turns)}
    msgs = []
    for t in range(n_turns):
        v = _CITIES[rng.integers(len(_CITIES))]
        j = rng.integers(len(_FILLER_USER))
        if t in by_turn:
            i = by_turn[t]
            msgs.append({"role": "user", "content": facts[i][0], "fact": i, "turn": t})
            msgs.append({"role": "assistant", "content": "C'est noté, j'en tiendrai compte pour la suite.", "fact": None, "turn": t})
        else:
            msgs.append({"role": "user", "content": _FILLER_USER[j].format(v=v), "fact": None, "turn": t})
            msgs.append({"role": "assistant", "content": _FILLER_ASSISTANT[j].format(v=v), "fact": None, "turn": t})
    return msgs


def approx_tokens(text: str) -> int:
    """Approximation grossière du nombre de tokens (≈ 1 token par mot ou signe de ponctuation,
    majorée de 30 % pour le français)."""
    return int(math.ceil(1.3 * len(re.findall(r"\w+|[^\w\s]", text))))


def memory_fifo(msgs: list[dict], budget: int, count: Callable[[str], int] = approx_tokens) -> list[dict]:
    """FIFO : on garde les messages les plus récents tant qu'ils tiennent dans le budget."""
    out, used = [], 0
    for m in reversed(msgs):
        c = count(m["content"])
        if used + c > budget:
            break
        out.append(m)
        used += c
    return out[::-1]


def memory_head_tail(msgs: list[dict], budget: int, head: int = 2,
                     count: Callable[[str], int] = approx_tokens) -> list[dict]:
    """Début + fin : on épingle les `head` premiers messages (souvent l'objectif de la
    conversation), puis on remplit le reste du budget avec les plus récents."""
    pinned = msgs[:head]
    used = sum(count(m["content"]) for m in pinned)
    tail = memory_fifo(msgs[head:], max(budget - used, 0), count)
    return pinned + tail


def memory_retrieval(msgs: list[dict], budget: int, query: str, embed_fn: Callable,
                     recent_share: float = 0.5, count: Callable[[str], int] = approx_tokens) -> list[dict]:
    """Mémoire court terme + long terme : une fenêtre récente (recent_share du budget) et, pour
    le reste, les anciens messages les plus proches de la requête (recherche par embeddings)."""
    recent = memory_fifo(msgs, int(budget * recent_share), count)
    old = msgs[:len(msgs) - len(recent)]
    left = budget - sum(count(m["content"]) for m in recent)
    if not old or left <= 0:
        return recent
    E = np.asarray(embed_fn([m["content"] for m in old]))
    q = np.asarray(embed_fn([query]))[0]
    picked, used = [], 0
    for i in np.argsort(-(E @ q)):
        c = count(old[i]["content"])
        if used + c <= left:
            picked.append(int(i))
            used += c
    return [old[i] for i in sorted(picked)] + recent


def memory_summary(msgs: list[dict], budget: int, summarize: Callable[[list[dict]], str],
                   summary_share: float = 0.4, count: Callable[[str], int] = approx_tokens) -> list[dict]:
    """Résumé : les messages qui ne tiennent pas dans la fenêtre récente sont remplacés par un
    résumé (produit par `summarize`, typiquement un LLM)."""
    recent = memory_fifo(msgs, int(budget * (1 - summary_share)), count)
    old = msgs[:len(msgs) - len(recent)]
    if not old:
        return recent
    words = summarize(old).split()
    limit = int(budget * summary_share)
    while words and count(" ".join(words)) > limit:  # le résumé doit tenir dans sa part du budget
        words = words[:-3]
    return [{"role": "system", "content": "Résumé de la conversation : " + " ".join(words), "fact": None, "turn": -1}] + recent


def fact_retention(context: list[dict], facts=None) -> np.ndarray:
    """Pour chaque fait : 1 si une de ses réponses attendues figure dans le contexte retenu."""
    facts = facts or MEMORY_FACTS
    blob = " ".join(m["content"] for m in context)
    return np.array([answer_ok(blob, f[2]) for f in facts], dtype=float)


# =====================================================================================
# 9. RAG tabulaire : petite base SQLite de ventes
# =====================================================================================

PRODUCTS = [(2044, "Fedora Fruité", 18.0), (2045, "Nœud Papillon Gala", 12.0),
            (2046, "Imperméable Pluie Douce", 35.0), (2047, "Collier Lumineux", 9.5),
            (2048, "Pull Torsadé", 24.0)]


def make_sales_db(n_orders: int = 300, seed: int = 0) -> sqlite3.Connection:
    """Base en mémoire avec deux tables : produits(id, nom, prix) et ventes(id, date, produit_id,
    quantite). Les dates couvrent septembre à novembre 2026."""
    rng = np.random.default_rng(seed)
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE produits (id INTEGER PRIMARY KEY, nom TEXT, prix REAL)")
    conn.execute("CREATE TABLE ventes (id INTEGER PRIMARY KEY, date TEXT, produit_id INTEGER, quantite INTEGER)")
    conn.executemany("INSERT INTO produits VALUES (?, ?, ?)", PRODUCTS)
    pop = np.array([0.4, 0.2, 0.1, 0.2, 0.1])
    start = date(2026, 9, 1).toordinal()
    for i in range(1, n_orders + 1):
        d = date.fromordinal(start + int(rng.integers(0, 86))).isoformat()
        p = PRODUCTS[rng.choice(5, p=pop)][0]
        conn.execute("INSERT INTO ventes VALUES (?, ?, ?, ?)", (i, d, p, int(rng.integers(1, 4))))
    conn.commit()
    return conn


SQL_SCHEMA = ("Table produits(id INTEGER, nom TEXT, prix REAL)\n"
              "Table ventes(id INTEGER, date TEXT au format AAAA-MM-JJ, produit_id INTEGER, quantite INTEGER)\n"
              "ventes.produit_id fait référence à produits.id")

SQL_QUESTIONS = [
    {"q": "Quel est le prix du Pull Torsadé ?", "gold": "SELECT prix FROM produits WHERE nom = 'Pull Torsadé'"},
    {"q": "Combien de commandes ont été passées au total ?", "gold": "SELECT COUNT(*) FROM ventes"},
    {"q": "Combien d'unités de Fedora Fruité ont été vendues au total ?",
     "gold": "SELECT SUM(v.quantite) FROM ventes v JOIN produits p ON v.produit_id = p.id WHERE p.nom = 'Fedora Fruité'"},
    {"q": "Quel est le produit le plus cher ?", "gold": "SELECT nom FROM produits ORDER BY prix DESC LIMIT 1"},
    {"q": "Combien de commandes ont été passées en octobre 2026 ?",
     "gold": "SELECT COUNT(*) FROM ventes WHERE date BETWEEN '2026-10-01' AND '2026-10-31'"},
    {"q": "Quel est le chiffre d'affaires total en euros ?",
     "gold": "SELECT SUM(v.quantite * p.prix) FROM ventes v JOIN produits p ON v.produit_id = p.id"},
]


def extract_sql(text: str) -> str:
    """Extrait la première requête SQL d'une sortie de modèle (bloc ```sql ...``` ou SELECT nu)."""
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    s = m.group(1) if m else text
    m = re.search(r"(SELECT\b.*?)(;|$)", s, re.S | re.I)
    return (m.group(1) if m else s).strip()


def run_sql(conn: sqlite3.Connection, sql: str):
    """Exécute une requête en lecture seule (SELECT uniquement) ; renvoie les lignes ou l'erreur."""
    if not re.match(r"^\s*SELECT\b", sql, re.I) or ";" in sql.strip().rstrip(";"):
        return "ERREUR : seules les requêtes SELECT uniques sont autorisées"
    try:
        return conn.execute(sql).fetchall()
    except sqlite3.Error as e:
        return f"ERREUR : {e}"


def same_result(a, b, tol: float = 1e-6) -> bool:
    """Compare deux résultats SQL (listes de tuples), avec tolérance numérique."""
    if isinstance(a, str) or isinstance(b, str) or len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for x, y in zip(ra, rb):
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if abs(x - y) > tol * max(1, abs(y)):
                    return False
            elif x != y:
                return False
    return True


# =====================================================================================
# 10. Recherche vectorielle approchée : outils d'illustration
# =====================================================================================

def knn_graph(X: np.ndarray, m: int = 4) -> list[list[int]]:
    """Graphe des m plus proches voisins (symétrisé), base des index de type NSW / HNSW."""
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    nbrs = np.argsort(D, axis=1)[:, :m]
    adj = [set(r.tolist()) for r in nbrs]
    for i, r in enumerate(nbrs):
        for j in r:
            adj[j].add(i)
    return [sorted(a) for a in adj]


def greedy_graph_search(X: np.ndarray, graph: list[list[int]], q: np.ndarray, entry: int) -> list[int]:
    """Recherche gloutonne dans un graphe de voisinage : on saute vers le voisin le plus proche de
    la requête tant que cela rapproche. Renvoie le chemin parcouru (dernier = résultat)."""
    path = [entry]
    cur = entry
    dcur = float(((X[cur] - q) ** 2).sum())
    while True:
        best, dbest = cur, dcur
        for j in graph[cur]:
            dj = float(((X[j] - q) ** 2).sum())
            if dj < dbest:
                best, dbest = j, dj
        if best == cur:
            return path
        cur, dcur = best, dbest
        path.append(cur)
