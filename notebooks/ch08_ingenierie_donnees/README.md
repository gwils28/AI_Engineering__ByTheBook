# Chapitre 8 - Ingénierie des données

Ce chapitre est consacré à l'ingénierie des données (« dataset engineering ») : constituer le jeu de données qui permet
d'affiner le meilleur modèle possible dans un budget donné. On y voit comment évaluer la **qualité**, la **couverture** et
la **quantité** des données, comment annoter et mesurer l'accord entre annotateurs, comment **augmenter et synthétiser**
des données (règles, perturbations, simulation, LLM) en les vérifiant, pourquoi l'entraînement récursif sur données
générées provoque un **effondrement de modèle**, et comment **traiter** un jeu brut (inspection, déduplication par
hachage et MinHash/LSH, nettoyage, mise au format avec le chat template).

Les fonctions réutilisables (données synthétiques à sous-populations, bruit d'étiquetage, courbes d'apprentissage et loi
de puissance, score de Vendi, kappas de Cohen et de Fleiss, gabarits, substitution de genre, fautes de frappe, problèmes
vérifiables, filtres Self-Instruct, simulations d'effondrement, perte de distillation, jeu d'instructions « sale »,
détection de langue, nettoyage HTML et PII, MinHash, LSH, construction d'exemples SFT avec masque de perte) sont dans
[`aie/ch08_donnees.py`](../../aie/ch08_donnees.py), testées par
[`tests/test_ch08_donnees.py`](../../tests/test_ch08_donnees.py).

## Notebooks de cours

| Notebook | Sections du livre | Contenu |
|---|---|---|
| [`ch08_cours_1_curation.ipynb`](ch08_cours_1_curation.ipynb) | Introduction, *A Data-Centric View of AI*, *Data Curation* (*Data Quality*, *Data Coverage*, *Data Quantity*, *Data Acquisition and Annotation*) | IA centrée modèle contre centrée données (diagramme) ; formats de données par phase, surcoût de la chaîne de pensée en tokens ; six critères de qualité, expérience bruit symétrique contre bruit systématique et démonstration de l'invariance du classifieur de Bayes ; couverture à budget égal, mélange de domaines de Llama 3, diversité mesurée par distinct-n et score de Vendi ; courbe d'apprentissage, loi de puissance $a n^{-b} + c$, inversion pour le budget ; volant d'inertie des données (diagramme), entonnoir de constitution d'un jeu, consignes d'annotation, paradoxe de l'accord brut, kappas de Cohen et de Fleiss, diagnostic des annotateurs biaisés |
| [`ch08_cours_2_augmentation_synthese.ipynb`](ch08_cours_2_augmentation_synthese.ipynb) | *Data Augmentation and Synthesis* (*Why Data Synthesis*, *Traditional Data Synthesis Techniques*, *AI-Powered Data Synthesis*, *Instruction data synthesis*, *Data verification*, *Limitations to AI-generated data*, *Model Distillation*) | Augmentation contre synthèse et cinq motivations (diagramme) ; transactions synthétiques par gabarit, diversité bornée par la combinatoire ; contrefactuels de genre, fautes de frappe et robustesse d'un classifieur d'intentions ; simulation « générer puis garder ce qui réussit » ; synthèse d'instructions avec Qwen2.5-1.5B (pipeline Alpaca/UltraChat en diagramme), température contre diversité, paraphrase, instruction inverse ; pipeline de code de Llama 3 (diagramme), vérification par réponse connue avec Qwen2.5-0.5B, heuristiques et contrôle thématique, rétro-traduction et ses angles morts ; effondrement de modèle démontré (gaussienne, loi de Zipf, remède par données réelles) ; distillation : distributions professeur/élève de Qwen, expérience d'étiquettes dures et douces |
| [`ch08_cours_3_traitement.ipynb`](ch08_cours_3_traitement.ipynb) | *Data Processing* (*Inspect Data*, *Deduplicate Data*, *Clean and Filter Data*, *Format Data*) | Inspection d'un jeu d'instructions sale : longueurs en tokens par source, langues, formes d'instructions, scores par annotateur, conflits ; doublons et biais, hachage exact brut et normalisé, contamination du test ; shingles et Jaccard, MinHash (démonstration de l'estimateur, erreur en $1/\sqrt{k}$), LSH (diagramme, courbes en S), application et matrice de similarité, temps $O(n^2)$ contre $O(n)$ ; retrait du HTML, masquage des PII, entonnoir de filtrage ; few-shot vers exemples d'affinage, chat template et masque de perte visualisé, pièges de format ; pipeline complet (diagramme) et ordre optimal des étapes |

## Notebooks d'exercices

| Notebook | Exercices |
|---|---|
| [`ch08_exercices_01-06.ipynb`](ch08_exercices_01-06.ipynb) | 8.1 à 8.6 |
| [`ch08_exercices_07-11.ipynb`](ch08_exercices_07-11.ipynb) | 8.7 à 8.11 |

- **Ex. 8.1** - Combien d'exemples acheter ? Ajuster une loi de puissance, extrapoler sous contrainte de budget, quantifier l'incertitude.
- **Ex. 8.2** - Kappa de Cohen à la main et paradoxe de la prévalence : formule fermée de $\kappa(\pi)$ vérifiée par simulation.
- **Ex. 8.3** - Kappa de Fleiss, détection d'un annotateur biaisé et d'un spammeur sans vérité terrain, exactitude du vote majoritaire.
- **Ex. 8.4** - Un peu de données propres ou beaucoup de données bruitées ? Facteur d'équivalence $1/\Delta^2$, mesure et seuil de rentabilité.
- **Ex. 8.5** - Trouver le bon mélange de données : balayage du simplexe, comparaison au trafic réel, prédiction à petite échelle.
- **Ex. 8.6** - Augmentation contrefactuelle de genre : neutralisation du biais, erreurs grammaticales, difficulté du français.
- **Ex. 8.7** - Synthèse vérifiable de raisonnements avec Qwen2.5-0.5B : filtre par la réponse, auto-cohérence, coût par exemple gardé.
- **Ex. 8.8** - Juge IA pour données de préférence : biais de position du 0,5B et du 1,5B, filtre de cohérence à la NVIDIA.
- **Ex. 8.9** - Effondrement de modèle : demi-vie de la variance, point fixe avec données réelles, disparition des événements rares.
- **Ex. 8.10** - Dimensionner MinHash + LSH pour un milliard de documents : mémoire, contraintes infaisables avec 128 hachages, compromis, faux candidats, simulation.
- **Ex. 8.11** - Mise au format SFT : masque de perte multi-tours, poids du gabarit, gabarit d'une autre famille, contrôle des espaces.
