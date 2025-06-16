# Chapitre 4 - Évaluer des systèmes d'IA

Le chapitre 3 présentait des méthodes d'évaluation ; le chapitre 4 part des **critères** qui
comptent pour une application (capacité métier, qualité de génération, suivi d'instructions,
coût et latence), montre comment **sélectionner un modèle** (filtres durs, construire ou
acheter, benchmarks publics et leurs pièges) et comment **concevoir son propre pipeline
d'évaluation** (composants, rubriques, tranches, taille d'échantillon, méta-évaluation). Les
notebooks mesurent tout pour de vrai : QCM et contraintes IFEval sur Qwen2.5 0.5B / 1.5B,
cohérence factuelle avec un classifieur NLI DeBERTa-v3, toxicité avec un classifieur RoBERTa,
TTFT / TPOT / débit sur le GPU de la machine.

Les fonctions réutilisables (QCM par logprobs, score NLI et SelfCheck, vérificateurs IFEval,
mesure de latence, coûts API / auto-hébergement et point mort, front de Pareto, agrégation de
benchmarks, n-grammes de contamination, bootstrap, taille d'échantillon, paradoxe de Simpson,
kappa de Cohen) sont dans [`aie/ch04_systemes.py`](../../aie/ch04_systemes.py), testées par
[`tests/test_ch04_systemes.py`](../../tests/test_ch04_systemes.py).

## Notebooks de cours

| Notebook | Sections du livre | Contenu |
|---|---|---|
| [`ch04_cours_1_criteres_generation.ipynb`](ch04_cours_1_criteres_generation.ipynb) | *Evaluation Criteria* (*Evaluation-Driven Development*, *Domain-Specific Capability*, *Generation Capability*, *Factual consistency*, *Safety*) | quatre familles de critères (diagramme) ; QCM par logprobs et test binomial contre le hasard ; sensibilité au format (espace, « Choix : », ordre inversé) et biais de position ; perplexité comme indicateur de fluidité ; cohérence factuelle locale par NLI sur 22 affirmations annotées (diagramme du vérificateur) ; auto-vérification SelfCheckGPT sur questions connues, rares et inexistantes ; SAFE (diagramme) ; classifieur de toxicité, pièges et seuil précision / rappel |
| [`ch04_cours_2_instructions_latence.ipynb`](ch04_cours_2_instructions_latence.ipynb) | *Instruction-Following Capability*, *Instruction-following criteria*, *Roleplaying*, *Cost and Latency* | conformité contre exactitude ; 19 contraintes vérifiables à la IFEval, scores prompt / instruction et strict / souple (diagramme) ; banc maison de 24 tâches sur Qwen 0.5B et 1.5B ; critères oui/non à la INFOBench avec un juge à logprobs ; heuristiques de jeu de rôle ; mesure réelle de TTFT, TPOT, ITL ; TTFT selon la longueur du prompt ; débit et TPOT selon la taille de lot ; percentiles P50 / P90 / P99 ; coût par million de tokens mesuré ; grille de critères avec exigences dures |
| [`ch04_cours_3_selection_modele.ipynb`](ch04_cours_3_selection_modele.ipynb) | *Model Selection*, *Model Selection Workflow*, *Model Build Versus Buy*, *Navigate Public Benchmarks*, *Data contamination with public benchmarks* | workflow itératif en quatre étapes (diagramme) ; filtres durs et front de Pareto ; licences et poids ouverts ; service d'inférence acheté ou construit (diagramme) et sept axes ; point mort API / auto-hébergement et sensibilité à l'utilisation et au coût fixe ; agrégation (moyenne, taux de victoire HELM, z-scores) et rangs ; corrélations du tableau 4-5, dendrogramme et nombre effectif de benchmarks ; contamination par n-grammes et par perplexité ; surestimation $f\beta/p$ ; saturation et pouvoir discriminant |
| [`ch04_cours_4_pipeline_evaluation.ipynb`](ch04_cours_4_pipeline_evaluation.ipynb) | *Design Your Evaluation Pipeline* (*Step 1*, *Step 2*, *Step 3*, *Evaluate your evaluation pipeline*, *Iterate*) | pipeline en trois étapes (diagramme) ; évaluation par composant d'un extracteur d'employeur sur CV bruités ; évaluation par tâche (vingt questions) ; rubrique -1/0/1 et lien avec les métriques métier ; logprobs comme confiance et courbe exactitude / couverture ; coût d'un mélange classifieur / juge / humain ; paradoxe de Simpson et pondération par la production ; bootstrap, largeur en $1/\sqrt n$, tableau 4-7 retrouvé par la formule, évaluation appariée ; kappa entre NLI et juges, stabilité du juge ; empreintes d'expériences |

## Notebooks d'exercices

| Notebook | Exercices |
|---|---|
| [`ch04_exercices_01-05.ipynb`](ch04_exercices_01-05.ipynb) | 4.1 à 4.5 |
| [`ch04_exercices_06-10.ipynb`](ch04_exercices_06-10.ipynb) | 4.6 à 4.10 |

- **Ex. 4.1** - QCM : seuil binomial pour battre le hasard, nombre de questions pour détecter 30 % contre 25 %, évaluation circulaire de Qwen 0.5B et 1.5B.
- **Ex. 4.2** - Vérificateur de cohérence factuelle de bout en bout : erreurs injectées, agrégation moyenne contre minimum, décomposition en faits atomiques.
- **Ex. 4.3** - Étendre IFEval : trois nouveaux vérificateurs (langue, premier mot, sections numérotées), leurs tests unitaires, mesure sur Qwen 0.5B.
- **Ex. 4.4** - Budget de latence : latence totale, longueur maximale sous une exigence P90, TPOT et vitesse de lecture, mesure par taille de lot sur le GPU.
- **Ex. 4.5** - Choisir un modèle pour un hôpital : attributs durs et souples, filtres, utilité pondérée qualité / coût et enveloppe convexe du front de Pareto.
- **Ex. 4.6** - Point mort API contre auto-hébergement : formule analytique, effet du taux d'utilisation, baisse des prix des API.
- **Ex. 4.7** - Agréger un classement à la main (moyenne, taux de victoire, z-scores) et choisir trois benchmarks peu redondants par le nombre effectif.
- **Ex. 4.8** - Détecter la contamination : n-grammes de tokens (copies contre reformulations) et perplexité (courbe ROC).
- **Ex. 4.9** - Construire et désamorcer un paradoxe de Simpson entre deux versions de prompt.
- **Ex. 4.10** - Taille d'échantillon pour comparer deux prompts : formule, gain de l'évaluation appariée, simulation de puissance, règle « ÷3 → ×10 ».
