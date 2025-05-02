# Chapitre 3 - Méthodologie d'évaluation

Le chapitre 3 explique pourquoi évaluer un modèle de fondation est difficile (réponses ouvertes, modèles opaques, benchmarks vite saturés) et passe en revue les méthodes d'évaluation automatique : les métriques de modélisation du langage (entropie croisée, perplexité, BPC, BPB), l'évaluation exacte (correction fonctionnelle, similarité exacte, lexicale et sémantique), l'IA comme juge et le classement de modèles par comparaisons. Les notebooks démontrent les liens entre métriques, mesurent tout sur de vrais petits modèles (GPT-2, Qwen2.5 0,5 B et 1,5 B, all-MiniLM) et simulent les algorithmes de classement.

Les fonctions réutilisables (métriques d'entropie et de perplexité, estimateur de pass@k, exécution isolée de code et mini-banc de problèmes, BLEU, ROUGE, distance d'édition, juge IA par lecture de probabilités, coûts, Elo, Bradley-Terry, simulations de matchs) sont dans [`aie/ch03_evaluation.py`](../../aie/ch03_evaluation.py), testées par [`tests/test_ch03_evaluation.py`](../../tests/test_ch03_evaluation.py).

## Notebooks de cours

| Notebook | Sections du livre | Contenu |
|---|---|---|
| [`ch03_cours_1_metriques_langage.ipynb`](ch03_cours_1_metriques_langage.ipynb) | *Challenges of Evaluating Foundation Models* ; *Understanding Language Modeling Metrics* (*Entropy*, *Cross Entropy*, *Bits-per-Character and Bits-per-Byte*, *Perplexity*, *Perplexity Interpretation and Use Cases*) | réponses ouvertes mesurées (20 traductions d'une même phrase), saturation des benchmarks, carte des méthodes du chapitre ; entropie et langue du carré ; démonstration de H(P,Q) = H(P) + KL ; conversions CE / BPC / BPB / PPL démontrées, lien avec la compression ; surprise token par token de GPT-2 ; comparaison GPT-2 / Qwen base / Qwen Instruct (seul le BPB est comparable) ; effet de la structure, du contexte (de 1 à 256 tokens, texte répété), du vocabulaire et du post-entraînement ; textes mémorisés, inédits et mélangés ; détection de contamination (perplexité, Min-K% Prob) |
| [`ch03_cours_2_evaluation_exacte.ipynb`](ch03_cours_2_evaluation_exacte.ipynb) | *Exact Evaluation* : *Functional Correctness*, *Similarity Measurements Against Reference Data*, *Exact match*, *Lexical similarity* | harnais d'évaluation de code (diagramme) ; exécution isolée sur tests unitaires ; pass@k : estimateur non biaisé démontré et simulé (contre plug-in et k essais) ; Qwen2.5-0.5B sur 10 problèmes, 20 échantillons à deux températures, pass@k, types d'erreur, effet du nettoyage du code par le harnais ; correspondance exacte, normalisée et par inclusion (faux positifs et négatifs) ; Levenshtein (table de programmation dynamique), n-grammes, BLEU et ROUGE implémentés et décomposés ; limites : références incomplètes, BLEU contre correction fonctionnelle |
| [`ch03_cours_3_embeddings_et_juge_ia.ipynb`](ch03_cours_3_embeddings_et_juge_ia.ipynb) | *Semantic similarity* ; *Introduction to Embedding* ; *AI as a Judge* (*Why*, *How to Use*, *Limitations*, *Biases*, *What Models Can Act as Judges?*) | forme contre sens (lexical vs cosinus) ; embeddings : similarités, ACP 2D de 24 phrases, distributions intra / inter thèmes, recherche sémantique, CLIP (diagramme) ; juge = modèle + prompt + échantillonnage ; juge par lecture de la distribution des notes 1-5 ; Qwen 0,5 B contre 1,5 B face à des notes humaines ; biais de position et de verbosité mesurés ; incohérence et ambiguïté des critères ; coût, échantillonnage et latence d'un juge |
| [`ch03_cours_4_evaluation_comparative.ipynb`](ch03_cours_4_evaluation_comparative.ipynb) | *Ranking Models with Comparative Evaluation* ; *Challenges of Comparative Evaluation* ; *The Future of Comparative Evaluation* | architecture d'une arène ; Bradley-Terry et Elo, Elo vu comme gradient stochastique ; trajectoires Elo, facteur K, sensibilité à l'ordre des matchs ; Bradley-Terry par algorithme MM et intervalles bootstrap ; table 3-6 du livre : ajustement, cycles, graphe des duels ; intransitivité (pierre-feuille-ciseaux) ; croissance quadratique, matchs nécessaires, appariement adaptatif ; du taux de victoire à la performance absolue |

## Notebooks d'exercices

| Notebook | Exercices |
|---|---|
| [`ch03_exercices_01-06.ipynb`](ch03_exercices_01-06.ipynb) | 3.1 à 3.6 |
| [`ch03_exercices_07-12.ipynb`](ch03_exercices_07-12.ipynb) | 3.7 à 3.12 |

- **Ex. 3.1** - Bornes de l'entropie (démonstration par l'inégalité de Gibbs), langue du carré non uniforme, additivité.
- **Ex. 3.2** - Conversions CE / BPC / BPB / perplexité sur le tableau 3-1 et sur deux corpus, vérification sur GPT-2 et Qwen.
- **Ex. 3.3** - Minimiser l'entropie croisée = maximum de vraisemblance ; gradient « prédiction moins cible » ; KL directe contre KL inverse.
- **Ex. 3.4** - Contamination : détection sous modification partielle des textes, test de complétion de préfixe (mini-lab Qwen base).
- **Ex. 3.5** - pass@k à la main, variance de l'estimateur, significativité d'un écart sur 164 problèmes.
- **Ex. 3.6** - Concevoir et évaluer un vérificateur de réponses courtes (précision, rappel), application à Qwen.
- **Ex. 3.7** - BLEU à la main : précisions écrêtées, pénalité de brièveté, moyenne géométrique.
- **Ex. 3.8** - Métriques lexicales et sémantiques face à des jugements humains (corrélation de Spearman, pires erreurs).
- **Ex. 3.9** - Mesurer et corriger le biais de position d'un juge Qwen 1,5 B (score invariant par permutation).
- **Ex. 3.10** - Budget d'un juge IA en production : options de juges, échantillonnage, tailles d'échantillon.
- **Ex. 3.11** - Elo à la main, dépendance à l'ordre, conservation de la somme, Elo = gradient stochastique.
- **Ex. 3.12** - Table 3-6 (victoires, Bradley-Terry, Copeland), chiffres de LMSYS, appariement adaptatif pour trouver le meilleur modèle.
