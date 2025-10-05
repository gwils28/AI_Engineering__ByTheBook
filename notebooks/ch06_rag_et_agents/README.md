# Chapitre 6 - RAG et agents

Un modèle a besoin d'instructions **et** d'informations propres à chaque requête. Ce chapitre
étudie les deux grands schémas de construction du contexte : le **RAG** (aller chercher
l'information dans une mémoire externe, puis générer) et les **agents** (un modèle qui planifie,
appelle des outils, observe et se corrige), ainsi que la **mémoire** qui les sous-tend. Tout est
construit à la main sur un petit corpus français rédigé pour l'occasion (10 documents, 64
passages, 42 questions annotées), avec Qwen2.5-0.5B-Instruct comme générateur et agent.

Les fonctions réutilisables (corpus et questions annotées, TF-IDF et BM25 implémentés à la main,
recherche dense FAISS, fusion RRF, métriques précision@k / rappel@k / MRR / MAP / NDCG,
stratégies de découpage, reranking par cross-encoder et par LLM, génération par lots, outils
sûrs, analyse et validation d'appels de fonctions JSON, boucle ReAct avec réflexion,
classification des modes d'échec, stratégies de mémoire, mini-base SQLite pour le text-to-SQL)
sont dans [`aie/ch06_rag_agents.py`](../../aie/ch06_rag_agents.py), testées par
[`tests/test_ch06_rag_agents.py`](../../tests/test_ch06_rag_agents.py).

## Notebooks de cours

| Notebook | Sections du livre | Contenu |
|---|---|---|
| [`ch06_cours_1_architecture_et_recherche.ipynb`](ch06_cours_1_architecture_et_recherche.ipynb) | *RAG*, *RAG Architecture*, *Retrieval Algorithms* (*Term-based*, *Embedding-based*, *Comparing retrieval algorithms*) | coût du RAG contre « tout dans le contexte » ; diagramme indexation / recherche / génération ; corpus jouet ; tokenisation, index inversé, IDF, TF-IDF ; BM25 à la main, identique à `rank_bm25`, saturation et normalisation, grille (k1, b) ; embeddings MiniLM anglais contre multilingue (ACP, matrice de similarité) ; k-NN exact, idée d'IVF et de HNSW illustrée en 2D, mesures FAISS (rappel, débit, construction, mémoire) ; vitesse et qualité par type de question |
| [`ch06_cours_2_evaluation_et_optimisation.ipynb`](ch06_cours_2_evaluation_et_optimisation.ipynb) | *Comparing retrieval algorithms*, *Combining retrieval algorithms*, *Retrieval Optimization* (*Chunking*, *Reranking*, *Query rewriting*, *Contextual retrieval*), *Evaluating Retrieval Solutions* | précision@k, rappel@k, MRR, MAP, NDCG ; rappel@k selon k ; hybride : RRF (sensibilité à k), fusion linéaire, cascade ; 13 stratégies de découpage mesurées (qualité contre tokens) ; reranking par cross-encoder `bge-reranker-v2-m3` et par Qwen ; réécriture de requête (heuristique, Qwen 0,5B et 1,5B) ; recherche contextuelle (titre, contexte généré) ; grille de choix d'une solution |
| [`ch06_cours_3_rag_de_bout_en_bout.ipynb`](ch06_cours_3_rag_de_bout_en_bout.ipynb) | *RAG Architecture*, *RAG Beyond Texts* (*Multimodal RAG*, *RAG with tabular data*) | pipeline complet (hybride, cross-encoder, prompt, Qwen) ; exactitude et abstention sans contexte, avec RAG, oracle, contexte hors sujet ; matrice « retriever / générateur » pour localiser les erreurs ; effet de k et de la position du bon passage ; RAG tabulaire : text-to-SQL sur SQLite (0,5B contre 1,5B), garde-fous ; RAG multimodal (CLIP) |
| [`ch06_cours_4_agents_et_memoire.ipynb`](ch06_cours_4_agents_et_memoire.ipynb) | *Agents* (*Agent Overview*, *Tools*, *Planning*, *Agent Failure Modes and Evaluation*), *Memory* | anatomie d'un agent, erreurs composées $p^n$ ; inventaire d'outils sûrs (calculatrice sans `eval`) ; appel de fonctions JSON natif de Qwen ; génération et validation de plans, flux de contrôle ; boucle ReAct ; réflexion par un vérificateur d'ancrage ; évaluation par modes d'échec (4 configurations, dont Qwen 1,5B), efficacité, ablation d'outil, distribution des appels ; mémoire : trois mécanismes, budget, éviction FIFO / début + fin / résumé / récupération mesurée |

## Notebooks d'exercices

| Notebook | Exercices |
|---|---|
| [`ch06_exercices_01-06.ipynb`](ch06_exercices_01-06.ipynb) | 6.1 à 6.6 |
| [`ch06_exercices_07-12.ipynb`](ch06_exercices_07-12.ipynb) | 6.7 à 6.12 |

- **Ex. 6.1** - Long contexte ou RAG : coût quotidien, effet du cache de préfixe, temps de pré-remplissage.
- **Ex. 6.2** - TF-IDF et BM25 calculés à la main sur un mini-corpus ; le bourrage de mots-clés.
- **Ex. 6.3** - Dimensionner un index de 50 millions de vecteurs : mémoire Flat / PQ / HNSW, calcul IVF, vérification FAISS.
- **Ex. 6.4** - Précision, rappel, RR, AP et NDCG calculés à la main pour trois requêtes.
- **Ex. 6.5** - RRF contre fusion linéaire : invariance aux transformations des scores, validation croisée de α, limite k infini.
- **Ex. 6.6** - Taille optimale des morceaux à budget de contexte fixé (mini-lab).
- **Ex. 6.7** - Reranking sous budget de latence : coût d'un cross-encoder, MRR et latence selon le nombre de candidats.
- **Ex. 6.8** - Réécriture de requête quand la conversation change de sujet : heuristiques contre Qwen 1,5B.
- **Ex. 6.9** - Erreurs composées et réflexion : calcul, nombre maximal d'étapes, simulation de Monte-Carlo, coût.
- **Ex. 6.10** - Classer des appels de fonctions (outil invalide, paramètres invalides, valeurs incorrectes) ; effet des descriptions d'outils.
- **Ex. 6.11** - Planification : ordre des filtres, validateur de dépendances, chemin critique en parallèle.
- **Ex. 6.12** - Mémoire : budget de contexte poste par poste, survie d'un fait selon sa position et la stratégie d'éviction.
