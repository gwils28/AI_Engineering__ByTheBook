# Chapitre 2 - Comprendre les modèles de fondation

Pourquoi deux modèles de fondation se comportent-ils différemment ? Ce chapitre remonte aux
décisions de conception qui les distinguent : les données d'entraînement (langues, domaines),
l'architecture transformer et sa taille (paramètres, mémoire, calcul, lois d'échelle), le
post-entraînement (SFT, préférences, RLHF) et enfin l'échantillonnage, qui explique à la fois
la créativité, les incohérences et les hallucinations des modèles. Les notebooks
réimplémentent l'attention, le top-p, le modèle de récompense de Bradley-Terry ou le décodage
contraint, et les confrontent à de vrais modèles (GPT-2, Qwen2.5-0.5B).

Les fonctions réutilisables (corpus parallèle multilingue, attention et RoPE en numpy,
comptage des paramètres à partir d'une configuration, cache KV, FLOPs et loi de Chinchilla,
Bradley-Terry, filtres top-k / top-p / min-p, vote majoritaire, décodage contraint par
expression régulière, réparation de JSON) sont dans
[`aie/ch02_modeles.py`](../../aie/ch02_modeles.py), testées par
[`tests/test_ch02_modeles.py`](../../tests/test_ch02_modeles.py).

## Notebooks de cours

| Notebook | Sections du livre | Contenu |
|---|---|---|
| [`ch02_cours_1_donnees_et_transformer.ipynb`](ch02_cours_1_donnees_et_transformer.ipynb) | *Training Data* (*Multilingual Models*, *Domain-Specific Models*), *Model Architecture* (*Transformer architecture*, *Attention mechanism*, *Transformer block*) | langues de Common Crawl et sous-représentation ; tokens par langue pour 4 tokenizers, coût et latence ; bits par caractère de GPT-2 et Qwen2.5 selon le domaine ; seq2seq contre transformer (schéma, profondeur séquentielle, travail) ; attention pas à pas, rôle de $1/\sqrt{d}$, masque causal ; attention multi-têtes reproduite à $10^{-6}$ près sur GPT-2 et Qwen2.5 (RoPE, GQA) ; cartes d'attention réelles, têtes « token précédent » et têtes d'induction ; schéma d'un transformer décodeur |
| [`ch02_cours_2_taille_et_echelle.ipynb`](ch02_cours_2_taille_et_echelle.ipynb) | *Model Architecture* (tableau des dimensions, *Other model architectures*), *Model Size*, *Scaling law*, *Scaling extrapolation*, *Scaling bottlenecks* | comptage exact des paramètres à partir de la configuration (vérifié sur les fichiers de poids) ; répartition plongements / attention / MLP ; mémoire des poids et du cache KV ; prefill contre decode (schéma et mesures) ; SSM sous forme récurrente et convolutive, Mamba, Jamba ; mélange d'experts (Mixtral) ; $C \approx 6ND$, durée et coût de GPT-3 ; courbes iso-FLOPs de Chinchilla, modèles sur-entraînés et optimum tenant compte de l'inférence ; extrapolation et capacités « émergentes » ; stock de données et énergie |
| [`ch02_cours_3_post_entrainement.ipynb`](ch02_cours_3_post_entrainement.ipynb) | *Post-Training* (*Supervised Finetuning*, *Preference Finetuning*, *Reward model*, *Finetuning using the reward model*) | modèle de base contre modèle Instruct ; cycle de vie pré-entraînement, SFT, modèle de récompense, RLHF (schéma) ; gabarit de conversation et masque de perte du SFT ; coût des démonstrations et des comparaisons ; notes ponctuelles contre comparaisons (simulation d'annotateurs) ; modèle de récompense de Bradley-Terry et ses biais ; politique optimale du RLHF régularisé par KL et vérification que DPO la retrouve ; best-of-N et sur-optimisation (loi de Goodhart) |
| [`ch02_cours_4_echantillonnage.ipynb`](ch02_cours_4_echantillonnage.ipynb) | *Sampling* (*Sampling Fundamentals*, *Sampling Strategies*, *Test Time Compute*, *Structured Outputs*, *The Probabilistic Nature of AI*) | vraie distribution du prochain token ; température et entropie ; logprobs et sous-dépassement ; top-k, top-p, min-p et taille du noyau ; troncature d'un JSON ; test time compute sur 24 problèmes (vote majoritaire, logprob, oracle) ; extraction JSON par prompt, post-traitement et décodage contraint ; classification contrainte ; incohérence face aux reformulations ; hallucinations en boule de neige, entités inventées, détection par cohérence entre échantillons |

## Notebooks d'exercices

| Notebook | Exercices |
|---|---|
| [`ch02_exercices_01-06.ipynb`](ch02_exercices_01-06.ipynb) | 2.1 à 2.6 |
| [`ch02_exercices_07-12.ipynb`](ch02_exercices_07-12.ipynb) | 2.7 à 2.12 |

- **Ex. 2.1** - Coût mensuel d'un assistant multilingue selon la langue et le tokenizer (mesure + calcul).
- **Ex. 2.2** - Propriétés de l'attention causale : indépendance au futur, équivariance par permutation, mémoire de la matrice de scores.
- **Ex. 2.3** - Paramètres, mémoire des poids, cache KV et capacité de service de Mistral-7B à partir de sa configuration (avec et sans GQA).
- **Ex. 2.4** - FLOPs, durée, coût et énergie du pré-entraînement de Llama 3-8B ; volume d'inférence équivalent.
- **Ex. 2.5** - Dimensionner un modèle pour un budget de 500 000 \$ avec la loi de Chinchilla ; coût d'un modèle plus petit.
- **Ex. 2.6** - Intensité arithmétique du decode, borne de vitesse liée à la mémoire et mesure du débit selon la taille de batch.
- **Ex. 2.7** - Classer des modèles par duels avec Bradley-Terry et scores Elo ; le piège du taux de victoire.
- **Ex. 2.8** - Démontrer $dH/dT = \mathrm{Var}_p(x)/T^3$ et le vérifier sur de vrais logits ; diversité selon la température.
- **Ex. 2.9** - Implémenter et tester le top-p ; taille du noyau contre top-k sur un vrai texte.
- **Ex. 2.10** - Vote majoritaire : théorème du jury de Condorcet, erreurs dispersées, essai sur Qwen2.5-0.5B.
- **Ex. 2.11** - Décodage contraint par arbre de préfixes pour router des messages ; extraction de dates au format AAAA-MM-JJ.
- **Ex. 2.12** - Logprob d'une séquence, longueur de sous-dépassement en float32 et float64, biais de longueur de la logprob totale.
