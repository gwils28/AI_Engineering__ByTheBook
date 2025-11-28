# Chapitre 7 - Finetuning

Le chapitre 7 traite de l'adaptation d'un modèle par la **modification de ses poids** :
quand le finetuning se justifie (et quand préférer le prompt ou le RAG), pourquoi il est si
gourmand en **mémoire** (poids, activations, gradients, états de l'optimiseur), comment les
**représentations numériques** et la **quantification** réduisent cette empreinte, comment
le **finetuning efficace en paramètres** (PEFT, LoRA, QLoRA) le rend accessible sur du
matériel modeste, comment **fusionner** plusieurs modèles, et quelles **tactiques**
d'entraînement adopter. Les notebooks remplacent les ordres de grandeur du livre par des
calculs, des mesures de mémoire GPU et de vrais finetunings LoRA et QLoRA de Qwen2.5-0.5B
sur une tâche de format en français, construite pour ce dépôt.

Les fonctions réutilisables (jeu de données « tickets clients », calculateur de mémoire et
mesure d'un pas d'entraînement, arrondi dans un format flottant quelconque, quantification
absmax / zéro-point / par groupe / NF4 et quantification simulée d'un modèle, couche LoRA
écrite à la main et comptage de paramètres, boucle de SFT avec poids de la perte du prompt,
fusion par moyenne, arithmétique des tâches, SLERP, TIES et DARE, réseaux `digits` pour les
expériences de fusion) sont dans [`aie/ch07_finetuning.py`](../../aie/ch07_finetuning.py),
testées par [`tests/test_ch07_finetuning.py`](../../tests/test_ch07_finetuning.py).

## Notebooks de cours

| Notebook | Sections du livre | Contenu |
|---|---|---|
| [`ch07_cours_1_vue_ensemble_et_decision.ipynb`](ch07_cours_1_vue_ensemble_et_decision.ipynb) | *Finetuning Overview* ; *When to Finetune* (*Reasons to Finetune*, *Reasons Not to Finetune*, *Finetuning and RAG*) | Apprentissage par transfert : courbe d'apprentissage de zéro contre embeddings pré-entraînés ; diagramme des chemins d'adaptation (pré-entraînement continu, SFT, préférence, contexte long, infilling) ; masque de perte d'un exemple SFT visualisé token par token ; échec de format de Qwen2.5-0.5B ; BloombergGPT et coût de calcul ; point mort exemples dans le prompt contre finetuning, avec et sans cache de prompt ; échecs d'information contre de comportement (démonstration), données d'Ovadia et al. ; arbre de décision prompt / RAG / finetuning ; flux de travail d'adaptation |
| [`ch07_cours_2_memoire_et_quantification.ipynb`](ch07_cours_2_memoire_et_quantification.ipynb) | *Memory Bottlenecks* : *Backpropagation and Trainable Parameters*, *Memory Math*, *Numerical Representations*, *Quantization* (*Inference quantization*, *Training quantization*) | Réseau à trois paramètres : passes avant et arrière à la main, un pas d'Adam ; calculateur de mémoire (inférence, finetuning complet, LoRA) pour 7 tailles de modèles ; activations selon Korthikanti et al. ; mesures GPU d'un pas d'entraînement de GPT-2 (Adam, SGD, partiel) et de Qwen2.5-0.5B (LoRA), poids des logits ; gradient checkpointing mesuré ; disposition des bits (FP32 à INT4), plages et précisions, tableau 7-3 recalculé, espacement des flottants, Qwen en FP32 / BF16 / FP16 ; accumulation et mises à jour en 16 bits (précision mixte) ; absmax et zéro-point illustrés ; distribution des poids de Qwen et niveaux NF4 ; SQNR selon la granularité ; perplexité de Qwen quantifié (INT8 à INT3) ; vraie quantification bitsandbytes (INT8, NF4) ; PTQ contre QAT avec estimateur straight-through |
| [`ch07_cours_3_peft_lora_qlora.ipynb`](ch07_cours_3_peft_lora_qlora.ipynb) | *Parameter-Efficient Finetuning* : *PEFT techniques*, *LoRA* (*Why does LoRA work?*, *LoRA configurations*, *Serving LoRA adapters*, *Quantized LoRA*) | Paramètres entraînables par méthode sur Qwen2.5-0.5B ; diagramme adaptateurs / LoRA / soft prompts ; LoRA écrite à la main (initialisation, fusion, effet du rang et de alpha sur une mise à jour de rang 4) ; spectre de la différence Qwen2.5-0.5B Instruct moins base ; tableau 7-5 et budget de 18,9 M sur GPT-3 ; carte paramètres selon matrices et rang ; vrai finetuning LoRA avec `peft` sur la tâche des tickets (avant / après, courbes de perte, effets de bord sur d'autres questions) ; stockage multi-LoRA et latence fusionné / séparé ; QLoRA réel (NF4 + LoRA) comparé à LoRA ; Elo de Guanaco |
| [`ch07_cours_4_fusion_et_tactiques.ipynb`](ch07_cours_4_fusion_et_tactiques.ipynb) | *Model Merging and Multi-Task Finetuning* (*Summing*, *Layer stacking*, *Concatenation*) ; *Finetuning Tactics* | Ensemble contre fusion (diagramme) ; oubli catastrophique en finetuning séquentiel ; vecteurs de tâche et arithmétique des tâches (addition, négation) sur des MLP `digits` ; SLERP (géométrie, normes, interpolation entre finetunings) ; redondance des vecteurs de tâche, étapes de TIES, comparaison moyenne / arithmétique / TIES / DARE ; SOLAR et upcycling en MoE (diagramme), concaténation d'adaptateurs ; chemins de progression et de distillation ; planifications et balayage du taux d'apprentissage (LoRA réel) ; bruit du gradient en 1/B et accumulation ; sur-apprentissage au fil des époques ; poids de la perte du prompt (0, 10 %, 100 %) |

## Notebooks d'exercices

| Notebook | Exercices |
|---|---|
| [`ch07_exercices_01-06.ipynb`](ch07_exercices_01-06.ipynb) | 7.1 à 7.6 |
| [`ch07_exercices_07-12.ipynb`](ch07_exercices_07-12.ipynb) | 7.7 à 7.12 |

- **Ex. 7.1** - Dimensionner le matériel : mémoire d'inférence, de finetuning complet, LoRA et QLoRA pour Llama 2-13B et 70B, nombre de GPU.
- **Ex. 7.2** - Prompt, RAG ou finetuning : diagnostic de quatre scénarios et point mort avec cache de prompt.
- **Ex. 7.3** - Mesurer la mémoire d'un pas d'entraînement de GPT-2 selon l'optimiseur, le format et les paramètres entraînables.
- **Ex. 7.4** - Les pièges du FP16 : softmax qui déborde, gradients qui s'annulent, loss scaling.
- **Ex. 7.5** - Quantifier à la main en absmax et zéro-point, puis sur des activations positives.
- **Ex. 7.6** - Choisir une quantification sous un budget de 4,6 bits par poids (SQNR et perplexité sur Qwen2.5-0.5B).
- **Ex. 7.7** - Compter les paramètres LoRA (GPT-3, Llama 2), repérer une incohérence du tableau 7-6, stockage multi-LoRA.
- **Ex. 7.8** - LoRA à la main : gradients, initialisation, rang de la mise à jour, coût de la version non fusionnée.
- **Ex. 7.9** - Quel rang pour la tâche des tickets ? LoRA r = 1, 4, 32 sur Qwen2.5-0.5B.
- **Ex. 7.10** - Soupes de modèles uniformes et gloutonnes sur des finetunings d'une même tâche.
- **Ex. 7.11** - SLERP : démonstration de la conservation de la norme, fusion en cascade de trois vecteurs de tâche.
- **Ex. 7.12** - Accumulation de gradient et normalisation par token, part du prompt dans la perte.
