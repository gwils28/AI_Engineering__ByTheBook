# Chapitre 9 - Optimisation de l'inférence

Un modèle n'est utile que s'il répond assez vite et pour un coût acceptable. Ce chapitre
apprend à diagnostiquer les goulots de l'inférence (calcul ou bande passante mémoire, modèle
roofline), à les mesurer (TTFT, TPOT, débit, goodput, MFU, MBU) et à les lever aux trois
niveaux du livre : le modèle (quantification, distillation, élagage, décodage spéculatif,
attention efficace), le matériel (accélérateurs, noyaux, compilateurs) et le service
(batching, dissociation prefill / decode, cache de prompt, parallélisme). Toutes les mesures
sont faites sur un vrai GPU (RTX 5070 Laptop, 8 Go) avec Qwen2.5-0.5B et Qwen2.5-1.5B ; le
GPU étant partagé, les commentaires portent sur des ordres de grandeur et des rapports.

Les fonctions réutilisables (fiches d'accélérateurs et roofline, métriques et coûts, mesures
GPU robustes avec relance après OOM, quantification simulée et élagage, décodage glouton avec
ou sans cache, décodage spéculatif, inférence avec référence et décodage de Jacobi écrits à
la main, taille du cache KV et allocation paginée, simulation à événements discrets du
batching, cache de prompt, parallélisme de tenseur et de pipeline, rangement des répliques)
sont dans [`aie/ch09_inference.py`](../../aie/ch09_inference.py), testées par
[`tests/test_ch09_inference.py`](../../tests/test_ch09_inference.py).

## Notebooks de cours

| Notebook | Sections du livre | Contenu |
|---|---|---|
| [`ch09_cours_1_goulots_et_metriques.ipynb`](ch09_cours_1_goulots_et_metriques.ipynb) | *Understanding Inference Optimization* (*Inference Overview*, *Computational bottlenecks*, *Online and batch inference APIs*, *Inference Performance Metrics*, *AI Accelerators*) | schéma d'un service d'inférence ; API en ligne et batch ; intensité arithmétique et roofline de la RTX 5070 Laptop, de l'A100 et du H100 ; produit matriciel mesuré sur le roofline ; prefill et decode placés sur les deux rooflines, puis mesurés ; TTFT et TPOT mesurés selon la longueur du prompt ; percentiles ; *time to publish* d'un agent ; coût par million de tokens ; compromis latence / débit mesuré selon la taille du lot ; goodput sous SLO ; MBU et MFU calculées à partir de mesures (0,5B et 1,5B, decode et prefill, selon le lot) ; CPU contre GPU ; FLOP/s par précision (fiche H100, mesure RTX) ; hiérarchie mémoire ; énergie par token |
| [`ch09_cours_2_modele_et_decodage.ipynb`](ch09_cours_2_modele_et_decodage.ipynb) | *Inference Optimization*, *Model Optimization* (*Model compression*, *Overcoming the autoregressive decoding bottleneck* : *Speculative decoding*, *Inference with reference*, *Parallel decoding*) | trois niveaux d'optimisation ; quantification absmax par tenseur, par ligne, par groupe ; activations aberrantes (*massive activations*) contre poids ; perplexité selon les bits (simulation int8 à int3) ; bf16, LLM.int8 et nf4 avec bitsandbytes (mémoire, vitesse, perplexité mesurées) ; distillation (schéma, accord top-1 et KL entre 0,5B et 1,5B, température) ; élagage non structuré (|w|, Wanda, hasard) et structuré, parcimonie 2:4 ; coût de vérification de k tokens ; décodage spéculatif implémenté à la main (0,5B -> 1,5B) avec taux d'acceptation, accélération mesurée et formule théorique ; inférence avec référence (*prompt lookup*) sur une réécriture de code ; Medusa (schéma) et décodage de Jacobi mesuré |
| [`ch09_cours_3_attention_et_noyaux.ipynb`](ch09_cours_3_attention_et_noyaux.ipynb) | *Attention mechanism optimization* (encadré *Calculating the KV Cache Size*, *Redesigning the attention mechanism*, *Optimizing the KV cache size*, *Writing kernels for attention computation*), *Kernels and compilers* | schéma d'un pas de decode avec cache KV ; génération avec et sans cache mesurée ; formule du cache vérifiée sur un vrai cache Hugging Face ; cache selon le contexte et le lot (Qwen, Llama) ; MHA, GQA, MQA (schéma), partage entre couches, fenêtre locale ; calcul de type Character.AI ; PagedAttention : carte mémoire, gaspillage et capacité simulés, taille des blocs ; fusion d'opérateurs mesurée (GELU naïve, `torch.compile`, noyau fusionné) ; tuilage et intensité ; attention naïve contre FlashAttention (SDPA) en temps et en mémoire ; abaissement et compilateurs ; étude de cas PyTorch |
| [`ch09_cours_4_service.ipynb`](ch09_cours_4_service.ipynb) | *Inference Service Optimization* (*Batching*, *Decoupling prefill and decode*, *Prompt caching*, *Parallelism*) | serveur vu comme une file (schéma) ; modèle de coût calibré sur des mesures GPU ; simulation à événements discrets des batchings statique, dynamique et continu (chronogrammes, courbes latence / débit) ; prefill et decode colocalisés contre dissociés (à-coups du TPOT, TTFT, rapport d'instances) ; cache de prompt mesuré (TTFT divisé) et économies calculées selon la tarification ; rangement de répliques ; parallélisme de tenseur (vérification numérique, schéma, coût de communication) ; pipeline (chronogramme, bulle) ; contexte et séquence |

## Notebooks d'exercices

| Notebook | Exercices |
|---|---|
| [`ch09_exercices_01-06.ipynb`](ch09_exercices_01-06.ipynb) | 9.1 à 9.6 |
| [`ch09_exercices_07-12.ipynb`](ch09_exercices_07-12.ipynb) | 9.7 à 9.12 |

- **Ex. 9.1** - Intensité arithmétique de cinq opérations et classement sur les rooflines du H100 et de la RTX.
- **Ex. 9.2** - MBU et MFU de Llama 2-70B sur 2 H100, puis MBU mesurée de Qwen2.5-0.5B en comptant le cache KV.
- **Ex. 9.3** - Goodput et percentiles du TTFT d'un serveur simulé (coûts mesurés) selon la charge ; charge maximale sous SLO.
- **Ex. 9.4** - Coût par million de tokens de Llama 3-8B auto-hébergé sur H100 contre une API ; utilisation minimale.
- **Ex. 9.5** - Dimensionner un service Llama 3-70B : mémoire, TPOT, loi de Little, nombre de répliques.
- **Ex. 9.6** - Quantification par groupes symétrique et asymétrique sur de vraies matrices ; vitesse de decode attendue.
- **Ex. 9.7** - Démonstration de la formule du décodage spéculatif, k optimal pour Chinchilla-70B, distribution mesurée des acceptations.
- **Ex. 9.8** - Cache KV de Llama 3-70B et utilisateurs simultanés sur 8 H100 (GQA, cache fp8, MHA).
- **Ex. 9.9** - PagedAttention : gaspillage moyen (b-1)/2 démontré et simulé, coût de la table des blocs.
- **Ex. 9.10** - Batching dynamique : effet de la fenêtre sur la latence p90, comparaison au batching continu.
- **Ex. 9.11** - Rentabilité du cache de prompt (stockage, durée de vie, arrivées de Poisson) et TTFT mesuré selon le préfixe.
- **Ex. 9.12** - Répartir Llama 3-70B sur 8 GPU : tenseur, pipeline, répliques ; latence contre débit.
