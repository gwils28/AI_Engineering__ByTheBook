# AI Engineering - *By The Book*

Notebooks Jupyter pédagogiques qui accompagnent la lecture de
**AI Engineering: Building Applications with Foundation Models** (Chip Huyen, O'Reilly, 2025) :
cours illustrés, expériences sur de vrais modèles de langage exécutés en local, diagrammes
d'architecture et **exercices corrigés en détail**, chapitre par chapitre.

## Objectif et public

Les notebooks sont écrits en français pour un lecteur qui **sait programmer en Python** et
connaît les bases de l'apprentissage automatique, mais découvre l'ingénierie des
applications construites sur des modèles de fondation (LLM, modèles multimodaux).

Le livre est volontairement peu technique et ne propose pas d'exercices. Ces notebooks
ajoutent ce qui manque pour **comprendre en profondeur** :

- chaque concept est motivé, expliqué par une intuition, puis formalisé (softmax,
  entropie et perplexité, lois d'échelle, mémoire GPU, latence, coûts...) ;
- chaque idée est **vérifiée expérimentalement** sur de petits modèles ouverts
  (Qwen2.5 0,5B et 1,5B, GPT-2, modèles d'embeddings MiniLM) qui tournent sur un GPU
  grand public, sans clé d'API ;
- les architectures (RAG, agents, passerelles, caches, pipelines d'évaluation...) sont
  représentées par des **diagrammes** ;
- chaque chapitre propose **8 à 12 exercices** conçus pour ce dépôt (compréhension,
  calculs de dimensionnement, mini-labs de code), tous corrigés.

Chaque exercice suit le même plan :

1. **énoncé** autonome ;
2. **rappels** des notions du cours utilisées (avec renvoi à la section du livre) ;
3. **solution** détaillée pas à pas ;
4. **vérification** numérique, expérimentale et/ou graphique ;
5. un court **bilan** (« À retenir »).

## Organisation du dépôt

```
notebooks/chXX_<theme>/   notebooks de cours (chXX_cours_*) et d'exercices (chXX_exercices_*)
aie/                      paquet Python : style graphique, diagrammes, accès aux modèles
                          locaux, implémentations « à la main » des méthodes de chaque chapitre
tests/                    tests unitaires (pytest)
tools/                    construction et exécution des notebooks
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # ou la version CPU
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name aie
jupyter notebook
```

Les modèles (quelques Go au total) sont téléchargés depuis Hugging Face à la première
exécution. Un GPU NVIDIA de 8 Go suffit ; sur CPU, tout fonctionne mais plus lentement.
Les notebooks du dépôt sont fournis **déjà exécutés** : sorties et figures sont visibles
directement sur GitHub.

Lancer les tests : `pytest tests/`.

## Avancement

| Chapitre | Thème | Statut |
|---|---|---|
| 1 | [Introduction : construire des applications avec des modèles de fondation](notebooks/ch01_introduction_ai_engineering) | ✅ |
| 2 | [Comprendre les modèles de fondation](notebooks/ch02_comprendre_modeles_fondation) | ✅ |
| 3 | [Méthodologie d'évaluation](notebooks/ch03_methodologie_evaluation) | ✅ |
| 4 | [Évaluer des systèmes d'IA](notebooks/ch04_evaluer_systemes_ia) | ✅ |
| 5 | Prompt engineering | ⬜ |
| 6 | RAG et agents | ⬜ |
| 7 | Finetuning | ⬜ |
| 8 | Ingénierie des jeux de données | ⬜ |
| 9 | Optimisation de l'inférence | ⬜ |
| 10 | Architecture d'une application IA et retours utilisateurs | ⬜ |

## Références

- C. Huyen, *AI Engineering: Building Applications with Foundation Models*, O'Reilly Media,
  2025. Ressources complémentaires de l'autrice : https://github.com/chiphuyen/aie-book
- Les textes du livre ne sont pas reproduits ici : les notebooks reformulent, expliquent et
  prolongent ses idées, et renvoient aux sections correspondantes.
