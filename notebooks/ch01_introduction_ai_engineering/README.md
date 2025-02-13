# Chapitre 1 - Introduction à la construction d'applications d'IA avec des modèles de fondation

Le premier chapitre explique pourquoi l'**échelle** a transformé l'IA : des modèles de langage
entraînés par auto-supervision sont devenus des LLM, puis des modèles de fondation
multimodaux et généralistes, accessibles par API ; d'où l'essor de l'**AI engineering**,
l'art de construire des applications sur ces modèles. Il passe ensuite en revue les grands
cas d'usage, la façon de planifier une application (faut-il la construire ? quelles
attentes ? quels jalons ?) et la pile technique, en la comparant au ML engineering classique
et au développement full-stack. Les notebooks ajoutent au texte du livre des calculs
(coûts, mémoire, latence, statistiques de décision) et des expériences sur de petits
modèles locaux (GPT-2, Qwen2.5-0.5B/1.5B, DistilRoBERTa, cinq tokenizers réels).

Les fonctions réutilisables (statistiques du langage et n-grammes, interface commune aux
tokenizers, mini-BPE, paires auto-supervisées, tailles et paramètres des modèles, coûts en
tokens, latence, grille de décision, modèle du « dernier kilomètre », intervalle de Wilson,
analyse construire / acheter) sont dans [`aie/ch01_introduction.py`](../../aie/ch01_introduction.py),
testées par [`tests/test_ch01_introduction.py`](../../tests/test_ch01_introduction.py).

## Notebooks de cours

| Notebook | Sections du livre | Contenu |
|---|---|---|
| [`ch01_cours_1_modeles_de_langage_et_tokens.ipynb`](ch01_cours_1_modeles_de_langage_et_tokens.ipynb) | *The Rise of AI Engineering* : *Language models* | Fréquences de lettres et entropie (Holmes, Shannon) ; n-grammes de caractères et entropie conditionnelle ; GPT-2 : P(« blue ») contre P(« car ») ; cinq tokenizers comparés (GPT-2, cl100k, o200k, Qwen2.5, XLM-R) ; BPE écrit de zéro, compromis vocabulaire / longueur, mots inconnus ; « taxe » en tokens de 7 langues ; modèles masqués contre autorégressifs (diagramme, masques d'attention, DistilRoBERTa contre GPT-2) ; génération gloutonne pas à pas, tâches reformulées en complétion, complétion contre conversation |
| [`ch01_cours_2_autosupervision_echelle_fondation.ipynb`](ch01_cours_2_autosupervision_echelle_fondation.ipynb) | *Self-supervision* ; *From Large Language Models to Foundation Models* ; *From Foundation Models to AI Engineering* | Supervision contre auto-supervision (diagramme, coût de l'étiquetage) ; paires (contexte, cible), teacher forcing, surprise et perplexité de GPT-2 ; croissance des tailles de modèles, formule $N \approx 12Ld^2 + Vd$ vérifiée sur GPT-2 et Qwen2.5, mémoire des poids ; règle de Chinchilla et calcul $6ND$ ; multimodalité (diagramme, tokens d'image, apprentissage contrastif à la CLIP illustré par analogie) ; un modèle pour plusieurs tâches ; prompt, RAG et finetuning (démonstration sur une fiche produit) ; trois facteurs de l'essor de l'AI engineering, baisse des coûts |
| [`ch01_cours_3_cas_usage_et_planification.ipynb`](ch01_cours_3_cas_usage_et_planification.ipynb) | *Foundation Model Use Cases* ; *Planning AI Applications* | Taxonomie des huit familles de cas d'usage ; exposition des métiers ; carte risque / évaluabilité ; gains de productivité mesurés ; démonstrations « Fast Breakdown » et extraction JSON (0,5 Md contre 1,5 Md) ; raisons de construire, rôle de l'IA (critique, proactive, dynamique) ; Crawl-Walk-Run et intervalle de Wilson ; défendabilité et volant de données ; seuil d'utilité : latence mesurée (TTFT, TPOT) et coût par requête ; dernier kilomètre (exponentielle contre loi de puissance) ; construire ou acheter avec des prix en baisse ; couche d'abstraction et suite d'évaluation |
| [`ch01_cours_4_pile_ai_engineering.ipynb`](ch01_cours_4_pile_ai_engineering.ipynb) | *The AI Engineering Stack* : *Three Layers of the AI Stack*, *AI Engineering Versus ML Engineering*, *AI Engineering Versus Full-Stack Engineering* | Diagramme des trois couches ; grille d'expériences (3 prompts × 0 ou 3 exemples) ; adaptation par le prompt ou par le finetuning ; tailles des modèles classiques et de fondation ; sorties ouvertes (10 réponses échantillonnées) ; pré-entraînement, post-entraînement, finetuning ; déduplication ; prefill parallèle contre décodage séquentiel (mesures) ; l'exemple Gemini / GPT-4 sur MMLU ; interfaces ; flux de travail inversé |

## Notebooks d'exercices

| Notebook | Exercices |
|---|---|
| [`ch01_exercices_01-05.ipynb`](ch01_exercices_01-05.ipynb) | 1.1 à 1.5 |
| [`ch01_exercices_06-10.ipynb`](ch01_exercices_06-10.ipynb) | 1.6 à 1.10 |

- **Ex. 1.1** - La taxe linguistique des tokens : ratio français / anglais selon cinq tokenizers et surcoût mensuel d'une application.
- **Ex. 1.2** - Construire un BPE à la main (trois fusions détaillées), vérifier avec le code, rendements décroissants des fusions.
- **Ex. 1.3** - Découper un texte en paires auto-supervisées : `<BOS>`/`<EOS>`, fenêtre de contexte, objectif masqué et signal d'apprentissage par passe.
- **Ex. 1.4** - Coût de l'étiquetage (tarification par paliers) contre auto-supervision : ce que coûterait l'écriture humaine du corpus de GPT-3.
- **Ex. 1.5** - Masqué ou autorégressif ? Choix argumenté pour six tâches, puis mini-lab DistilRoBERTa contre GPT-2 sur des phrases à trou.
- **Ex. 1.6** - Coût mensuel d'un chatbot multi-tours : formule quadratique de l'historique, petit contre grand modèle, leviers de réduction.
- **Ex. 1.7** - Marge et latence d'une application par abonnement : utilisateurs intensifs (simulation log-normale), seuil de rentabilité, latence.
- **Ex. 1.8** - Grille de décision « faut-il construire cette application d'IA ? » sur cinq projets, robustesse aux pondérations (Dirichlet).
- **Ex. 1.9** - Planifier les jalons : ajustement exponentiel contre loi de puissance, bootstrap de l'échéance, planning Crawl-Walk-Run.
- **Ex. 1.10** - Construire ou acheter quand les prix des API baissent : seuil de rentabilité analytique et carte (volume, demi-vie du prix).
