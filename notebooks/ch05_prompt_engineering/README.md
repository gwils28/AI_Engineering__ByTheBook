# Chapitre 5 - Ingénierie des prompts

L'ingénierie des prompts est la technique d'adaptation la plus simple et la plus répandue : elle
guide un modèle de fondation sans toucher à ses poids. Ce chapitre la traite comme une discipline
expérimentale : chaque conseil du livre (exemples few-shot, gabarits de conversation, position de
l'information, consignes explicites, chaîne de pensée, versionnage) est **mesuré** sur
Qwen2.5-0.5B-Instruct, avec une métrique et sa variance. La seconde moitié adopte le point de vue
de l'attaquant (extraction du prompt, jailbreak, injection directe et indirecte, extraction
d'informations) et mesure l'efficacité réelle des défenses, toujours sur des cibles inoffensives.

Les fonctions réutilisables (anatomie et assemblage de prompts, jeu de classification d'intentions
en français, score des étiquettes, visualisation des gabarits de conversation, botte de foin,
registre de prompts versionnés, délimiteurs, détecteur d'injection, canaris, métriques de
violation et de faux refus) sont dans [`aie/ch05_prompts.py`](../../aie/ch05_prompts.py),
testées par [`tests/test_ch05_prompts.py`](../../tests/test_ch05_prompts.py).

## Notebooks de cours

| Notebook | Sections du livre | Contenu |
|---|---|---|
| [`ch05_cours_1_anatomie_et_icl.ipynb`](ch05_cours_1_anatomie_et_icl.ipynb) | Introduction to Prompting ; In-Context Learning: Zero-Shot and Few-Shot ; Terminology Ambiguity: Prompt Versus Context | anatomie d'un prompt et coût en tokens de chaque partie ; robustesse à six reformulations ; courbe de précision en fonction du nombre d'exemples (exemples en tours de conversation vs en ligne) ; matrices de confusion ; sensibilité à l'ordre et au choix des exemples, effondrement et biais de récence ; le prompt comme sélecteur de « programme » |
| [`ch05_cours_2_gabarits_et_contexte.ipynb`](ch05_cours_2_gabarits_et_contexte.ipynb) | System Prompt and User Prompt ; Context Length and Context Efficiency | système et utilisateur concaténés en une séquence ; gabarit ChatML réel de Qwen visualisé token par token (tokens spéciaux, prompt système par défaut, rôle outil) ; comparaison avec Llama 2 / Llama 3 ; erreurs de gabarit mesurées ; placement de la consigne et conflits système / utilisateur ; course aux fenêtres de contexte, cache KV, coût quadratique mesuré ; aiguille dans une botte de foin (facile puis à leurres) et courbe en U |
| [`ch05_cours_3_bonnes_pratiques.ipynb`](ch05_cours_3_bonnes_pratiques.ipynb) | Prompt Engineering Best Practices (toutes les sous-sections) ; Evaluate Prompt Engineering Tools ; Organize and Version Prompts | consignes vagues vs explicites (respect du format) ; persona et note espérée ; coût des formats d'exemples, marqueurs de fin, conformité JSON ; contexte suffisant et abstention ; décomposition en sous-tâches (monolithique vs chaîné) ; chaîne de pensée et auto-critique mesurées ; registre de prompts versionnés (épinglage, diff, métriques) ; nombre d'appels des outils et « malédiction du vainqueur » de l'optimisation automatique |
| [`ch05_cours_4_ingenierie_defensive.ipynb`](ch05_cours_4_ingenierie_defensive.ipynb) | Defensive Prompt Engineering : Proprietary Prompts and Reverse Prompt Engineering ; Jailbreaking and Prompt Injection ; Information Extraction ; Defenses Against Prompt Attacks | diagramme des surfaces d'attaque ; extraction d'un canari du prompt système et authenticité des prompts « extraits » ; jailbreaks par technique (jeu de rôle, format, langue, DAN...) et attaque automatisée par enrobages ; injection directe et défenses (délimiteurs, sandwich, hiérarchie, détection) ; injection indirecte dans un assistant de messagerie, portier d'approbation et filtre d'URL ; sondage factuel et mémorisation verbatim ; hiérarchie d'instructions, défense en profondeur, bilan violation / faux refus |

## Notebooks d'exercices

| Notebook | Exercices |
|---|---|
| [`ch05_exercices_01-06.ipynb`](ch05_exercices_01-06.ipynb) | 5.1 à 5.6 |
| [`ch05_exercices_07-12.ipynb`](ch05_exercices_07-12.ipynb) | 5.7 à 5.12 |

- **Ex. 5.1** - Coût mensuel d'un prompt poste par poste et leviers d'économie (exemples, format, cache de préfixe, longueur des réponses).
- **Ex. 5.2** - Rendement marginal du few-shot : courbe mesurée, modèle de saturation et nombre d'exemples optimal selon la valeur d'une bonne réponse.
- **Ex. 5.3** - Calibration contextuelle (Zhao et al., 2021) pour corriger le biais d'étiquettes d'un prompt few-shot.
- **Ex. 5.4** - Réécrire le gabarit ChatML de Qwen à la main, débusquer un gabarit bogué par diff de tokens et mesurer son effet.
- **Ex. 5.5** - Budget de contexte : cache KV de Qwen2.5-0.5B et 7B, requêtes simultanées sur un GPU de 24 Go.
- **Ex. 5.6** - « Perdu au milieu » : test exact de Fisher et remède par la position de la question.
- **Ex. 5.7** - Sortie JSON fiable : probabilité de succès avec relances, nombre espéré d'appels, vérification sur Qwen.
- **Ex. 5.8** - Chaîne de pensée et auto-cohérence : précision, tokens et coût par bonne réponse.
- **Ex. 5.9** - Comparer deux versions de prompt : test de McNemar apparié et taille d'échantillon nécessaire.
- **Ex. 5.10** - Détecteur d'injection : courbe ROC, contournement par obfuscation, détecteur sémantique par embeddings.
- **Ex. 5.11** - Défense en profondeur : formules sous indépendance et mesure réelle de l'empilement de couches.
- **Ex. 5.12** - Injection indirecte dans un mini-RAG (BM25) et défenses à l'indexation, dans le prompt et en sortie.
