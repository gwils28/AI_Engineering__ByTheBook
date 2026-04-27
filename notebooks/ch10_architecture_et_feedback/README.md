# Chapitre 10 - Architecture d'une application d'IA et retours utilisateurs

Ce dernier chapitre assemble les techniques des chapitres précédents en une application
complète. L'architecture est construite **pas à pas**, comme en production : on part de
requête -> modèle -> réponse, puis on ajoute contexte, garde-fous, routeur et passerelle, caches,
motifs agentiques et actions d'écriture, avec un **diagramme qui s'enrichit à chaque étape** et
une **mini-application fonctionnelle** (l'assistant d'une boutique fictive, propulsé par
Qwen2.5-0.5B) qui grandit en même temps. On rend ensuite ce système observable (métriques,
journaux, traces, dérives, MTTD/MTTR), on l'orchestre, et on étudie les **retours
utilisateurs** : signaux conversationnels, conception de la collecte, biais et boucles de
rétroaction dégénérées.

Les fonctions réutilisables (diagramme évolutif, détection et masquage de données personnelles,
détection d'injection, garde-fous de sortie et relance, routeur d'intention, passerelle avec
repli et seau à jetons, caches exact et sémantique, classe `MiniApp`, simulation de production,
CUSUM, orchestrateur `Pipeline`, extraction de signaux conversationnels, simulations de biais)
sont dans [`aie/ch10_architecture.py`](../../aie/ch10_architecture.py),
testées par [`tests/test_ch10_architecture.py`](../../tests/test_ch10_architecture.py).

## Notebooks de cours

| Notebook | Sections du livre | Contenu |
|---|---|---|
| [`ch10_cours_1_architecture_contexte_gardefous.ipynb`](ch10_cours_1_architecture_contexte_gardefous.ipynb) | *AI Engineering Architecture*, *Step 1. Enhance Context*, *Step 2. Put in Guardrails* | démarche incrémentale et carte d'ensemble ; architecture minimale (coût, latence, trace) ; RAG sur la base documentaire et coût du contexte ; masquage / démasquage des données personnelles (emails, téléphones, IBAN avec clé modulo 97) et évaluation précision / rappel ; détection d'injection par motifs et ses limites ; garde-fous de sortie (format JSON, toxicité, refus) ; politique de relance séquentielle ou parallèle mesurée sur Qwen ; garde-fous et streaming |
| [`ch10_cours_2_routeur_passerelle_caches_agents.ipynb`](ch10_cours_2_routeur_passerelle_caches_agents.ipynb) | *Step 3. Add Model Router and Gateway*, *Step 4. Reduce Latency with Caches*, *Step 5. Add Agent Patterns* | routeur d'intention par embeddings MiniLM (centroïdes, k plus proches voisins, comparaison d'encodeurs, règle de marge pour les requêtes ambiguës, économies) ; passerelle (repli, disponibilité, queue de latence, seau à jetons, contrôle d'accès, journalisation) ; cache exact sous loi de Zipf (LRU, FIFO, borne théorique) et fuites ; cache sémantique (compromis taux de succès / erreurs selon le seuil, coût et latence) ; boucle agentique d'auto-vérification ; actions d'écriture avec confirmation et risque attendu |
| [`ch10_cours_3_observabilite_orchestration.ipynb`](ch10_cours_3_observabilite_orchestration.ipynb) | *Monitoring and Observability* (*Metrics*, *Logs and traces*, *Drift detection*), *AI Pipeline Orchestration* | monitoring vs observabilité, MTTD / MTTR / CFR ; tableau de bord de deux semaines de production simulée (panne, mise à jour silencieuse du modèle) ; corrélations et contrôles ponctuels ; journaux JSON et traces de la mini-application, débogage ; dérives du prompt (empreinte), du modèle (CUSUM) et des utilisateurs ; compromis délai / fausses alertes ; orchestrateur en graphe, chemin critique et parallélisme |
| [`ch10_cours_4_retours_utilisateurs.ipynb`](ch10_cours_4_retours_utilisateurs.ipynb) | *User Feedback*, *Extracting Conversational Feedback*, *Feedback Design*, *Feedback Limitations* | volant d'inertie des données ; retours explicites (rares, biaisés) et implicites (abondants, bruités) ; extraction heuristique des signaux (interruptions, corrections, reformulations par embeddings, plaintes, sentiment, refus, régénérations) et score implicite ; classification des retours par Qwen comparée aux heuristiques ; trajectoire du sentiment ; conception de la collecte (moments, échantillonnage, éditions comme paires de préférences) ; biais de complaisance, de position (randomisation et correction), de sélection ; boucles dégénérées (glouton, epsilon-glouton, IPW) et complaisance des modèles |

## Notebooks d'exercices

| Notebook | Exercices |
|---|---|
| [`ch10_exercices_01-05.ipynb`](ch10_exercices_01-05.ipynb) | 10.1 à 10.5 |
| [`ch10_exercices_06-11.ipynb`](ch10_exercices_06-11.ipynb) | 10.6 à 10.11 |

- **Ex. 10.1** - Choisir et ordonner les composants pour un assistant RH auto-hébergé, un chatbot d'e-commerce et un agent bancaire ; diagramme de l'agent bancaire.
- **Ex. 10.2** - Étendre le masquage aux cartes bancaires avec la clé de Luhn : probabilité 1/10, détection des erreurs d'un chiffre, précision et rappel.
- **Ex. 10.3** - Dimensionner une politique de relance : nombre d'appels pour 99,9 %, centiles de latence, séquentiel / parallèle / hybride.
- **Ex. 10.4** - Régler la marge de rejet d'un routeur d'intention en chiffrant le coût de chaque type d'erreur.
- **Ex. 10.5** - Disponibilité d'une passerelle avec repli et causes communes ; dimensionner un seau à jetons pour absorber des rafales.
- **Ex. 10.6** - Seuil optimal d'un cache sémantique, condition de rentabilité, sensibilité à l'ordre et à la part de reformulations.
- **Ex. 10.7** - Reproduire une fuite de données par le cache et comparer les corrections (filtre, clé de segment).
- **Ex. 10.8** - Régler une alerte CUSUM sur un budget de fausses alertes ; délai de détection, approximation théorique, comparaison à une moyenne glissante.
- **Ex. 10.9** - Chemin critique d'un pipeline et placement d'un détecteur d'injection lent (série ou parallèle avec annulation).
- **Ex. 10.10** - Apprendre un score de satisfaction implicite par régression logistique et le comparer aux poids manuels.
- **Ex. 10.11** - Biais de position et puissance statistique ($n \propto 1/(1-\pi)^2$) ; exploration contre correction IPW dans une boucle de rétroaction.
