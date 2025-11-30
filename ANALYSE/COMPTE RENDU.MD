# Compte-rendu : AI Impact on Jobs 2030 Analysis

## 📄 Résumé du notebook

-   Le projet analyse le jeu de données “AI Impact on Jobs 2030” (3 000 entrées) provenant de Kaggle.
-   L'auteur effectue un nettoyage des données, une analyse exploratoire (histogrammes, matrice de corrélation) pour comprendre les facteurs influençant l'automatisation.
-   Objectif principal : **prédire la probabilité d'automatisation** (`automation_probability_2030`) en fonction du salaire, de l'expérience et des compétences.

---

## 🔧 Types de régressions proposées et leurs objectifs

### • Régression linéaire

-   Estime une relation linéaire entre les caractéristiques (salaire, compétences, etc.) et la probabilité d'automatisation.
-   Variable cible : `automation_probability_2030` (continue).
-   Modèle appliqué après normalisation des données (`StandardScaler`).
-   Intérêt : Tenter de prédire un score de risque précis pour chaque emploi.

### • Régression logistique

-   Utilisée après transformation de la cible continue en variable binaire (seuil médian à 0.5).
-   Modélise la **classification** du risque (Risque élevé vs Faible risque).
-   Objectif : Déterminer si un emploi est susceptible d'être automatisé ou non.

---

## ✅ Avantages & limites observés

| Modèle | Avantages | Limites (dans ce projet) |
| :--- | :--- | :--- |
| **Régression linéaire** | Prédit une probabilité continue, simple à interpréter. | **R² négatif (-0.0062)** : Le modèle ne trouve aucune corrélation linéaire et performe moins bien qu'une simple moyenne. |
| **Régression logistique** | Permet une classification binaire claire, fournit des probabilités de classe. | **Exactitude ~50%** : Le modèle équivaut au hasard (pile ou face). Les données ne présentent pas de frontières de décision claires. |

---

## 🎯 Analyse

-   Pour prédire la probabilité exacte, la **régression linéaire** a été testée mais a échoué à capturer une tendance (données probablement aléatoires).
-   Pour classifier le risque, la **régression logistique** a montré qu'il n'y a pas de distinction statistique entre les classes dans ce jeu de données.
-   L'analyse exploratoire (EDA) et les matrices de corrélation montrent une absence de lien fort entre les compétences et le risque d'automatisation.

---

## 💡 Recommandations

-   **Qualité des données** : Les métriques suggèrent que ce jeu de données est synthétique/aléatoire. Il est recommandé d'utiliser des données économiques réelles pour obtenir des résultats pertinents.
-   **Modélisation** : Si des données réelles sont disponibles, tester des modèles non linéaires (Forêts aléatoires, XGBoost) pour capturer des interactions complexes entre compétences et automatisation.
-   **Feature Engineering** : Créer de nouvelles variables ou segmenter les métiers par industrie pour améliorer la performance des régressions.
