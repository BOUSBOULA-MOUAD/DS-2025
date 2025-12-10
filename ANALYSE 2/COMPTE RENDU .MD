# 📘 PROJET DATA SCIENCE  
## Analyse et Prédiction de la Performance Sportive

---

## 1. Contexte et Problématique

La performance sportive des athlètes dépend de multiples facteurs tels que la condition physique, l’intensité de l’entraînement, les paramètres physiologiques et le contexte de compétition. Une mauvaise évaluation de ces facteurs peut entraîner une baisse de performance ou un surentraînement.

L’utilisation de la Data Science permet d’analyser ces facteurs de manière objective et de construire des modèles capables de prédire la performance sportive.

---

## 2. Objectif du Projet

L’objectif principal de ce projet est de :
- analyser les facteurs influençant la performance sportive,
- prédire le niveau de performance d’un athlète,
- classifier les athlètes selon leur niveau de performance (faible/moyen ou haut niveau) à l’aide d’algorithmes de Machine Learning.

---

## 3. Présentation du Jeu de Données

Le jeu de données utilisé est **Athlete Performance Prediction Dataset**, disponible sur Kaggle.

Il s’agit de données tabulaires contenant :
- des informations physiques et physiologiques des athlètes,
- des paramètres liés à l’entraînement (heures, intensité, entraînement en altitude),
- des informations contextuelles sur la compétition.

La variable cible principale est :
- **Performance_Metric** : variable numérique continue représentant la performance globale de l’athlète.

Une variable binaire dérivée a été créée :
- **High_Performer** :
  - 1 : athlète de haut niveau (performance ≥ médiane)
  - 0 : athlète de niveau faible ou moyen

---

## 4. Préparation et Nettoyage des Données

Les données contiennent des valeurs manquantes et des variables de différents types.  
Les étapes suivantes ont été réalisées :
- imputation des variables numériques par la moyenne,
- imputation des variables catégorielles par la valeur la plus fréquente,
- encodage des variables catégorielles par la méthode One-Hot Encoding,
- normalisation des variables explicatives lorsque nécessaire.

Ces étapes garantissent la compatibilité des données avec les algorithmes de Machine Learning.

---

## 5. Analyse Exploratoire des Données (EDA)

Une analyse exploratoire a été menée afin de :
- étudier la distribution de la performance sportive,
- analyser la relation entre les variables explicatives et la performance,
- identifier les corrélations entre les variables numériques.

Des visualisations telles que des histogrammes, boxplots et matrices de corrélation ont été utilisées pour mieux comprendre les données.

---

## 6. Méthodologie Expérimentale

Les données ont été séparées en deux ensembles :
- **Ensemble d’entraînement** : 80 % des données
- **Ensemble de test** : 20 % des données

Cette séparation permet d’évaluer la capacité des modèles à généraliser sur de nouvelles données.

---

## 7. Algorithme Utilisé

L’algorithme principal utilisé pour l’apprentissage est la **Régression Logistique**.

Cet algorithme permet de :
- modéliser la probabilité qu’un athlète soit classé comme haut niveau,
- analyser l’influence des variables explicatives sur la performance sportive,
- résoudre efficacement un problème de classification binaire.

---

## 8. Évaluation des Résultats

Les performances du modèle ont été évaluées à l’aide de :
- la matrice de confusion,
- l’accuracy,
- la précision (Precision),
- le rappel (Recall),
- le F1-score.

Ces métriques permettent d’évaluer la qualité globale du modèle ainsi que sa capacité à identifier correctement les athlètes performants.

---

## 9. Conclusion

Ce projet montre que l’application des techniques de Data Science permet d’analyser et de prédire efficacement la performance sportive.  
Grâce à un processus structuré incluant le nettoyage des données, l’analyse exploratoire et la modélisation, il est possible d’extraire des informations utiles et d’aider à la prise de décision dans le domaine sportif.

---

