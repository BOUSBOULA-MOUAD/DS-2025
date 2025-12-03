

# Compte-rendu scientifique détaillé

**Projet : Online Shoppers Purchasing Intention**
**Thématique : Marketing & Commerce**


---

## Résumé (Abstract)

Le présent rapport étudie la capacité à prédire, à partir du comportement de navigation, si une session utilisateur d’un site e-commerce aboutira à un achat (`Revenue`). Le dataset utilisé contient 12 330 sessions et un jeu de variables décrivant les actions des visiteurs (durées, pages consultées, métriques de sortie, canal de trafic, etc.). Nous réalisons une analyse exploratoire approfondie, construisons un pipeline de prétraitement, testons plusieurs modèles supervisés (régression logistique, Random Forest, XGBoost) et évaluons les performances à l’aide de métriques pertinentes (Accuracy, Precision, Recall, F1-Score, ROC-AUC). Enfin, nous discutons les limites et proposons des pistes d’amélioration opérationnelles pour un usage marketing.

---

## 1. Introduction

Le commerce en ligne repose sur la conversion des visiteurs en acheteurs. Comprendre les comportements qui précèdent un achat permet d’optimiser l’interface, les campagnes marketing et la personnalisation. L’objectif de ce projet est de développer et d’évaluer des modèles prédictifs capables d’identifier, en temps réel ou quasi-temps réel, la probabilité d’achat d’une session.

**Question de recherche :** Peut-on prédire si une session aboutira à un achat à partir des métriques de navigation et des caractéristiques de la session ?

**Objectifs opérationnels :**

1. Décrire et comprendre les déterminants de la conversion.
2. Construire un modèle de classification binaire robuste.
3. Mesurer la performance selon des métriques utiles pour le marketing (F1, ROC-AUC).
4. Proposer des actions exploitables (ex : ciblage visiteurs à forte probabilité d’achat).

---

## 2. Description du jeu de données

**Source :** Kaggle — *Online Shoppers Purchasing Intention Dataset* (imakash3011) / notebook de référence : `asmitameghrajchaskar/online-shopping-intention-analysis-with-python`.

**Taille :** 12 330 lignes (sessions) × 18 variables.

**Variables principales :**

* `Administrative`, `Administrative_Duration` — nombre/temps pages administratives
* `Informational`, `Informational_Duration` — pages info
* `ProductRelated`, `ProductRelated_Duration` — pages produits
* `BounceRates`, `ExitRates`, `PageValues` — métriques d’engagement
* `SpecialDay` — proximité d’un évènement spécial
* `OperatingSystems`, `Browser`, `Region`, `TrafficType` — facteurs techniques / source trafic
* `VisitorType`, `Month`, `Weekend` — variables contextuelles
* `Revenue` — **target** (True/False)

**Remarques qualité :**

* Jeu globalement propre ; quelques valeurs manquantes éventuelles ont été éliminées (cf. preprocessing).
* Distribution de la cible : la proportion d’acheteurs est plus faible que celle des non-acheteurs → **déséquilibre de classes** à prendre en compte lors de l’évaluation.

---

## 3. Méthodologie

### 3.1 Pipeline général

1. **Chargement** du CSV.
2. **Nettoyage** : suppression/gestion des valeurs manquantes, conversion `Revenue` → binaire 0/1.
3. **EDA** (exploration) : distributions, heatmap, scatter plots, boxplots, K-Means (visual).
4. **Prétraitement** : encodage des catégorielles (One-Hot), scaling des numériques (StandardScaler).
5. **Sélection de features** et Feature Engineering (ratios, totaux, flags).
6. **Modélisation** : test de plusieurs modèles, validation croisée, optimisation hyperparamètres.
7. **Évaluation** : métriques et matrice de confusion.
8. **Interprétation** et recommandations.

### 3.2 Prétraitement détaillé

* **Encodage** : One-Hot sur `VisitorType`, `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `Weekend`.
* **Scaling** : StandardScaler pour `*_Duration`, `BounceRates`, `ExitRates`, `PageValues`, etc.
* **Gestion du déséquilibre** : essais avec `class_weight='balanced'` pour certains algorithmes et essais avec SMOTE (oversampling) si nécessaire.
* **Feature engineering** (exemples) :

  * `TotalPages = Administrative + Informational + ProductRelated`
  * `AvgDurationPerPage = (Administrative_Duration + Informational_Duration + ProductRelated_Duration) / TotalPages`
  * `IsHolidayPeriod = SpecialDay > 0.5` (flag)

### 3.3 Modèles testés

* **Régression logistique** (baseline, interprétable).
* **Random Forest** (robuste, bonne prise en compte d’interactions).
* **XGBoost / Gradient Boosting** (souvent meilleur en production pour ce type de tabulaire).
* **(Optionnel)** SVM / KNN pour vérifier la robustesse.

### 3.4 Stratégie d’évaluation

* **Split** : 70% entraînement / 30% test (ou 80/20 selon contrainte).
* **Validation** : Cross-validation stratifiée 5-fold pour comparaison et GridSearchCV/RandomizedSearchCV pour réglage d’hyperparamètres.
* **Métriques** : Accuracy, Precision, Recall, F1-Score, ROC-AUC. Pour le business, F1 et ROC-AUC sont prioritaires car la classe positive (achat) est minoritaire.
* **Interprétabilité** : importance des features (RandomForest.feature_importances_ / coefficients de la régression logistique), courbes ROC, matrice de confusion.

---

## 4. Analyse exploratoire (exemples et interprétations attendues)

> **Exécuter les commandes suivantes** pour générer les figures et les coller ici :

```python
# Heatmap corrélation
plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), annot=True, fmt=".2f")
plt.show()

# Distribution de la target
sns.countplot(x='Revenue', data=data)
plt.show()

# Scatter ProductRelated_Duration vs BounceRates
sns.scatterplot(x='ProductRelated_Duration', y='BounceRates', hue='Revenue', data=data, alpha=0.5)
plt.show()

# Boxplot ProductRelated_Duration par Revenue
sns.boxplot(x='Revenue', y='ProductRelated_Duration', data=data)
plt.show()
```

### Observations typiques (à vérifier sur tes figures) :

* Les sessions avec `Revenue = True` tendent à avoir des `ProductRelated_Duration` plus élevés.
* `BounceRates` est plus faible pour les sessions ayant abouti à un achat.
* `PageValues` a généralement une corrélation positive avec `Revenue`.
* `VisitorType = Returning_Visitor` montre un taux de conversion plus élevé.

*(Colle ici les figures ou description de ce que tu obtiens)*

---

## 5. Modélisation et résultats

> **Exécute ces snippets** (ou adapte-les) pour obtenir les métriques ; colle ensuite les sorties ci-dessous.

### 5.1 Préparation des données (code court)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# features / target
features = ['Administrative','Administrative_Duration','Informational','Informational_Duration',
            'ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues','SpecialDay',
            'OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend','Month']
X = data[features]
y = data['Revenue'].astype(int)

cat_cols = ['OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend','Month']
num_cols = [c for c in features if c not in cat_cols]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
```

### 5.2 Modèle 1 — Régression logistique (baseline)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

pipe_lr = Pipeline([('pre', preprocessor), ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))])
pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)
y_proba_lr = pipe_lr.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_lr))
```

**Résultats (Régression logistique) :**

* Accuracy : `[INSÉRER]`
* Precision (classe 1) : `[INSÉRER]`
* Recall (classe 1) : `[INSÉRER]`
* F1-Score (classe 1) : `[INSÉRER]`
* ROC-AUC : `[INSÉRER]`

**Matrice de confusion (LR) :**
`[COLLER ICI la matrice 2×2 ou screenshot]`

---

### 5.3 Modèle 2 — Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
pipe_rf = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))])
pipe_rf.fit(X_train, y_train)
y_pred_rf = pipe_rf.predict(X_test)
y_proba_rf = pipe_rf.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))
```

**Résultats (Random Forest) :**

* Accuracy : `[INSÉRER]`
* Precision : `[INSÉRER]`
* Recall : `[INSÉRER]`
* F1 : `[INSÉRER]`
* ROC-AUC : `[INSÉRER]`

---

### 5.4 Modèle 3 — XGBoost (optionnel)

```python
import xgboost as xgb
pipe_xgb = Pipeline([('pre', preprocessor), ('clf', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))])
pipe_xgb.fit(X_train, y_train)
y_pred_xgb = pipe_xgb.predict(X_test)
y_proba_xgb = pipe_xgb.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_xgb))
```

**Résultats (XGBoost) :**

* Accuracy : `[INSÉRER]`
* Precision : `[INSÉRER]`
* Recall : `[INSÉRER]`
* F1 : `[INSÉRER]`
* ROC-AUC : `[INSÉRER]`

---

### 5.5 Comparaison synthétique (tableau)

Crée un tableau comparatif des modèles (ex : Accuracy / F1 / ROC-AUC) :

```
| Modèle            | Accuracy | Precision | Recall | F1    | ROC-AUC |
|-------------------|----------|-----------|--------|-------|---------|
| LogisticRegression|  ...     |   ...     |  ...   |  ...  |   ...   |
| RandomForest      |  ...     |   ...     |  ...   |  ...  |   ...   |
| XGBoost           |  ...     |   ...     |  ...   |  ...  |   ...   |
```

*(Insère tes résultats réels dans le tableau.)*

---

## 6. Analyse des erreurs et interprétation

### 6.1 Matrice de confusion (exemple d’interprétation)

* **Faux négatifs (FN)** : sessions réellement acheteuses prédites comme non-acheteuses → perte potentielle de chiffre d’affaires ; priorité à réduire si objectif marketing = capter acheteurs.
* **Faux positifs (FP)** : sessions prédites comme acheteuses mais qui ne le sont pas → coût possible si on déclenche actions marketing coûteuses (ex : coupon).

Donne ici les nombres concrets et calcule :

* Taux de faux négatifs = FN / (FN + TP)
* Taux de faux positifs = FP / (FP + TN)

### 6.2 Importance des variables

Pour RandomForest ou XGBoost, récupère `feature_importances_` et affiche les top-10 variables :

```python
importances = pipe_rf.named_steps['clf'].feature_importances_
# Si OneHotEncoder -> reconstruire noms des features après preprocessing (ou utiliser shap)
```

**Interprétation attendue :** `PageValues`, `ProductRelated_Duration`, `BounceRates`, `VisitorType` souvent en tête.

*(Insère ton graphique ou liste des features ici.)*

---

## 7. Conclusion

### Bilan

* Un modèle simple (régression logistique) permet d’obtenir une baseline interprétable.
* Les modèles d’ensemble (RandomForest / XGBoost) améliorent généralement la précision et le ROC-AUC.
* Les variables liées à l’engagement (temps produit, page value, bounce rate) sont déterminantes.

### Limites

1. **Déséquilibre de classes** : impact sur la détection des achats (classe minoritaire).
2. **Variables manquantes potentielles** : logs enrichis (scroll, clicks) pourraient améliorer le modèle.
3. **Généralisation** : dataset sur une période donnée — attention à la temporalité / saisonnalité.

### Pistes d’amélioration

* Essayer SMOTE ou class_weight pour réduire les FN.
* Faire une recherche d’hyperparamètres (GridSearchCV/RandomizedSearchCV).
* Utiliser SHAP pour interprétabilité fine (expliquer prédiction par session).
* Déployer un scoring en ligne + A/B test pour mesurer gain réel (ex : augmentation du taux de conversion via triggers).

---

## 8. Annexes (code utile & commandes)

### Générer et enregistrer la matrice de confusion :

```python
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, cmap='Blues')
plt.show()
```

### Courbe ROC :

```python
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(y_test, y_proba_rf)
plt.show()
```

### Calcul F1/Precision/Recall :

```python
from sklearn.metrics import precision_score, recall_score, f1_score
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
```

### Obtenir feature importances (RandomForest) — récupérer noms réels :

```python
# Récupérer noms après OneHotEncoder
ohe = preprocessor.named_transformers_['cat']
ohe_features = ohe.get_feature_names_out(cat_cols)
feature_names = num_cols + list(ohe_features)
importances = pipe_rf.named_steps['clf'].feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
feat_imp.head(15).plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()
```

