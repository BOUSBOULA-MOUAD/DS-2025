# Bousboula mouad

<img src="mouad.jpeg" style="height:464px;margin-right:432px"/>

# THÈME : Sécurité et fiabilité des systèmes technologiques — étude de l’influence des facteurs environnementaux (température) sur la performance des composants critiques dans le domaine spatial.

# Descriptif du jeu de données : Challenger USA Space Shuttle O-Ring

## 🔷 Titre du jeu de données
**Challenger USA Space Shuttle O-Ring Data Set**

---

## 🔷 Grand Thème
**Sécurité et fiabilité des systèmes technologiques dans le domaine spatial**

Ce thème traite de la manière dont les conditions environnementales, notamment la température, peuvent influencer la performance et la sécurité des composants critiques utilisés dans les systèmes aérospatiaux.

---

## 🔷 Sous-thème
**Analyse statistique du risque de défaillance des joints d’étanchéité (“O-rings”) en fonction de la température de lancement.**

Ce sous-thème met l’accent sur l’étude de la relation entre la température et les défaillances observées sur les O-rings, dans le but d’identifier des facteurs de risque susceptibles d’affecter la fiabilité des navettes spatiales.

---

## 🔷 Contexte
Le 28 janvier 1986, la navette spatiale américaine **Challenger** a explosé peu après son décollage, causant la mort de ses sept astronautes.  
L’enquête menée par la NASA a révélé que la **défaillance des joints d’étanchéité (O-rings)**, due à une **température de lancement exceptionnellement basse (31 °F)**, avait provoqué une fuite de gaz chaud et conduit à la destruction du véhicule.  

Avant cette tragédie, plusieurs vols avaient déjà signalé des anomalies similaires, mais les données disponibles n’avaient pas été correctement exploitées.  
Ce jeu de données a donc été constitué pour analyser ces vols antérieurs et comprendre l’impact des conditions thermiques sur la performance des O-rings.

---

## 🔷 Description du contenu
Le jeu de données comprend **23 observations**, correspondant à différents vols de la navette spatiale, et comporte **5 variables principales** :  

| Variable | Description | Type |
|-----------|--------------|------|
| `temporal_order` | Numéro chronologique du vol | Numérique |
| `num_O_rings` | Nombre total d’O-rings examinés | Numérique |
| `num_thermal_distress` | Nombre d’O-rings ayant subi une défaillance thermique | Numérique |
| `launch_temp` | Température de lancement (en °F) | Numérique |
| `leak_check_pressure` | Pression de test d’étanchéité (en psi) | Numérique |

---

## 🔷 Objectif de l’analyse
L’objectif de cette étude est de :
- déterminer s’il existe une **corrélation entre la température de lancement et le nombre de défaillances des O-rings** ;
- **modéliser le risque de défaillance** en fonction de la température ;
- **prévoir le comportement** des O-rings à basse température, notamment autour de 31 °F (température du lancement du Challenger).

---

## 🔷 Méthodes d’analyse possibles
- Analyse descriptive et graphique de la relation température ↔ défaillances.  
- Application d’une **régression linéaire ou logistique** pour modéliser le risque.  
- Évaluation de la probabilité de défaillance selon la température.  
- Interprétation des résultats pour une meilleure **gestion du risque technique**.

---

## 🔷 Intérêt du jeu de données
Ce jeu de données présente un intérêt **pédagogique et scientifique majeur** :  
- Il illustre l’importance de l’**analyse de données dans la prise de décision technique**.  
- Il met en évidence comment une **mauvaise interprétation statistique** peut avoir des conséquences catastrophiques.  
- Il constitue un **cas d’étude concret** en matière de sécurité, de gestion du risque et de fiabilité des systèmes.

---

## 🔷 Source
- **Origine** : UCI Machine Learning Repository  
- **Licence** : Creative Commons Attribution 4.0 (CC BY)  
- **Lien** : [https://archive.ics.uci.edu/dataset/92/challenger+usa+space+shuttle+o+ring](https://archive.ics.uci.edu/dataset/92/challenger+usa+space+shuttle+o+ring)

---

## 🔷 Conclusion
Le jeu de données **Challenger O-Ring** illustre parfaitement le rôle essentiel de l’analyse statistique dans la **prévention des risques technologiques**.  
Son étude met en lumière la nécessité d’intégrer des approches de **fiabilité et de prédiction** dans la conception et l’exploitation des systèmes complexes, en particulier dans le domaine **aérospatial**.


# 🚀 Analyse du jeu de données : Challenger USA Space Shuttle O-Ring

## 📘 1. Introduction

Ce projet étudie la relation entre la **température de lancement** et les **défaillances des joints d’étanchéité (O-Rings)** du propulseur de la navette spatiale **Challenger**, dont l’explosion en 1986 fut causée par une défaillance mécanique liée au froid.

L’objectif est de montrer, par une analyse statistique et visuelle, comment la **baisse de température augmente le risque de défaillance**, et d’illustrer la nécessité d’une approche de fiabilité dans l’ingénierie aérospatiale.

---

## 🎯 2. Objectif du projet

- Analyser les **facteurs influençant les défaillances** des O-Rings.  
- Établir une **corrélation entre la température de lancement et les défaillances**.  
- Créer un **modèle prédictif** simple permettant d’estimer le risque à basse température.  
- Mettre en évidence l’importance de l’analyse des données dans la **prise de décision technique**.

---

## 🧩 3. Description du jeu de données

### Source  
- **Origine :** UCI Machine Learning Repository  
- **Lien :** [https://archive.ics.uci.edu/dataset/92/challenger+usa+space+shuttle+o+ring](https://archive.ics.uci.edu/dataset/92/challenger+usa+space+shuttle+o+ring)
- **Licence :** Creative Commons Attribution 4.0 (CC BY)

### Structure des données  
Le dataset comprend **23 observations** issues de vols précédents de la navette spatiale avant la catastrophe du Challenger.

| Variable | Description | Type |
|-----------|--------------|------|
| `Temperature` | Température de lancement (°F) | Numérique |
| `Failures` | Nombre de défaillances observées sur les O-Rings | Numérique |

---

## 📊 4. Exploration et statistiques descriptives

| Mesure | Température (°F) | Défaillances |
|---------|------------------|--------------|
| Moyenne | ≈ 69.57 | - |
| Médiane | ≈ 70.00 | - |
| Écart-type | ≈ 8.36 | - |
| Minimum | 53 °F | 0 |
| Maximum | 81 °F | 2 |

Les données montrent une **forte variabilité** de la température, mais très peu de défaillances sauf lors des lancements à basse température.

---

## 📉 5. Visualisation

### 🔹 Corrélation entre température et défaillance
Le graphique de dispersion (`Temperature` vs `Failures`) montre clairement une **corrélation négative** :  
> ➡️ plus la température diminue, plus le nombre de défaillances augmente.

```python
sns.scatterplot(x="Temperature", y="Failures", data=df, color="red")
plt.title("Température vs Défaillances des O-Rings")
