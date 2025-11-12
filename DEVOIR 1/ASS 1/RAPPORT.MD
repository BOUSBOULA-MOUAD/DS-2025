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
