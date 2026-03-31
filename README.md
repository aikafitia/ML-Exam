# Projet Machine Learning – Morpion Intelligent (Tic-Tac-Toe AI)

---

## Institut Supérieur Polytechnique de Madagascar (ISPM)

🔗 Site officiel : https://www.ispm-edu.com

---

## Nom du Groupe

**Golden AI**

---

## Membres du Groupe

* **ARISOA Aika Fitia n°9, IGGLIA 4**
* **ANJARATIANA NY AINA Antsa Christiana n°29, IGGLIA 4**
* **RALAIAVY Fetraniaina Paul n°37, IGGLIA 4**
* **RAKOTO Jean De Rivaze Jocelyn n°38, IGGLIA 4**

---

## Description du Projet

Ce projet consiste à développer une intelligence artificielle capable de jouer au jeu du **Morpion** en utilisant des techniques de **Machine Learning**.

L’objectif principal est de :

* Analyser des états de jeu
* Prédire les probabilités de victoire ou de match nul
* Implémenter une IA capable de prendre des décisions intelligentes

Trois modes de jeu ont été développés :

* **Human vs Human**
* **Human vs IA (Machine Learning)**
* **Human vs IA Hybride (Minimax + ML)**

---

## Structure du Repository

```
ML-Exam/
│
|── generator/
|   └──generate_dataset.py
|
├── ressources/
│   └── dataset.csv
│
├── notebooks/
│   ├── notebook.ipynb
│   ├── notebookAvance.ipynb
│   └── notebookTraining.ipynb
│
├── models/
│   ├── model_xwins.pkl
│   └── model_draw.pkl
│
├── interfaces/
│   └── games.py
│
├── README.md
|── .gitignore
```

---

## 📊 Résultats Machine Learning

### 🔹 Baseline – Régression Logistique

| Modèle  | Accuracy      | F1-score      |
| ------- | ------------- | ------------- |
| x_wins  |     0.64      |     0.71      |
| is_draw |     0.59      |     0.72      |

---

### 🔹 Modèles Avancés

Modèles testés :

* Random Forest
* Gradient Boosting

| Modèle       | Accuracy      | F1-score      |
| ------------ | ------------- | ------------- |
| x_wins (RF)  |    0.86       |    0.91       |
| is_draw (RF) |    0.92       |    0.95       |

**Conclusion :**
Les modèles avancés surpassent la régression logistique, notamment grâce à leur capacité à capturer des relations non linéaires.

---

## Exploratory Data Analysis (EDA)

### Distribution des cibles

* **x_wins** : dataset relativement équilibré
* **is_draw** : classe minoritaire → problème de déséquilibre

---

### Case la plus importante pour X

La case **centrale (index 4)** est la plus corrélée avec la victoire de X.

---

### Corrélation des features

* Forte corrélation entre certaines positions et la victoire
* Les diagonales jouent un rôle clé

---

## Réponses aux Questions

### 1. Le dataset est-il équilibré ?

* **x_wins** : relativement équilibré
* **is_draw** : déséquilibré (peu de matchs nuls)

---

### 2. Quelle cible est la plus difficile à apprendre ?

    **is_draw**

**Pourquoi ?**

* Moins de données
* Cas plus complexes
* Peu de patterns clairs

---

### 3. Pourquoi utiliser deux modèles ?

Parce que :

* Les objectifs sont différents
* La distribution des classes diffère
* Meilleure spécialisation des modèles

---

### 4. Pourquoi le modèle hybride est meilleur ?

Le modèle hybride combine :

* **Recherche Minimax (logique)**
* **Évaluation ML (probabiliste)**

Résultat : décisions plus intelligentes et plus réalistes

---

## Interface Utilisateur

Une interface graphique a été développée avec **Tkinter**, permettant :

* Interaction intuitive
* Choix du mode de jeu
* Visualisation en temps réel

---

## Lancer le projet

### Lancer le jeu

```
python interfaces/game.py
```

---

## Vidéo de Présentation

🔗 Lien : ****

---

## Améliorations Possibles

* UI plus moderne (PyQt / Web)
* IA Deep Learning
* Mode multijoueur en ligne
* Dashboard d’analyse des parties

---

## Conclusion

Ce projet démontre l’application concrète du Machine Learning dans un jeu simple, en combinant :

* Analyse de données
* Modélisation
* Intelligence artificielle

Il met également en évidence l’intérêt des approches hybrides pour améliorer les performances.

---

## Fin
