# LIADTECH – Test Technique Développement AI

## 🧠 Objectif du projet
Ce projet implémente un système de recherche sémantique basé sur un **mécanisme d’attention personnalisé**.  
L’objectif est de relier des **requêtes** à des **documents scientifiques** (résumés de recherche) en apprenant leurs similarités à l’aide d’un encodeur BERT.

Le pipeline complet couvre :
1. Chargement et préparation des données (`docs.json`, `pairs.json`)
2. Encodage des textes (requêtes / documents)
3. Entraînement du modèle via **Triplet Loss**
4. Évaluation (MRR, Recall@3, comparaison TF-IDF)
5. Visualisation de la courbe de perte et des poids d’attention

---

## ⚙️ Installation & Configuration

### Cloner le dépôt
```bash
git clone https://github.com/AymanBj99/liadtech-test.git
cd liadtech-test


## ⚙️ Installation & Exécution

### 1. Création l’environnement
```bash
python -m venv env
env\Scripts\activate     


### 2. Installation des dépendances
Installer les dépendances
pip install -r requirements.txt

### 3.Lancement de pipeline complet
###Etape 1 - Encode
python encoder.py

###Étape 2 — Entraînement
python train.py
Entraîne le modèle sur 150 paires et génère training_loss_curve.png

###Étape 3 — Évaluation
python evaluate.py

Temps d’exécution (Laptop standard)

Environnement : CPU, 8 Go RAM

Documents : ~700

Paires d’entraînement : 150

Entraînement (5 époques) : ≈ 5min

Évaluation : ≈ 15min