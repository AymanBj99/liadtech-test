#ce fichier est utilisé pour charger les données et les paires c pour ça je l'ai ajouter dans le projet

import json
import random
from sklearn.model_selection import train_test_split

def load_data(docs_path="data/docs.json", pairs_path="data/pairs.json"):
    # Charger les documents
    with open(docs_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    # Charger les paires
    with open(pairs_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    # Créer un dictionnaire {id: texte complet}
    docs_dict = {d["id"]: d["title"] + " " + d["abstract"] for d in docs}

    # Séparer train et validation (80/20)
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=42)

    print(f"Documents chargés : {len(docs)}")
    print(f"Paires d'entraînement : {len(train_pairs)}")
    print(f"Paires de validation : {len(val_pairs)}")

    return docs_dict, train_pairs, val_pairs


# Exécuter la fonction
if __name__ == "__main__":
    docs_dict, train_pairs, val_pairs = load_data()
    # Afficher un exemple
    print("Exemple de document :", list(docs_dict.items())[0])
    print("Exemple de paire d'entraînement :", train_pairs[0])
