#j'ai ajouté ce fichier pour générer les paires 

import json
import random

docs_path = "data/docs.json"
pairs_path = "data/pairs.json"

# charger les documents 
with open(docs_path, "r", encoding="utf-8") as f:
    docs = json.load(f)


n_pairs = 150  
random.seed(42)

pairs = []

for _ in range(n_pairs):
    pos_doc = random.choice(docs)
    pos_id = pos_doc["id"]
    title = pos_doc["title"]

    
    query_words = title.lower().split()[:4]  
    query = " ".join(query_words)

    
    neg_docs = random.sample([d for d in docs if d["id"] != pos_id], 2)
    neg_ids = [neg_docs[0]["id"], neg_docs[1]["id"]]

    pairs.append({
        "query": query,
        "pos_id": pos_id,
        "neg_ids": neg_ids
    })


with open(pairs_path, "w", encoding="utf-8") as f:
    json.dump(pairs, f, ensure_ascii=False, indent=2)

print(f"✅ {len(pairs)} paires créées et sauvegardées dans {pairs_path}")
