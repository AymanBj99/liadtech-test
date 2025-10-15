import json

with open("data/docs.json", "r", encoding="utf-8") as f:
    docs = json.load(f)
    print(f"Nombre de documents : {len(docs)}")
    print(docs[0])
