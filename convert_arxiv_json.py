import json

input_path = "data/arxiv_data.json"     # ton fichier téléchargé depuis Kaggle
output_path = "data/docs.json"          # le fichier final compatible avec ton code

docs = []

with open(input_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        try:
            paper = json.loads(line)
            docs.append({
                "id": i + 1,
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", "")  # 'summary' → 'abstract'
            })
        except json.JSONDecodeError:
            continue  # ignore les lignes corrompues

        if i >= 700:  # garde les 1000 premiers pour être léger
            break

# Sauvegarde au format JSON standard
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

print(f"✅ {len(docs)} documents convertis et sauvegardés dans {output_path}")
