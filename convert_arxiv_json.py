#j'ai ajouté ce fichier pour convertir le fichier json en un fichier json compatible avec le code 
# car la format du fichier json n'était pas compatible avec le code 

import json

input_path = "data/arxiv_data.json"     
output_path = "data/docs.json"          

docs = []

with open(input_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        try:
            paper = json.loads(line)
            docs.append({
                "id": i + 1,
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", "")  
            })
        except json.JSONDecodeError:
            continue  

        if i >= 700:  # j'ai gardé les 700 premiers pour être léger
            break

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

print(f"✅ {len(docs)} documents convertis et sauvegardés dans {output_path}")
