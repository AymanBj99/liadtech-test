# encoder.py
from transformers import AutoTokenizer, AutoModel
import torch

class TextEncoder:
    def __init__(self, model_name="bert-base-uncased", device=None):
        # Choisir le GPU si disponible
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🔹 Chargement du modèle {model_name} sur {self.device}...")

        # Charger le tokenizer et le modèle BERT
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # mode évaluation

    def encode(self, texts, max_length=128, batch_size=8):
        """
        Encode une liste de textes en embeddings BERT
        Retourne un tenseur (nb_textes, 768)
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # Tokenisation
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)

            # Passage dans BERT
            with torch.no_grad():
                outputs = self.model(**inputs)
                # On prend les embeddings [CLS] comme représentation globale
                embeddings = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, 768)

            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)



if __name__ == "__main__":
    from train import load_data

    docs_dict, train_pairs, val_pairs = load_data()
    doc_texts = list(docs_dict.values())

    encoder = TextEncoder()
    doc_embeddings = encoder.encode(doc_texts)

    queries = [p["query"] for p in train_pairs]
    query_embeddings = encoder.encode(queries)
    print("Taille des embeddings (queries) :", query_embeddings.shape)

    print("✅ Encodage terminé !")
    print("Taille des embeddings :", doc_embeddings.shape) 
