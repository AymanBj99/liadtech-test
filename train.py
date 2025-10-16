# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_loader import load_data
from encoder import TextEncoder
from attention import CrossAttention


class TripletTrainer:
    def __init__(self, embed_dim=768, margin=0.2, lr=2e-5, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = TextEncoder().to(self.device)
        self.model = CrossAttention(embed_dim).to(self.device)
        self.loss_fn = nn.TripletMarginLoss(margin=margin)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.model.parameters()),
            lr=lr
        )

    def compute_score(self, query_emb, doc_emb):
        """Calcule le score de similarité entre une requête et un document."""
        score, _ = self.model(query_emb, doc_emb)
        return score

    def train_one_epoch(self, pairs, docs_dict, batch_size=2):
        """Boucle d'entraînement sur un epoch."""
        total_loss = 0.0
        self.encoder.train()
        self.model.train()

        for i in tqdm(range(0, len(pairs), batch_size), desc="Training"):
            batch = pairs[i:i+batch_size]
            if not batch:
                continue

            queries = [p["query"] for p in batch]
            pos_docs = [docs_dict[p["pos_id"]] for p in batch]
            neg_docs = [docs_dict[p["neg_ids"][0]] for p in batch]  # un négatif par requête

            # Encode
            q_emb = self.encoder.encode(queries).to(self.device)
            pos_emb = self.encoder.encode(pos_docs).to(self.device)
            neg_emb = self.encoder.encode(neg_docs).to(self.device)

            # Forward
            pos_score, _ = self.model(q_emb, pos_emb.unsqueeze(1))
            neg_score, _ = self.model(q_emb, neg_emb.unsqueeze(1))

            # Calcul de la perte
            loss = self.loss_fn(pos_score, neg_score, torch.zeros_like(pos_score))
            total_loss += loss.item()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        avg_loss = total_loss / len(pairs)
        print(f"✅ Epoch terminée — Loss moyenne : {avg_loss:.4f}")
        return avg_loss


if __name__ == "__main__":
    # 1️⃣ Charger les données
    docs_dict, train_pairs, val_pairs = load_data()

    # 2️⃣ Initialiser le trainer
    trainer = TripletTrainer()

    # 3️⃣ Entraîner pour quelques epochs
    for epoch in range(2):
        print(f"\n🚀 Epoch {epoch + 1}")
        trainer.train_one_epoch(train_pairs, docs_dict)
