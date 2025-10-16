# attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """
    Implémentation simplifiée du mécanisme d'attention croisée
    entre la requête (Q) et un document (D).
    """

    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        # Poids d'attention à apprendre
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wd = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, D):
        """
        Q : (batch_size, embed_dim)
        D : (batch_size, seq_len, embed_dim)
        Retourne un score de pertinence entre Q et D.
        """
        # 1️⃣ Projections
        Q_proj = self.Wq(Q).unsqueeze(1)          # (batch, 1, embed_dim)
        D_proj = self.Wd(D)                       # (batch, seq_len, embed_dim)

        # 2️⃣ Calcul des poids d’attention
        # Matrice (1, seq_len)
        attn_scores = torch.bmm(Q_proj, D_proj.transpose(1, 2)) / (D_proj.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)            # Normalisation softmax

        # 3️⃣ Combinaison pondérée du document
        context = torch.bmm(attn_weights, D_proj)                # (batch, 1, embed_dim)

        # 4️⃣ Calcul du score final (produit scalaire entre Q et le contexte)
        score = torch.sum(Q_proj * context, dim=-1)              # (batch, 1)
        return score.squeeze(1), attn_weights.squeeze(1)


if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    embed_dim= 128

    # Fake data
    Q = torch.rand(batch_size, embed_dim)
    D = torch.rand(batch_size, seq_len, embed_dim)

    model = CrossAttention(embed_dim)
    score, weights = model(Q, D)

    print("Score shape:", score.shape)
    print("Attention shape:", weights.shape)
