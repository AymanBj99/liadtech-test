import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from data_loader import load_data
from encoder import TextEncoder
from attention import CrossAttention


def compute_mrr(model, encoder, docs_dict, val_pairs, device):
    """Calcul du Mean Reciprocal Rank (MRR)"""
    model.eval()
    encoder.eval()

    ranks = []

    for pair in val_pairs:
        query = pair["query"]
        pos_id = pair["pos_id"]

        # Embedding de la requête
        q_emb = encoder.encode([query]).to(device)

        # Calculer les scores pour tous les documents
        all_docs = list(docs_dict.values())
        all_embs = encoder.encode(all_docs).to(device)

        # Scores attention
        scores = []
        for d_emb in all_embs:
            d_emb = d_emb.unsqueeze(0).unsqueeze(1)
            q_tmp = q_emb.clone()
            score, _ = model(q_tmp, d_emb)
            scores.append(score.item())

        # Trier les documents selon le score décroissant
        sorted_indices = np.argsort(scores)[::-1]
        ranked_doc_ids = [list(docs_dict.keys())[i] for i in sorted_indices]

        # Trouver le document positif
        rank = ranked_doc_ids.index(pos_id) + 1
        ranks.append(1.0 / rank)

    return np.mean(ranks)


def compute_recall_at_k(model, encoder, docs_dict, val_pairs, device, k=3):
    """Calcul du Recall@K"""
    model.eval()
    encoder.eval()
    correct = 0

    for pair in val_pairs:
        query = pair["query"]
        pos_id = pair["pos_id"]

        q_emb = encoder.encode([query]).to(device)
        all_docs = list(docs_dict.values())
        all_embs = encoder.encode(all_docs).to(device)

        scores = []
        for d_emb in all_embs:
            d_emb = d_emb.unsqueeze(0).unsqueeze(1)
            q_tmp = q_emb.clone()
            score, _ = model(q_tmp, d_emb)
            scores.append(score.item())

        sorted_indices = np.argsort(scores)[::-1]
        top_k_ids = [list(docs_dict.keys())[i] for i in sorted_indices[:k]]

        if pos_id in top_k_ids:
            correct += 1

    return correct / len(val_pairs)


def tfidf_baseline(docs_dict, val_pairs):
    vectorizer = TfidfVectorizer()
    all_docs = list(docs_dict.values())
    X = vectorizer.fit_transform(all_docs)
    mrr_scores = []

    for pair in val_pairs:
        query = pair["query"]
        pos_id = pair["pos_id"]

        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, X).flatten()
        sorted_indices = np.argsort(sims)[::-1]
        ranked_doc_ids = [list(docs_dict.keys())[i] for i in sorted_indices]
        rank = ranked_doc_ids.index(pos_id) + 1
        mrr_scores.append(1.0 / rank)

    return np.mean(mrr_scores)


def visualize_attention(model, encoder, query, doc_text, device):
    import seaborn as sns

    q_emb = encoder.encode([query]).to(device)
    d_emb = encoder.encode([doc_text]).to(device).unsqueeze(1)

    _, attn_weights = model(q_emb, d_emb)
    attn = attn_weights.detach().cpu().numpy()

    plt.figure(figsize=(6, 1))
    sns.heatmap(attn, cmap="viridis", cbar=True)
    plt.title("Poids d'attention pour la requête")
    plt.show()


if __name__ == "__main__":
    docs_dict, train_pairs, val_pairs = load_data()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = TextEncoder().to(device)
    model = CrossAttention(embed_dim=128).to(device)

    # Calculer des métriques
    mrr = compute_mrr(model, encoder, docs_dict, val_pairs, device)
    recall = compute_recall_at_k(model, encoder, docs_dict, val_pairs, device, k=3)
    baseline_mrr = tfidf_baseline(docs_dict, val_pairs)

    print("\n📊 Résultats d'évaluation :")
    print(f"MRR (modèle attention) : {mrr:.4f}")
    print(f"Recall@3 (modèle attention) : {recall:.4f}")
    print(f"MRR (baseline TF-IDF) : {baseline_mrr:.4f}")

    sample = val_pairs[0]
    query = sample["query"]
    doc_text = docs_dict[sample["pos_id"]]
    visualize_attention(model, encoder, query, doc_text, device)
