import pickle
import numpy as np
import faiss
import torch

from sentence_transformers import SentenceTransformer, CrossEncoder


# ---------------------------------
# Device Setup
# ---------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


class Retriever:

    # =====================================
    # INIT
    # =====================================

    def __init__(self):

        print("🔄 Lade Dokumente...")

        with open("data/documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

        print(f"📄 Dokumente geladen: {len(self.documents)}")

        # ---------------------------------
        # FAISS INDEX
        # ---------------------------------

        print("🔄 Lade FAISS Index...")

        self.index = faiss.read_index("data/faiss.index")

        print(f"🧠 Vektoren im Index: {self.index.ntotal}")

        # ---------------------------------
        # EMBEDDING MODEL
        # ---------------------------------

        print("🔄 Lade Embedding-Modell...")

        self.model = SentenceTransformer(
            "intfloat/multilingual-e5-base",
            device=device
        )

        # ---------------------------------
        # CROSS ENCODER (Re-Ranker)
        # ---------------------------------

        print("🔄 Lade Re-Ranker...")

        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=device
        )

        print("✅ Retriever bereit auf:", device)

    # =====================================
    # SEARCH
    # =====================================

    def search(self, query: str, top_k: int = 5):

        print("\n🔍 SEARCH:", query)

        # ---------------------------------
        # 1. EMBEDDING
        # ---------------------------------

        query_embedding = self.model.encode([query])[0]
        query_embedding = np.array([query_embedding], dtype=np.float32)

        faiss.normalize_L2(query_embedding)

        # ---------------------------------
        # 2. FAISS RETRIEVAL (CANDIDATES)
        # ---------------------------------

        scores, indices = self.index.search(query_embedding, top_k * 10)

        candidates = []

        query_lower = query.lower()
        query_words = query_lower.split()

        # ---------------------------------
        # 3. BUILD CANDIDATES
        # ---------------------------------

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            doc = self.documents[idx]
            text = doc["content"]

            candidates.append({
                "index": idx,
                "content": text,
                "source": doc["source"],
                "faiss_score": float(score)
            })

        # ---------------------------------
        # 4. CROSS ENCODER RERANKING
        # ---------------------------------

        pairs = [
            (query, c["content"])
            for c in candidates
        ]

        if pairs:

            rerank_scores = self.reranker.predict(pairs)

            for i, score in enumerate(rerank_scores):
                candidates[i]["rerank_score"] = float(score)

        else:
            return []

        # ---------------------------------
        # 5. FINAL SCORE (clean weighting)
        # ---------------------------------

        results = []

        for c in candidates:

            text_lower = c["content"].lower()

            # keyword boost (light, safe)
            keyword_boost = 0.0

            for word in query_words:
                if len(word) < 3:
                    continue
                if word in text_lower:
                    keyword_boost += 0.05

            if query_lower in text_lower:
                keyword_boost += 0.3

            final_score = (
                0.5 * c["rerank_score"] +
                0.3 * c["faiss_score"] +
                0.2 * keyword_boost
            )

            results.append({
                "content": c["content"],
                "source": c["source"],
                "score": float(final_score)
            })

        # ---------------------------------
        # 6. SORT
        # ---------------------------------

        results.sort(key=lambda x: x["score"], reverse=True)

        # ---------------------------------
        # DEBUG OUTPUT
        # ---------------------------------

        for r in results[:top_k]:
            print(f"{r['source']} | score={r['score']:.3f}")

        return results[:top_k]

    #===============================================
    # Dokumente auflisten
    #===============================================
    def get_document_stats(self):

        stats = {}

        for doc in self.documents:
            src = doc.get("source", "unknown")
            stats[src] = stats.get(src, 0) + 1

        return stats