import pickle
import numpy as np
import faiss
import torch

from sentence_transformers import SentenceTransformer

from indexing.build_index import (
    DOCUMENTS_PATH,
    FAISS_INDEX_PATH
)

# =====================================
# DEVICE
# =====================================

device = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================
# REBUILD FROM DOCUMENTS
# =====================================

def rebuild_from_documents():

    print("\n🔄 Rebuild aus documents.pkl")

    # ---------------------------------
    # Dokumente laden
    # ---------------------------------

    try:

        with open(DOCUMENTS_PATH, "rb") as f:
            documents = pickle.load(f)

    except Exception as e:

        print(
            f"❌ Fehler beim Laden "
            f"der Dokumente: {e}"
        )

        return False

    # ---------------------------------
    # Leer?
    # ---------------------------------

    if not documents:

        print("⚠ Keine Dokumente vorhanden")

        return False

    print(
        f"📄 Chunks geladen: "
        f"{len(documents)}"
    )

    # ---------------------------------
    # Texte extrahieren
    # ---------------------------------

    texts = [
        doc["content"]
        for doc in documents
    ]

    # ---------------------------------
    # Modell laden
    # ---------------------------------

    print("🧠 Lade Embedding-Modell...")

    model = SentenceTransformer(
        "intfloat/multilingual-e5-base",
        device=device
    )

    # ---------------------------------
    # Embeddings erzeugen
    # ---------------------------------

    print("🔄 Erzeuge Embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    faiss.normalize_L2(
        embeddings
    )

    # ---------------------------------
    # FAISS neu erzeugen
    # ---------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    # ---------------------------------
    # Speichern
    # ---------------------------------

    faiss.write_index(
        index,
        FAISS_INDEX_PATH
    )

    print(
        f"✅ Index gespeichert "
        f"({index.ntotal} Vektoren)"
    )

    return True