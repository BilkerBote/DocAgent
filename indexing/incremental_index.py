import os
import pickle
import numpy as np
import faiss
import torch

import hashlib

from sentence_transformers import SentenceTransformer

from indexing.build_index import (
    load_file,
    chunk_text,
    FAISS_INDEX_PATH,
    DOCUMENTS_PATH
)

# =====================================
# DEVICE
# =====================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================
# Hash-Wert ermitteln
# ======================================
def file_hash(file_path):

    md5 = hashlib.md5()

    with open(file_path, "rb") as f:

        while chunk := f.read(8192):
            md5.update(chunk)

    return md5.hexdigest()


# =====================================
# DATEI BEREITS INDEXIERT?
# =====================================

def hash_exists(documents, hash_value):

    for doc in documents:

        if doc.get("file_hash") == hash_value:
            return True

    return False


# =====================================
# DOKUMENT HINZUFÜGEN
# =====================================

def add_document_to_index(file_path):

    print("\n📄 Incremental Indexing")

    filename = os.path.basename(file_path)

    # ---------------------------------
    # Dokumente laden
    # ---------------------------------
    with open(DOCUMENTS_PATH, "rb") as f:
        documents = pickle.load(f)

    hash_value = file_hash(file_path)
    print("🔑 HASH:", hash_value)


    # ---------------------------------
    # Datei lesen
    # ---------------------------------

    print(f"📖 Lade: {filename}")

    content = load_file(file_path)

    if not content:

        print("❌ Kein Inhalt gefunden")
        return False

    chunks = chunk_text(content)

    if not chunks:

        print("❌ Keine Chunks erzeugt")
        return False

    print(f"✂️ Chunks: {len(chunks)}")

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
        chunks,
        show_progress_bar=False
    )

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    faiss.normalize_L2(embeddings)

    # ---------------------------------
    # Index laden
    # ---------------------------------

    print("📂 Lade FAISS Index...")

    index = faiss.read_index(
        FAISS_INDEX_PATH
    )

    old_count = index.ntotal

    # ---------------------------------
    # Neue Vektoren hinzufügen
    # ---------------------------------

    index.add(embeddings)

    print(
        f"➕ {index.ntotal - old_count} "
        f"Vektoren hinzugefügt"
    )

    # ---------------------------------
    # Dokumente erweitern
    # ---------------------------------

    for chunk in chunks:
        documents.append({
            "content": chunk,
            "source": filename,
            "file_hash": hash_value
        })

    # ---------------------------------
    # Speichern
    # ---------------------------------

    faiss.write_index(
        index,
        FAISS_INDEX_PATH
    )

    with open(DOCUMENTS_PATH, "wb") as f:

        pickle.dump(documents, f)

    print(
        f"✅ Dokument hinzugefügt: "
        f"{filename}"
    )

    print(
        f"📊 Index jetzt: "
        f"{index.ntotal} Vektoren"
    )

    return True


