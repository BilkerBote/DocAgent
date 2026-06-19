import os
import pickle
import numpy as np
import faiss
import torch

import hashlib

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document
from odf import text, teletype
from odf.opendocument import load


# =====================================
# PROJECT ROOT (WICHTIG!)
# =====================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "docs")
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "faiss.index")
DOCUMENTS_PATH = os.path.join(PROJECT_ROOT, "data", "documents.pkl")


# =====================================
# DEVICE
# =====================================

device = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================
# LOADERS
# =====================================

def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )


def load_docx(file_path):
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_odt(file_path):
    odt = load(file_path)
    paragraphs = odt.getElementsByType(text.P)
    return "\n".join(teletype.extractText(p) for p in paragraphs)


def load_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return load_txt(file_path)
    elif ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext == ".odt":
        return load_odt(file_path)
    # elif ext == ".doc":
    #     return load_doc(file_path)
    else:
        return None

# =====================================
# FILE HASH
# =====================================

def file_hash(file_path):

    md5 = hashlib.md5()

    with open(file_path, "rb") as f:

        while chunk := f.read(8192):
            md5.update(chunk)

    return md5.hexdigest()

# =====================================
# CHUNKING
# =====================================

def chunk_text(text, max_length=500):
    paragraphs = text.split("\n")

    chunks = []
    current = ""

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        if len(current) + len(p) < max_length:
            current += "\n" + p
        else:
            if current:
                chunks.append(current.strip())
            current = p

    if current:
        chunks.append(current.strip())

    return chunks


# =====================================
# INDEX BUILD
# =====================================

def build_index():

    print("\n🔍 Starte Index-Build...")

    print("🧭 PROJECT_ROOT:", PROJECT_ROOT)
    print("📁 DATA_PATH:", DATA_PATH)
    print("📁 EXISTS:", os.path.exists(DATA_PATH))

    documents = []

    # ---------------------------------
    # FILE COLLECTION
    # ---------------------------------

    files = []

    for root, _, filenames in os.walk(DATA_PATH):
        print("➡️ ROOT:", root)

        for filename in filenames:
            path = os.path.join(root, filename)
            files.append(path)
            print("📄 gefunden:", path)

    print(f"\n📂 Dateien gefunden: {len(files)}")

    # ---------------------------------
    # LOAD + CHUNK
    # ---------------------------------

    for file_path in files:

        try:
            print(f"\n📄 Verarbeite: {file_path}")

            content = load_file(file_path)

            if not content:
                print("⚠️ Kein Inhalt")
                continue

            chunks = chunk_text(content)

            print(f"✂️ Chunks: {len(chunks)}")
            hash_value = file_hash(file_path)

            for c in chunks:
                documents.append({
                    "content": c,
                    "source": os.path.basename(file_path),
                    "file_hash": hash_value
                })

        except Exception as e:
            print("⚠️ Fehler:", file_path, e)

    if not documents:
        print("❌ Keine Dokumente zum Indexieren")
        return

    print(f"\n📄 Gesamt-Chunks: {len(documents)}")

    # ---------------------------------
    # EMBEDDING MODEL
    # ---------------------------------

    print("\n🧠 Lade Embedding-Modell...")

    model = SentenceTransformer(
        "intfloat/multilingual-e5-base",
        device=device
    )

    texts = [d["content"] for d in documents]

    print("🔄 Erzeuge Embeddings...")

    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    faiss.normalize_L2(embeddings)

    # ---------------------------------
    # FAISS INDEX
    # ---------------------------------

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"✅ FAISS Index: {index.ntotal} Vektoren")

    # ---------------------------------
    # SAVE
    # ---------------------------------

    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(DOCUMENTS_PATH, "wb") as f:
        pickle.dump(documents, f)

    print("\n💾 Gespeichert:")
    print("📁 FAISS:", FAISS_INDEX_PATH)
    print("📁 DOCS:", DOCUMENTS_PATH)
    print("\n✅ Fertig!")