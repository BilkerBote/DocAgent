import os
import pickle

from indexing.build_index import (
    DOCUMENTS_PATH,
    DATA_PATH
)

from indexing.rebuild_index import (
    rebuild_from_documents
)
#========================================================================
#========================================================================

def delete_document(filename):

    print(
        f"\n🗑 Lösche Dokument: "
        f"{filename}"
    )

    # ---------------------------------
    # documents.pkl laden
    # ---------------------------------

    with open(
        DOCUMENTS_PATH,
        "rb"
    ) as f:

        documents = pickle.load(f)

    old_count = len(documents)

    # ---------------------------------
    # Chunks entfernen
    # ---------------------------------

    documents = [

        doc

        for doc in documents

        if doc["source"] != filename
    ]

    removed = (
        old_count -
        len(documents)
    )

    print(
        f"🗑 Entfernte Chunks: "
        f"{removed}"
    )

    # ---------------------------------
    # speichern
    # ---------------------------------

    with open(
        DOCUMENTS_PATH,
        "wb"
    ) as f:

        pickle.dump(
            documents,
            f
        )

    # ---------------------------------
    # Originaldatei löschen
    # ---------------------------------

    file_path = os.path.join(
        DATA_PATH,
        filename
    )

    if os.path.exists(file_path):

        os.remove(file_path)

        print(
            "📄 Datei gelöscht"
        )

    # ---------------------------------
    # Index neu erzeugen
    # ---------------------------------

    return rebuild_from_documents()

#========================================================================
#========================================================================