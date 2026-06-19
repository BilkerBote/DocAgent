import os
import pickle
import hashlib

from indexing.build_index import PROJECT_ROOT, DATA_PATH, DOCUMENTS_PATH


class DocInspector:

    def __init__(self):
        self.doc_path_pkl = DOCUMENTS_PATH

        with open(self.doc_path_pkl, "rb") as f:
            self.documents = pickle.load(f)

    # ======================================
    # HASH
    # ======================================
    def file_hash(self, file_path):

        md5 = hashlib.md5()

        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                md5.update(chunk)

        return md5.hexdigest()

    # =====================================
    # FILE EXISTS
    # =====================================

    def __document_exists(self, filename):

        for doc in self.documents:

            if doc.get("source") == filename:
                return True

        return False
    # ======================================
    # HASH EXISTS
    # =====================================
    def __hash_exists(self, hash_value):

        for doc in self.documents:

            if doc.get("file_hash") == hash_value:
                return True

        return False


    def check_add_file(self, file_path):
        # file_path --> Dateipfad, der hinzu zufügenden Datei!
        # Rückgabe:
        # 0 --> Datei nicht vorhanden in docAgent
        # 1 --> Datei vorhanden in docAgent
        # 2 --> Datei wurde ausserhalb von docAgent geändert

        filename = os.path.basename(file_path)
        hash_value = self.file_hash(file_path)

        target_path = os.path.join(
            "data/docs",
            filename
        )

        if os.path.exists(target_path):

            if self.__hash_exists(hash_value):
                return 1
            else:
                return 2

        else:
            return 0