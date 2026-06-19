from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

from indexing.build_index import build_index

import sys, os

if (
    not os.path.exists("data/faiss.index")
    or
    not os.path.exists("data/documents.pkl")
):
    print("Erstelle Initial-Index...")
    build_index()

app = QApplication(sys.argv)

window = MainWindow()
window.show()

sys.exit(app.exec())
