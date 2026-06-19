import shutil
import os
import qtawesome as qta

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QListWidget, QLabel,
    QLineEdit, QSplitter, QFileDialog, QMessageBox,
    QApplication
)

from PySide6.QtCore import Qt, QThread, QObject, Signal, QTimer
from PySide6.QtGui import QTextCursor, QFont, QIcon
from networkx.algorithms.bipartite.basic import color

from llm.ollama_client import OllamaClient
from agent.core import Agent
from rag.retriever import Retriever
from ui.styles import DARK_STYLE
import indexing.build_index as bi
from indexing.incremental_index import add_document_to_index
from indexing.build_index import DOCUMENTS_PATH
from indexing.document_manager import delete_document

from helpers.doc_inspector import DocInspector

# =====================================
# WORKER
# =====================================

class AgentWorker(QObject):

    token = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, agent, query):
        super().__init__()
        self.agent = agent
        self.query = query

    def run(self):
        try:
            result = self.agent.run_stream(
                self.query,
                self.token.emit
            )
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


# =====================================
# MAIN WINDOW
# =====================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("DocAgent - Dokumenten-Verwaltung")
        self.resize(1500, 950)

        self.setWindowIcon(qta.icon('mdi.file-search-outline', color='black'))

        self.setStyleSheet(DARK_STYLE)

        # CORE
        self.agent = Agent(
            llm=OllamaClient(),
            retriever=Retriever()
        )

        # STATE
        self.last_results = []
        self.thinking_dots = 0
        self.thinking_active = False

        # UI
        self.build_ui()
        self.setup_thinking_animation()

        self.refresh_document_view()

        QTimer.singleShot(0, self.input.setFocus)
        QTimer.singleShot(0, self.input.selectAll)


    # =====================================
    # UI
    # =====================================

    def build_ui(self):

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout()
        central.setLayout(layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setSizes([250, 900, 350])
        layout.addWidget(splitter)

        # LEFT
        left = QWidget()
        left_l = QVBoxLayout()
        left.setLayout(left_l)

        left_l.addWidget(QLabel("📁 System"))

        self.add_doc_btn = QPushButton("📂 Dokument hinzufügen")
        self.add_doc_btn.clicked.connect(self.add_document)

        left_l.addWidget(self.add_doc_btn)

        self.reindex_btn = QPushButton("🔄 Index neu erstellen")
        self.reindex_btn.clicked.connect(self.on_index_new_click)

        left_l.addWidget(self.reindex_btn)

        #Neu -Delete Button
        self.delete_button = QPushButton("🗑 Dokument löschen")
        self.delete_button.clicked.connect(
            self.on_delete_click
        )
        left_l.addWidget(self.delete_button)

        # Neu Datei-Liste
        self.doc_list = QListWidget()
        self.doc_list.itemClicked.connect(self.update_item_font)
        self.doc_list.mousePressEvent = self.custom_mouse_press
        left_l.addWidget(self.doc_list)
        # =======================================================

        self.status = QTextEdit()
        self.status.setMaximumHeight(120)
        self.status.setReadOnly(True)

        self.set_status_text("DocAgent ready")
        left_l.addWidget(self.status)

        splitter.addWidget(left)

        # CENTER
        center = QWidget()
        c_l = QVBoxLayout()
        center.setLayout(c_l)

        c_l.addWidget(QLabel("💬 Chat"))

        self.chat_output = QTextEdit()
        self.chat_output.setReadOnly(True)
        c_l.addWidget(self.chat_output)

        input_row = QHBoxLayout()

        self.input = QLineEdit()
        self.input.setPlaceholderText("Frage eingeben...")
        self.input.returnPressed.connect(self.send_message)

        self.btn = QPushButton("Senden")
        self.btn.clicked.connect(self.send_message)

        input_row.addWidget(self.input)
        input_row.addWidget(self.btn)

        c_l.addLayout(input_row)

        splitter.addWidget(center)

        # RIGHT
        right = QWidget()
        r_l = QVBoxLayout()
        right.setLayout(r_l)

        r_l.addWidget(QLabel("📚 Quellen"))

        self.sources_list = QListWidget()
        self.sources_list.setWordWrap(True)
        self.sources_list.setSpacing(8)
        self.sources_list.itemClicked.connect(self.on_source_clicked)
        r_l.addWidget(self.sources_list)

        self.source_preview = QTextEdit()
        self.source_preview.setReadOnly(True)
        r_l.addWidget(self.source_preview)

        splitter.addWidget(right)

    # =====================================
    # THINKING ANIMATION (ROBUST)
    # =====================================

    def setup_thinking_animation(self):

        self.thinking_timer = QTimer()
        self.thinking_timer.timeout.connect(self.update_thinking)
        self.thinking_dots = 0

    def set_thinking(self, active: bool, text="🧠 Agent denkt"):

        self.thinking_active = active

        if active:
            self.thinking_dots = 0

            self.set_status_text(text)
            self.status.repaint()

            if not self.thinking_timer.isActive():
                self.thinking_timer.start(350)

        else:
            self.thinking_timer.stop()
            self.set_status_text("DocAgent ready")
            self.status.repaint()

    def update_thinking(self, text="🧠 Agent denkt"):

        if not self.thinking_active:
            return

        self.thinking_dots = (self.thinking_dots + 1) % 4
        dots = "." * self.thinking_dots

        text += dots
        self.set_status_text(text)
        self.status.repaint()

    # =====================================
    # SEND
    # =====================================

    def send_message(self):

        self.sources_list.clear()
        self.source_preview.clear()
        self.last_results = []

        query = self.input.text().strip()
        if not query:
            return

        test_query = query
        test_query = test_query.lower()
        if test_query == "exit":
            QApplication.quit()



        self.chat_output.append(f"\n🧑 USER:\n{query}\n")
        self.chat_output.append("🤖 DocAgent:\n")

        self.input.clear()

        # FOCUS FIX
        QTimer.singleShot(0, self.input.setFocus)

        # THINKING START
        self.set_thinking(True)

        # THREAD
        self.thread = QThread()
        self.worker = AgentWorker(self.agent, query)

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        self.worker.token.connect(self.on_token)
        self.worker.finished.connect(self.on_result)
        self.worker.error.connect(self.on_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)

        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    # =====================================
    # STREAMING
    # =====================================

    def on_token(self, token: str):

        cursor = self.chat_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(token)
        self.chat_output.setTextCursor(cursor)
        self.chat_output.ensureCursorVisible()

    # =====================================
    # RESULT
    # =====================================

    def on_result(self, result):

        self.set_thinking(False)

        self.last_results = []

        if isinstance(result, dict):
            self.last_results = result.get("sources") or []

        self.sources_list.clear()

        self.last_results.sort(
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        for s in self.last_results:

            if isinstance(s, dict):

                source = s.get("source", "unknown")
                score = s.get("score", 0.0)

                content = s.get("content", "")

                preview = content.replace("\n", " ")
                preview = " ".join(preview.split())

                if len(preview) > 180:
                    preview = preview[:180] + "..."

                text = (
                    f"📄 {source}\n"
                    f"Score: {score:.2f}\n\n"
                    f"{preview}"
                )

            else:

                text = str(s)

            self.sources_list.addItem(text)

        self.chat_output.append("------------------------")

    # =====================================
    # ERROR
    # =====================================

    def on_error(self, msg):

        self.set_thinking(False)
        self.chat_output.append(f"\n❌ ERROR: {msg}\n")

    # =====================================
    # SOURCE CLICK
    # =====================================

    def on_source_clicked(self, item):

        idx = self.sources_list.row(item)

        if idx >= len(self.last_results):
            return

        s = self.last_results[idx]

        self.source_preview.setPlainText(
            f"📄 {s.get('source','unknown')}\n\n"
            f"⭐ Score: {s.get('score',0.0)}\n\n"
            f"{s.get('content','')}"
        )

    # =====================================
    # ADD DOCUMENT
    # =====================================

    def add_document(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Dokument auswählen",
            "",
            "Dokumente (*.pdf *.txt *.docx *.odt)"
        )

        if not file_path:
            return

        try:
            import os
            filename = os.path.basename(file_path)

            target_path = os.path.join(
                "data/docs",
                filename
            )


            if not os.path.exists(DOCUMENTS_PATH):

                self.set_status_text("🔴 Es ist ein interner Fehler aufgetreten!"
                                    "<br>Bitte führen Sie die Indexierung neu durch!")

                return

            docInspect = DocInspector()
            result_file_check = docInspect.check_add_file(file_path)


            if result_file_check == 0:
                # Dokument hinzufügen

                # Datei kopieren
                shutil.copy(file_path, target_path)

                self.status.setPlainText(
                    f"Status:\n📂 Kopiert: {filename}"
                )
                self.status.repaint()

                # Index neu bauen
                self.rebuild_index(1, target_path)

            elif result_file_check == 1:
                self.status.setPlainText(
                    f"Status:\n⚠ Das Dokument ({filename}) existiert bereits! "
                )
                self.status.repaint()
                return
            else:
                self.status.setPlainText(
                    f"Status:\n⚠ Das Dokument ({filename}) existiert bereits,\n"
                    f"wurde aber extern verändert!\nZurzeit keine Aktion möglich! "
                )
                self.status.repaint()
                return


        except Exception as e:

            self.on_error(str(e))

    # =====================================
    # REBUILD INDEX
    # =====================================

    def rebuild_index(self, t=1, t_path=""):
        target_path = t_path

        try:
            if t == 1:
                text = "Dokument hinzufügen: 🔄 Erstelle neuen Index..."
            elif t == 2:
                text = "🔄 Erstelle neuen Index..."
            else:
                text = "Dokument löschen: 🔄 Erstelle neuen Index..."

            self.set_status_text(text)


            # BUILD

            if t == 1:                  # Dokument hinzufügen
                # Neue Variante Incremental Indexing
                act = add_document_to_index(target_path)

            else:                       # Dokument löschen
                bi.build_index()

            # RETRIEVER RELOAD
            self.agent.retriever = Retriever()

            if t==1:
                text = "Dokument erfolgreich hinzugefügt<br>✅ Index aktualisiert"
            elif t==2:
                text = "✅ Index wurde neu erstellt"
            else:
                text = "Dokument wir jetzt gelöscht...<br>✅ Index aktualisiert"

            self.set_status_text(text)


        except Exception as e:

            self.on_error(str(e))


        self.refresh_document_view()
    #=====================================================

    def refresh_document_view(self):

        self.doc_list.clear()

        stats = self.agent.retriever.get_document_stats()

        for name, count in sorted(stats.items(), key=lambda x: x[0].lower()):
            self.doc_list.addItem(
                f"{name} ({count} chunks)"
            )
    #=====================================================
    #=====================================================

    def on_index_new_click(self):

        msg_box = QMessageBox()
        msg_box.setWindowTitle("Index neu erstellen...")
        msg_box.setText("Wollen Sie den Index wirklich neu erstellen?\n"
                        "Dieser Vorgang kann einige Zeit in Anspruch nehmen...")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)

        antwort = msg_box.exec()

        if antwort == QMessageBox.StandardButton.Yes:
            self.rebuild_index(t=2)
        else:
            text = "Aktion wurde abgebrochen!"
            self.set_status_text(text)
    #===========================================================

    def on_delete_click(self):

        if not self.doc_list.selectedItems():
            QMessageBox.information(None, "Information", "Es wurde keine Datei zum Löschen ausgewählt...")
        else:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Datei löschen...")
            msg_box.setText("Möchten Sie die Datei wirklich löschen?")
            msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msg_box.setDefaultButton(QMessageBox.StandardButton.No)

            antwort = msg_box.exec()

            if antwort == QMessageBox.StandardButton.Yes:
                # Status Ausgabe
                text = "Starte Löschen..."
                self.set_status_text(text)

                self.delete_doc()
            else:
                text = "Aktion wurde abgebrochen!"
                self.set_status_text(text)
    #==============================================================

    def delete_doc(self):

        item = self.doc_list.currentItem()

        if not item:
            return

        filename = item.text().split(" (")[0]

        #=================================================
        # Neuerung: Inkrementelles Löschen
        delete_document(filename)

        # Retriever neu laden
        self.agent.retriever = Retriever()

        # Liste aktualisieren
        self.refresh_document_view()

        # Status Ausgabe
        text = "🗑 Gelöscht:  " + filename
        self.set_status_text(text)


    #==================================================================
    # Listenelemente Highlighting

    def update_item_font(self, current):
        # 1. Alle Elemente in der Liste standardmäßig zurücksetzen
        for i in range(self.doc_list.count()):
            item = self.doc_list.item(i)
            # Wir überspringen das aktuell geklickte Element beim Zurücksetzen,
            # damit wir seinen Zustand gleich gezielt umkehren können.
            if item == current:
                continue
            font = item.font()
            font.setBold(False)
            font.setItalic(False)
            item.setFont(font)

        # 2. Zustand des ausgewählten Elements umschalten (Toggle)
        if current:
            font = current.font()
            # Wenn es schon fett ist, wird es normal. Sonst wird es fett/kursiv.
            is_already_styled = font.bold()

            font.setBold(not is_already_styled)
            font.setItalic(not is_already_styled)
            current.setFont(font)

            # 3. Auswahl in der GUI aufheben, wenn das Format zurückgesetzt wurde
            if is_already_styled:
                self.doc_list.setCurrentItem(None)

    # Klick in leere Liste
    def custom_mouse_press(self, event):

        # Nachschauen, ob an der geklickten Stelle ein Element ist
        item = self.doc_list.itemAt(event.position().toPoint())

        if item is None:
            self.doc_list.clearSelection()
            self.doc_list.setCurrentItem(None)
            self.window().reset_all_item_fonts()
        else:
            QListWidget.mousePressEvent(self.doc_list, event)

    def reset_all_item_fonts(self):
        # Setzt alle Schriften in der Liste wieder auf den Standardzustand zurück
        for i in range(self.doc_list.count()):
            idx_item = self.doc_list.item(i)
            idx_font = idx_item.font()
            idx_font.setBold(False)
            idx_font.setItalic(False)
            idx_item.setFont(idx_font)

    # Status Anzeigen
    def set_status_text(self, text):
        #self.status.setPlainText(text)
        self.status.setHtml(f"🔵 <b>Status</b><br>{text}")
        self.status.repaint()

    # ======================================================================
    # ======================================================================

    def create_source_preview(self, text, length=180):

        text = text.replace("\n", " ")
        text = " ".join(text.split())

        if len(text) > length:
            text = text[:length] + "..."

        return text
























