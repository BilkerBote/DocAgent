import os
import json
import re
import pickle

from indexing.build_index import DOCUMENTS_PATH

class Agent:

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    # =========================================================
    # LLM TOOL DECISION
    # =========================================================

    def decide(self, query: str, history: str):

        prompt = f"""
DU BIST EIN STRICT TOOL ROUTER.

DU DARFST NUR EIN JSON AUSGEBEN.

TOOLS:

1)
{{"tool": "search_docs", "input": "<string>"}}

2)
{{"tool": "list_files"}}

3)
{{"tool": "finish", "answer": "<string>"}}

REGELN:
- Inhaltliche Fragen → search_docs
- Dateiliste → list_files
- nie raten

FRAGE:
{query}

HISTORY:
{history}

ANTWORTE NUR JSON:
"""

        return self.llm.generate(prompt)

    # =========================================================
    # JSON PARSER
    # =========================================================

    def parse_tool_call(self, response: str):

        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                return {"tool": ""}

            data = json.loads(match.group(0))

            allowed = ["search_docs", "list_files", "finish"]

            if data.get("tool") not in allowed:
                return {"tool": ""}

            return data

        except Exception:
            return {"tool": ""}

    # =========================================================
    # LLM ANSWER WITH CONTEXT
    # =========================================================

    def answer_with_context(self, query, results, history=None):

        print("\n🧠 Baue Kontext für LLM...")

        context_blocks = []
        sources = []

        for r in results:

            if not isinstance(r, dict):
                continue

            context_blocks.append(
                f"[QUELLE: {r['source']}]\n{r['content']}"
            )

            sources.append(r["source"])

        context_text = "\n---\n".join(context_blocks)

        system_prompt = """
Du bist ein präziser Dokumenten-Assistent.

REGELN:
- nutze ausschließlich die Quellen
- keine Erfindungen
- wenn nichts passt: "Keine Information gefunden"
- nenne am Ende die Quellen
- Nenne ausschließlich Dateinamen als Quellen
- Erfinde niemals Webseiten oder externe Quellen
- Nutze nur die bereitgestellten QUELLE-Tags

BEI BRIEFEN:
- Verwechsle Absender und Empfänger nicht.
- Der obere Adressblock gehört häufig zum Absender.
- Der Empfänger steht häufig nach "An", "Empfänger" oder im Adressfeld des Briefes.
- Wenn die Zuordnung nicht eindeutig ist, sage dies ausdrücklich.

BEI RECHNUNGEN:
- Rechnungssteller und Rechnungsempfänger unterscheiden
- Betrag nicht mit Kontostand verwechseln

BEI VERTRÄGEN:
- Vertragsbeginn und Kündigungsfrist unterscheiden
"""

        user_prompt = f"""
FRAGE:
{query}

QUELLEN:
{context_text}

ANTWORT:
kurz, klar, mit Quellenangabe
"""

        if history:
            user_prompt = f"CHATVERLAUF:\n{history}\n\n{user_prompt}"

        prompt = system_prompt + "\n\n" + user_prompt

        response = self.llm.generate(prompt)

        return {
            "answer": response,
            "sources": list(set(sources))
        }

    # =========================================================
    # Aufarbeiten der Frage
    # =========================================================

    def rewrite_query(self, query: str) -> str:

        # ---------------------------------
        # FAST PATH (KEIN LLM CALL)
        # ---------------------------------

        if len(query.split()) <= 5:
            print("⚡ Fast Path aktiv (keine Rewrite)")
            return query

        # ---------------------------------
        # LLM REWRITE
        # ---------------------------------

        prompt = f"""
    Du bist ein Query-Optimierer für ein Dokumenten-Suchsystem.

    AUFGABE:
    Forme die Nutzerfrage in eine kurze Suchanfrage um.

    REGELN:
    - maximal 6-10 Wörter
    - keine ganzen Sätze
    - keine Fragen
    - nur Suchbegriffe
    - behalte wichtige Entitäten

    FRAGE:
    {query}

    ANTWORT:
    """

        rewritten = self.llm.generate(prompt).strip()

        return rewritten

    # =========================================================
    # Intent Function
    # =========================================================

    def detect_intent(self, query: str) -> str:

        prompt = f"""
    Du bist ein Intent-Klassifikator.

    KLASSIFIZIERE DIE FRAGE IN EINE KATEGORIE:

    1) FACT
    - konkrete Werte, Zahlen, Fakten
    - z.B. Betrag, Datum, Adresse

    2) SUMMARY
    - Erklärungen, Inhalte, Zusammenfassungen

    3) LIST
    - Aufzählung, Existenz, Dateien, Dokumente

    REGELN:
    - Antworte nur mit: FACT, SUMMARY oder LIST
    - keine Erklärung

    BEISPIELE:

    Frage: Wie hoch war die Rechnung 2023?
    Antwort: FACT

    Frage: Worum geht es im Dokument?
    Antwort: SUMMARY

    Frage: Welche Dateien gibt es?
    Antwort: LIST

    FRAGE:
    {query}

    ANTWORT:
    """

        intent = self.llm.generate(prompt).strip().upper()

        if "FACT" in intent:
            return "FACT"
        if "SUMMARY" in intent:
            return "SUMMARY"
        if "LIST" in intent:
            return "LIST"

        return "SUMMARY"
    # =================================================================


    def verify_answer(self, question: str, answer: str, context: str) -> bool:

        prompt = f"""
    Du bist ein Fakten-Checker.

    AUFGABE:
    Prüfe, ob die Antwort vollständig durch die Quellen gestützt wird.

    REGELN:
    - Wenn die Antwort im Kontext steht → OK
    - Wenn auch nur ein Teil erfunden ist → NOT_OK
    - Zahlen müssen exakt übereinstimmen
    - Keine Vermutungen erlauben

    ANTWORT NUR MIT:
    OK oder NOT_OK

    FRAGE:
    {question}

    ANTWORT:
    {answer}

    QUELLEN:
    {context}

    ERGEBNIS:
    """

        result = self.llm.generate(prompt).strip().upper()

        if "OK" in result:
            return True

        return False

    # =========================================================
    # MAIN LOOP
    # =========================================================

    def run(self, query: str):

        print("\n🚀 NEW RUN:", query)

        context = ""
        history = ""

        final_answer = None

        found_docs = False
        found_files = False

        sources_last = []

        # # =====================================
        # # 1.1 SPECIAL SUMMARY MODE
        # # =====================================
        query_lower = query.lower()

        if (
            "zusammenfassung" in query_lower
            or "zusammenfassen" in query_lower
            or "fasse das dokument" in query_lower
        ):
            result = self.handle_summary(query)

            if result:
                return result

        # =====================================
        # 1. INTENT DETECTION
        # =====================================

        intent = self.detect_intent(query)
        print("🧠 INTENT:", intent)

        # =====================================
        # 2. QUERY REWRITE
        # =====================================

        optimized_query = self.rewrite_query(query)
        print("🔧 Rewritten Query:", optimized_query)

        # =====================================
        # 3. DYNAMIC TOP_K
        # =====================================

        if intent == "FACT":
            top_k = 3
        elif intent == "SUMMARY":
            top_k = 5
        else:  # LIST
            top_k = 10

        # =====================================
        # 4. RETRIEVAL
        # =====================================

        docs = self.retriever.search(optimized_query, top_k=top_k)

        if docs:
            found_docs = True

        # =====================================
        # 5. BUILD CONTEXT
        # =====================================

        context_blocks = []

        for d in docs:
            block = (
                f"[SOURCE: {d['source']}]\n"
                f"{d['content']}\n"
            )

            context_blocks.append(block)
            sources_last.append(d["source"])

        context = "\n---\n".join(context_blocks)

        # =====================================
        # 6. LIST MODE SPECIAL HANDLING
        # =====================================

        if intent == "LIST":

            # LIST = often direct existence answer possible
            # but we still use LLM for formatting

            system_prompt = """
    Du bist ein Dokumenten-Assistent.

    Gib eine klare Auflistung der gefundenen Informationen.
    Wenn möglich, gruppiere ähnliche Einträge.

    WICHTIG:
    - nur aus den Quellen antworten
    - keine Erfindungen
    """

        else:

            system_prompt = """
    Du bist ein präziser Dokumenten-Assistent.

    REGELN:
    - Antworte nur auf Basis der Quellen
    - Keine Halluzinationen
    - Wenn nichts gefunden: sage "Keine Information gefunden"
    - Nenne am Ende die Quellen
    """

        # =====================================
        # 7. USER PROMPT
        # =====================================

        user_prompt = f"""
    FRAGE:
    {query}

    QUELLEN:
    {context}

    ANTWORT:
    - präzise
    - keine Zusatzinformationen
    """

        # =====================================
        # 8. HISTORY (optional)
        # =====================================

        if history:
            user_prompt = f"VERLAUF:\n{history}\n\n{user_prompt}"

        # =====================================
        # 9. LLM CALL
        # =====================================

        response = self.llm.generate(
            system_prompt + "\n\n" + user_prompt
        )

        is_valid = self.verify_answer(query, response, context)

        print("🧪 VERIFICATION:", is_valid)

        if not is_valid:
            print("⚠️ Antwort nicht valid → fallback")

            fallback_prompt = f"""
        Beantworte die Frage NUR mit den Quellen.

        FRAGE:
        {query}

        QUELLEN:
        {context}

        Wenn keine klare Antwort möglich ist, sage:
        "Keine ausreichenden Informationen vorhanden."
        """

            response = self.llm.generate(fallback_prompt)


        # =====================================
        # 10. FINAL OUTPUT
        # =====================================

        if not final_answer:
            final_answer = response

        # =====================================
        # 11. SOURCES
        # =====================================

        unique_sources = list(dict.fromkeys(sources_last[:3]))

        if unique_sources:
            final_answer += "\n\n📚 Quellen:\n"
            final_answer += "\n".join(f"- {s}" for s in unique_sources)

        return final_answer

    def run_stream(self, query, token_callback):
        """
        Streamt Antwort ins UI + liefert strukturierte Quellen zurück
        """

        # ---------------------------------
        # 1. RETRIEVAL
        # ---------------------------------

        docs = self.retriever.search(query)

        # Sicherheit: falls irgendwas schiefgeht
        if not isinstance(docs, list):
            docs = []

        # ---------------------------------
        # 2. PROMPT BAUEN
        # ---------------------------------

        prompt = self.build_prompt(query, docs)

        # ---------------------------------
        # 3. STREAMING RESPONSE
        # ---------------------------------

        full_response = ""

        for token in self.llm.generate_stream(prompt):
            full_response += token
            token_callback(token)

        # ---------------------------------
        # 4. SOURCES NORMALISIEREN
        # ---------------------------------

        sources = []

        for d in docs:

            if isinstance(d, dict):

                sources.append({
                    "source": d.get("source", "unknown"),
                    "score": d.get("score", 0.0),
                    "content": d.get("content", "")
                })

            elif isinstance(d, (list, tuple)):

                # Fallback falls Retriever mal Liste liefert
                sources.append({
                    "source": d[0] if len(d) > 0 else "unknown",
                    "score": d[1] if len(d) > 1 else 0.0,
                    "content": d[2] if len(d) > 2 else ""
                })

            else:

                sources.append({
                    "source": str(d),
                    "score": 0.0,
                    "content": ""
                })

        # ---------------------------------
        # 5. CLEAN ANSWER (optional)
        # ---------------------------------

        answer = full_response

        # harte Reinigung
        if "Quellen:" in answer:
            answer = answer.split("Quellen:")[0].strip()

        if "Quellen:" in answer:
            answer = answer.replace("Quellen:", "")

        # ---------------------------------
        # 6. RETURN
        # ---------------------------------

        return {
            "answer": answer,
            "sources": sources
        }

    def build_prompt(self, query: str, context):
        """
        Baut den Prompt für Streaming / normal Mode
        """

        # falls context Liste ist
        if isinstance(context, list):
            context_text = "\n\n".join(
                f"[{c.get('source')}]\n{c.get('content')}"
                for c in context
            )
        else:
            context_text = str(context)

        return f"""
    DU BIST EIN DOKUMENTEN-ASSISTENT.

    REGELN:
    - Antworte nur auf Basis der Quellen.
    - Wenn nichts passt: sage "Keine Information gefunden."
    - Nenne Quellen am Ende.
    
    WICHTIG:
    - Schreibe KEINE Quellen
    - Schreibe KEINE "Quellen:" Sektion
    - Antworte nur mit dem reinen Ergebnis

    FRAGE:
    {query}

    QUELLEN:
    {context_text}

    ANTWORT:
    """

    #=============================================================
    # =============================================================
    # Ermöglicht Fragen wie: Fasse das Dokument xyz zusammen

    def handle_summary(self, query):

        with open(DOCUMENTS_PATH, "rb") as f:
            documents = pickle.load(f)

        query_lower = query.lower()

        # passendes Dokument finden

        matching_sources = []

        for doc in documents:

            source = doc["source"]

            if source.lower() in query_lower:
                matching_sources.append(doc)

        if not matching_sources:
            return None

        full_text = "\n\n".join(
            d["content"]
            for d in matching_sources
        )

        prompt = f"""
    Erstelle eine strukturierte Zusammenfassung.

    Liefere:

    - Thema
    - Wichtige Personen
    - Wichtige Termine
    - Wichtige Beträge
    - Wichtige Fristen
    - Kurzfassung

    Dokument:

    {full_text[:12000]}
    """

        answer = self.llm.generate(prompt)

        return (
                answer
                + "\n\n📚 Quelle:\n"
                + matching_sources[0]["source"]
        )