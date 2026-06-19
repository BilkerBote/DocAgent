class SearchTool:
    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, query: str):
        return self.retriever.search(query)