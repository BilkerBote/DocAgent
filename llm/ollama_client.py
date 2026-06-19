import requests


class OllamaClient:

    def __init__(self, model="llama3"):
        self.url = "http://localhost:11434/api/generate"
        self.model = model

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code != 200:
            raise Exception(f"Ollama error: {response.text}")

        return response.json()["response"]

    def generate_stream(self, prompt: str):
        """
        Bildet Tokens / Chunks statt komplette Antwort
        """

        import ollama

        stream = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        for chunk in stream:
            if "message" in chunk:
                content = chunk["message"].get("content", "")
                if content:
                    yield content