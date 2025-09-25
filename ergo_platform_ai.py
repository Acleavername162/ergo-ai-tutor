import os
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()


class ErgoAITutor:
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        self.default_model = os.getenv("DEFAULT_MODEL", "llama3.1:8b-instruct-q4_K_M")

    async def get_answer(self, question: str, model: Optional[str] = None) -> str:
        """
        Call Ollama /api/generate with a tutoring-friendly prompt.
        """
        model_to_use = (model or self.default_model).strip()

        prompt = (
            "You are an AI tutor. Provide a clear, educational response. "
            "Explain step-by-step when appropriate and be encouraging.\n\n"
            f"Question: {question}\n\nAnswer:"
        )

        payload = {
            "model": model_to_use,
            "prompt": prompt,
            "options": {"temperature": 0.6, "top_p": 0.9},
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
            data = r.json()

        resp = data.get("response")
        if not isinstance(resp, str):
            raise RuntimeError(f"Unexpected Ollama response: {data}")

        return resp.strip()

    # optional sync helper; safe for CLI/admin use
    def get_answer_sync(self, question: str, model: Optional[str] = None) -> str:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.get_answer(question, model))
        else:
            return loop.run_until_complete(self.get_answer(question, model))

    async def health_check(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{self.ollama_url}/api/tags")
                r.raise_for_status()
            return {
                "status": "healthy",
                "ollama_url": self.ollama_url,
                "message": "Ollama service is responding",
            }
        except Exception as e:
            return {"status": "unhealthy", "ollama_url": self.ollama_url, "error": str(e)}
