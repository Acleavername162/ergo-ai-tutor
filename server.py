import os
import time
import asyncio
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from ergo_ai_tutor import ErgoAITutor

# -------------------------
# Environment & App Setup
# -------------------------
load_dotenv()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Comma-separated list of origins in .env (e.g., "https://ai.therisehub.org,http://localhost:3000")
allowed_origins = [
    o.strip() for o in (os.getenv("ALLOWED_ORIGINS") or "").split(",") if o.strip()
]

app = FastAPI(
    title="Ergo AI Tutor API",
    description="AI tutoring service with Ollama backend",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Simple Rate Limiter
# -------------------------
RATE_LIMIT_REQUESTS = 30
RATE_LIMIT_WINDOW = 60  # seconds
_rate_buckets = defaultdict(list)

def rate_limit_dependency(request: Request):
    """
    Very lightweight, in-memory IP-based rate limiter.
    Exempts OPTIONS, /health, /docs, /openapi.json.
    """
    path = request.url.path
    if request.method == "OPTIONS" or path in ("/health", "/docs", "/openapi.json"):
        return

    now = time.time()
    ip = request.client.host if request.client else "unknown"
    bucket = _rate_buckets[ip]

    # Drop old timestamps out of the window
    cutoff = now - RATE_LIMIT_WINDOW
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)

    if len(bucket) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Too many requests")

    bucket.append(now)

# -------------------------
# Optional API Key Guard
# -------------------------
def require_api_key(request: Request):
    """
    Enforce X-API-Key only if API_KEY is set in the environment.
    If API_KEY is empty/unset, this becomes a no-op (open endpoint).
    """
    expected = os.getenv("API_KEY", "").strip()
    if not expected:
        return  # no API key configured → do not enforce
    provided = request.headers.get("X-API-Key", "").strip()
    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

# -------------------------
# Models
# -------------------------
class QuestionRequest(BaseModel):
    user_id: str
    question: str
    subject: Optional[str] = None
    level: Optional[str] = None
    model: Optional[str] = None

class AnswerResponse(BaseModel):
    model_used: str
    answer: str

# -------------------------
# Tutor Instance
# -------------------------
tutor = ErgoAITutor()

@app.on_event("startup")
async def _startup():
    # Ensure local data folder for sqlite, logs, etc.
    os.makedirs("data", exist_ok=True)
    # Warm up Ollama health (non-fatal if it fails)
    await tutor.warmup()

# -------------------------
# Routes
# -------------------------
@app.get("/health")
async def health():
    """
    Lightweight liveness/readiness endpoint.
    """
    return {"status": "ok"}

@app.get("/dev/ollama")
async def dev_ollama():
    """
    Debug endpoint to verify Ollama connectivity and list pulled models.
    """
    try:
        data = await asyncio.to_thread(tutor.ai_backend.health)
        return {"ollama_url": tutor.ai_backend.base_url, "models": data}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama not reachable: {e}")

@app.post(
    "/tutor/question",
    response_model=AnswerResponse,
    dependencies=[Depends(rate_limit_dependency), Depends(require_api_key)],
)
async def ask_question(req: QuestionRequest):
    """
    Main tutor endpoint. Passes through subject/level and optional model.
    Enforces API key if API_KEY is set in .env.
    """
    try:
        answer = await tutor.get_answer(
            req.question,
            model=req.model,
            user_id=req.user_id,
            subject=req.subject,
            level=req.level,
        )
        model_used = req.model or tutor.ai_backend.model
        return AnswerResponse(model_used=model_used, answer=answer)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI error: {e}")

# -------------------------
# Dev Entrypoint
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False)
