import os
import time
import asyncio
import logging
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

# Require security-critical env vars
API_KEY = os.getenv("API_KEY", "").strip()
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "").strip()
if len(API_KEY) < 24:
    raise RuntimeError("API_KEY must be set to a strong value (>= 24 chars).")
if len(JWT_SECRET_KEY) < 32:
    raise RuntimeError("JWT_SECRET_KEY must be set to a strong value (>= 32 chars).")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# CORS: explicit origins; include local dev by default if none provided
_raw_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
if _raw_origins:
    allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
else:
    allowed_origins = ["http://localhost:3000", "http://localhost:5173"]

# Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("ergo-api")

app = FastAPI(
    title="Ergo AI Tutor API",
    description="AI tutoring service with Ollama backend",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Simple Rate Limiter
# -------------------------
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
_rate_buckets = defaultdict(list)

def rate_limit_dependency(request: Request):
    path = request.url.path
    if request.method == "OPTIONS" or path in ("/health", "/docs", "/openapi.json"):
        return

    now = time.time()
    ip = request.client.host if request.client else "unknown"
    bucket = _rate_buckets[ip]

    cutoff = now - RATE_LIMIT_WINDOW
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)

    if len(bucket) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Too many requests")

    bucket.append(now)

# -------------------------
# API Key Guard (always enforced)
# -------------------------
def require_api_key(request: Request):
    provided = request.headers.get("X-API-Key", "").strip()
    if provided != API_KEY:
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
    os.makedirs("data", exist_ok=True)
    await tutor.warmup()
    try:
        info = await asyncio.to_thread(tutor.ai_backend.health)
        models = info.get("models") or info  # shape differs across Ollama versions
        count = len(models) if isinstance(models, list) else 0
        log.info("Connected to Ollama at %s (models: %s)", tutor.ai_backend.base_url, count)
    except Exception as e:
        log.warning("Ollama health check failed on startup: %s", e)

# -------------------------
# Routes
# -------------------------
@app.get("/health")
async def health():
    """
    Liveness/readiness. Returns Ollama status too.
    """
    ok = True
    ollama_ok = False
    models_count = 0
    try:
        info = await asyncio.to_thread(tutor.ai_backend.health)
        models = info.get("models") or info
        if isinstance(models, list):
            models_count = len(models)
        ollama_ok = True
    except Exception:
        ok = False

    return {
        "status": "ok" if ok else "degraded",
        "ollama": {"ok": ollama_ok, "url": tutor.ai_backend.base_url, "models_count": models_count},
    }

@app.get("/dev/ollama", dependencies=[Depends(require_api_key)])
async def dev_ollama():
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
