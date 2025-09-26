import os
import time
import asyncio
import logging
from datetime import datetime
from collections import defaultdict
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from ergo_ai_tutor import ErgoAITutor
from platform_bridge import PlatformBridge

# =========================
# Env, Logging, App
# =========================
load_dotenv()

API_KEY = os.getenv("API_KEY", "").strip()
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "").strip()  # reserved for future JWT
if len(API_KEY) < 24:
    raise RuntimeError("API_KEY must be set to a strong value (>= 24 chars) in .env")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

_raw_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()] or [
    "http://localhost:3000",
    "http://localhost:5173",
]

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("ergo-platform-api")

app = FastAPI(
    title="Ergo AI Tutor - Platform Integration",
    description="AI-powered educational platform with Tutor + SQLite bridge",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Rate Limit (simple, IP)
# =========================
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
_rate_buckets = defaultdict(list)

def rate_limit_dependency(request: Request):
    path = request.url.path
    if request.method == "OPTIONS" or path in ("/", "/health", "/docs", "/openapi.json"):
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

# =========================
# API Key Guard (Bearer or Header)
# =========================
def require_api_key(request: Request):
    """
    Accept either:
      - Header: X-API-Key: <API_KEY>
      - Authorization: Bearer <API_KEY>
    """
    provided = request.headers.get("X-API-Key", "").strip()
    if not provided:
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            provided = auth.split(" ", 1)[1].strip()
    if provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# =========================
# Data Models
# =========================
class PlatformSessionRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    subject: str = Field(..., min_length=1)
    subtopic: Optional[str] = None
    level: Optional[str] = None

class PlatformQuestionRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    course_id: Optional[str] = None
    subject: Optional[str] = None
    level: Optional[str] = None
    model: Optional[str] = None

class ProgressUpdateRequest(BaseModel):
    user_id: str
    course_id: str
    lesson_id: str
    completion: float = Field(ge=0.0, le=1.0)
    status: Optional[str] = Field(
        default="in_progress",
        description="e.g., started | in_progress | completed",
    )

class AnswerEnvelope(BaseModel):
    model_used: str
    answer: str
    timestamp: str

class SessionEnvelope(BaseModel):
    user_id: str
    subject: str
    subtopic: Optional[str] = None
    level: Optional[str] = None
    started_at: str

class AnalyticsEnvelope(BaseModel):
    user_id: str
    totals: Dict[str, int]
    last_update: Optional[str] = None
    rows: List[Dict[str, Any]]

class RecommendationsEnvelope(BaseModel):
    user_id: str
    recommendations: List[Dict[str, Any]]

# =========================
# Tutor, Bridge, Session Store
# =========================
tutor = ErgoAITutor()
bridge = PlatformBridge()

# In-memory session context per user
_sessions: Dict[str, Dict[str, Optional[str]]] = {}

@app.on_event("startup")
async def _startup():
    os.makedirs("data", exist_ok=True)
    await tutor.warmup()
    try:
        info = await asyncio.to_thread(tutor.ai_backend.health)
        models = info.get("models") or info
        log.info("Ollama OK at %s (models: %s)", tutor.ai_backend.base_url, len(models) if isinstance(models, list) else "n/a")
    except Exception as e:
        log.warning("Ollama health check failed: %s", e)

# =========================
# Basic Routes
# =========================
@app.get("/")
async def root():
    return {
        "message": "Ergo AI Tutor - Platform Integration API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health")
async def health():
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
        "time": datetime.utcnow().isoformat() + "Z",
    }

@app.get("/dev/ollama", dependencies=[Depends(require_api_key)])
async def dev_ollama():
    try:
        data = await asyncio.to_thread(tutor.ai_backend.health)
        return {"ollama_url": tutor.ai_backend.base_url, "models": data}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama not reachable: {e}")

# =========================
# Platform-Integrated Routes
# =========================
@app.post(
    "/ai/session/start",
    dependencies=[Depends(rate_limit_dependency), Depends(require_api_key)],
    response_model=SessionEnvelope,
)
async def start_platform_session(req: PlatformSessionRequest):
    """
    Start an AI tutoring session with subject/subtopic context.
    Persists/updates user profile in SQLite.
    """
    try:
        # Persist/refresh basic user profile
        await asyncio.to_thread(bridge.upsert_user, req.user_id, None)

        # Store session context in memory
        _sessions[req.user_id] = {
            "subject": (req.subject or "").strip().lower(),
            "subtopic": (req.subtopic or "").strip().lower() if req.subtopic else None,
            "level": (req.level or "").strip().lower() if req.level else None,
        }
        return SessionEnvelope(
            user_id=req.user_id,
            subject=_sessions[req.user_id]["subject"],
            subtopic=_sessions[req.user_id]["subtopic"],
            level=_sessions[req.user_id]["level"],
            started_at=datetime.utcnow().isoformat() + "Z",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start session: {e}")

@app.post(
    "/ai/question",
    dependencies=[Depends(rate_limit_dependency), Depends(require_api_key)],
    response_model=AnswerEnvelope,
)
async def handle_platform_question(req: PlatformQuestionRequest):
    """
    Answer a learner's question using Tutor and current session context.
    Falls back to req.subject/level if no session stored.
    """
    try:
        ctx = _sessions.get(req.user_id, {})
        subject = req.subject or ctx.get("subject")
        level = req.level or ctx.get("level")

        answer = await tutor.get_answer(
            req.question,
            model=req.model,
            user_id=req.user_id,
            subject=subject,
            level=level,
        )
        model_used = req.model or tutor.ai_backend.model
        # (Optional) record a lightweight "touched" progress state
        if req.course_id:
            try:
                await asyncio.to_thread(
                    bridge.set_progress,
                    req.user_id,
                    req.course_id,
                    lesson_id="qa",
                    status="touched",
                )
            except Exception as _:
                # ignore progress errors for Q&A
                pass

        return AnswerEnvelope(
            model_used=model_used,
            answer=answer,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {e}")

@app.post(
    "/ai/progress",
    dependencies=[Depends(rate_limit_dependency), Depends(require_api_key)],
)
async def update_progress(req: ProgressUpdateRequest):
    """
    Update student learning progress for a specific lesson.
    """
    try:
        status = req.status or ("completed" if req.completion >= 0.999 else "in_progress")
        await asyncio.to_thread(
            bridge.set_progress, req.user_id, req.course_id, req.lesson_id, status
        )
        return {"success": True, "message": "Progress updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update progress: {e}")

@app.get(
    "/ai/analytics/{user_id}",
    dependencies=[Depends(rate_limit_dependency), Depends(require_api_key)],
    response_model=AnalyticsEnvelope,
)
async def get_analytics(user_id: str):
    """
    Return simple analytics summary from SQLite progress.
    """
    try:
        rows = await asyncio.to_thread(bridge.get_progress, user_id)
        totals: Dict[str, int] = defaultdict(int)
        last = None
        parsed_rows: List[Dict[str, Any]] = []
        for r in rows:
            totals[str(r.get("status"))] += 1
            last = last or r.get("updated_at")
            parsed_rows.append(dict(r))
        return AnalyticsEnvelope(
            user_id=user_id,
            totals=dict(totals),
            last_update=last,
            rows=parsed_rows,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute analytics: {e}")

@app.get(
    "/ai/recommendations/{user_id}",
    dependencies=[Depends(rate_limit_dependency), Depends(require_api_key)],
    response_model=RecommendationsEnvelope,
)
async def get_recommendations(user_id: str):
    """
    Naive recommendations based on progress:
      - If no progress, suggest starting 'course_101:lesson_1'
      - If incomplete lessons exist, suggest the next lesson_id in same course
      - Otherwise, suggest review of last 1–2 completed items
    """
    try:
        rows = await asyncio.to_thread(bridge.get_progress, user_id)
        if not rows:
            recs = [{"course_id": "course_101", "lesson_id": "lesson_1", "reason": "Get started"}]
            return RecommendationsEnvelope(user_id=user_id, recommendations=recs)

        # Organize by course
        by_course: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_course[str(r["course_id"])].append(dict(r))

        recommendations: List[Dict[str, Any]] = []
        for course_id, items in by_course.items():
            # Try to find any non-completed item to continue
            incomplete = [i for i in items if i.get("status") != "completed"]
            if incomplete:
                # Recommend the most recent incomplete
                candidate = incomplete[0]
                recommendations.append(
                    {
                        "course_id": course_id,
                        "lesson_id": candidate.get("lesson_id"),
                        "reason": "Continue where you left off",
                    }
                )
            else:
                # All completed—recommend review of the last one
                latest = items[0]
                recommendations.append(
                    {
                        "course_id": course_id,
                        "lesson_id": latest.get("lesson_id"),
                        "reason": "Quick review to reinforce learning",
                    }
                )

        return RecommendationsEnvelope(user_id=user_id, recommendations=recommendations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build recommendations: {e}")

# =========================
# Dev Entrypoint
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_platform:app", host=HOST, port=PORT, reload=False)
