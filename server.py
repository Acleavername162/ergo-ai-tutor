import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Type, TypeVar

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from ergo_ai_tutor import ErgoAITutor, UserProfile, SubjectType, LearningStyle

T = TypeVar("T")

app = FastAPI(
    title="Ergo AI Tutor API",
    description="AI-powered educational platform with Ollama backend",
    version="1.0.0",
)

# --- CORS: allow your domain + Replit previews ---
ALLOWED_ORIGINS = [
    "https://ai.therisehub.org",
]
# Regex to allow any *.replit.app origin
ALLOWED_ORIGIN_REGEX = r"^https:\/\/[a-zA-Z0-9-]+\.replit\.app$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS + ["http://localhost:3000", "http://localhost:5173"],
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ergo = ErgoAITutor()
rate_limit_store: Dict[str, int] = {}

# ---------- Models ----------
class UserProfileRequest(BaseModel):
    user_id: str
    name: str
    grade_level: int
    learning_style: str
    mbti_type: str
    strengths: List[str]
    weaknesses: List[str]
    current_subjects: List[str]
    progress: Dict[str, Dict] = {}

class StartSessionRequest(BaseModel):
    user: UserProfileRequest
    subject: str
    subtopic: str

class QuestionRequest(BaseModel):
    user_id: str
    question: str

# ---------- Helpers ----------
def check_rate_limit(user_id: str, limit_per_minute: int = 30) -> bool:
    current_minute = int(time.time() / 60)
    key = f"{user_id}_{current_minute}"
    rate_limit_store[key] = rate_limit_store.get(key, 0) + 1
    # Garbage-collect previous minutes
    for k in list(rate_limit_store.keys()):
        if not k.endswith(str(current_minute)):
            del rate_limit_store[k]
    return rate_limit_store[key] <= limit_per_minute

def parse_enum(enum_cls: Type[T], value: str) -> Optional[T]:
    """Case-insensitive enum parser by .value or .name; returns None if not found."""
    if value is None:
        return None
    s = str(value).strip().lower()
    # match by .value
    for item in enum_cls:
        if str(item.value).lower() == s:
            return item
    # match by .name
    for item in enum_cls:
        if item.name.lower() == s:
            return item
    return None

# ---------- Middleware ----------
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Skip health + preflight
    if request.url.path == "/health" or request.method == "OPTIONS":
        return await call_next(request)

    user_id = request.headers.get("X-User-ID") or "anonymous"
    if not check_rate_limit(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again in a minute.")

    return await call_next(request)

# ---------- Routes ----------
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/tutor/session/start")
async def start_session(req: StartSessionRequest):
    try:
        # learning_style: flexible, fallback to READING if unknown
        ls = parse_enum(LearningStyle, req.user.learning_style) or LearningStyle.READING

        user_profile = UserProfile(
            user_id=req.user.user_id,
            name=req.user.name,
            grade_level=req.user.grade_level,
            learning_style=ls,
            mbti_type=req.user.mbti_type,
            strengths=req.user.strengths,
            weaknesses=req.user.weaknesses,
            current_subjects=req.user.current_subjects,
            progress=req.user.progress,
        )

        # subject: flexible; if not a valid enum value, pass through as string (ErgoAITutor handles it)
        subj_enum = parse_enum(SubjectType, req.subject)
        subject_arg: Any = subj_enum if subj_enum is not None else req.subject

        session = await ergo.start_tutoring_session(user_profile, subject_arg, req.subtopic)

        # Frontend expects { success, data: { session_context, welcome_message } }
        return {"success": True, "data": session}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")

@app.post("/tutor/question")
async def handle_question(req: QuestionRequest):
    try:
        result = await ergo.handle_student_question(req.user_id, req.question)
        # Flatten to { success, answer: "<string>" } for the frontend
        return {"success": True, "answer": result.get("answer", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")

@app.get("/dev/stats")
async def get_system_stats():
    try:
        return {
            "active_sessions": len(ergo.active_sessions),
            "conversation_history_users": len(ergo.ai_backend.conversation_history),
            "rate_limit_entries": len(rate_limit_store),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception:
        # keep this endpoint non-fatal
        return {
            "active_sessions": len(ergo.active_sessions),
            "conversation_history_users": "unknown",
            "rate_limit_entries": len(rate_limit_store),
            "timestamp": datetime.now().isoformat(),
        }

# ---------- Entrypoint ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
