import os
import time
from typing import Optional
from enum import Enum
from collections import defaultdict
from datetime import datetime, timedelta

import psutil
import httpx
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv  # <-- fixed
from ergo_ai_tutor import ErgoAITutor

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Ergo AI Tutor API",
    description="Dual-model AI tutoring service with subject-based routing",
    version="2.0.0",
)

# CORS Configuration
allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting storage
rate_limit_store = defaultdict(list)
RATE_LIMIT_REQUESTS = 30
RATE_LIMIT_WINDOW = 60  # seconds

class SubjectType(str, Enum):
    MATH = "MATH"
    CS = "CS"
    COMPUTER_SCIENCE = "COMPUTER_SCIENCE"
    SCIENCE = "SCIENCE"
    HUMANITIES = "HUMANITIES"
    ELA = "ELA"
    ENGLISH = "ENGLISH"
    HISTORY = "HISTORY"
    SOCIAL = "SOCIAL"
    SOCIAL_STUDIES = "SOCIAL_STUDIES"

class QuestionRequest(BaseModel):
    user_id: str
    question: str
    subject: Optional[SubjectType] = None
    level: Optional[str] = None
    model: Optional[str] = None

    @field_validator("subject", mode="before")
    @classmethod
    def validate_subject(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            v_upper = v.upper().replace(" ", "_")
            for subject in SubjectType:
                if subject.value == v_upper:
                    return subject
            aliases = {
                "COMPUTER_SCIENCE": SubjectType.CS,
                "ENGLISH": SubjectType.ELA,
                "SOCIAL_STUDIES": SubjectType.SOCIAL,
            }
            if v_upper in aliases:
                return aliases[v_upper]
        return v

class AnswerResponse(BaseModel):
    model_used: str
    answer: str

class SessionStartRequest(BaseModel):
    user_id: str

class SessionStartResponse(BaseModel):
    session_id: str
    message: str

def check_rate_limit(user_id: str) -> bool:
    now = datetime.now()
    cutoff = now - timedelta(seconds=RATE_LIMIT_WINDOW)
    rate_limit_store[user_id] = [t for t in rate_limit_store[user_id] if t > cutoff]
    if len(rate_limit_store[user_id]) >= RATE_LIMIT_REQUESTS:
        return False
    rate_limit_store[user_id].append(now)
    return True

def get_user_id(request: Request) -> str:
    return request.headers.get("X-User-ID", "anonymous")

def rate_limit_dependency(request: Request):
    if request.method == "OPTIONS" or request.url.path == "/health":
        return
    user_id = get_user_id(request)
    if not check_rate_limit(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

def get_model_for_request(
    subject: Optional[SubjectType],
    level: Optional[str],
    model_override: Optional[str],
) -> str:
    if model_override:
        return model_override

    default_model = os.getenv("DEFAULT_MODEL", "llama3.1:8b-instruct-q4_K_M")
    if not subject:
        return default_model

    is_hard_level = False
    if level:
        L = level.lower()
        is_hard_level = any(k in L for k in ["college", "ap", "advanced"])

    if subject in [SubjectType.MATH, SubjectType.CS, SubjectType.COMPUTER_SCIENCE]:
        if is_hard_level:
            m = os.getenv("MATH_CS_MODEL_HARD")
            if m:
                return m
        return os.getenv("MATH_CS_MODEL", default_model)

    elif subject == SubjectType.SCIENCE:
        if is_hard_level:
            m = os.getenv("SCIENCE_MODEL_HARD")
            if m:
                return m
        return os.getenv("SCIENCE_MODEL", default_model)

    elif subject in [
        SubjectType.HUMANITIES,
        SubjectType.ELA,
        SubjectType.ENGLISH,
        SubjectType.HISTORY,
        SubjectType.SOCIAL,
        SubjectType.SOCIAL_STUDIES,
    ]:
        if is_hard_level:
            m = os.getenv("HUMANITIES_MODEL_HARD")
            if m:
                return m
        return os.getenv("HUMANITIES_MODEL", default_model)

    return default_model

# Initialize tutor
tutor = ErgoAITutor()

# Warm models on process start so first live request doesn't time out
@app.on_event("startup")
async def warm_models():
    try:
        await tutor.warm_up()
    except Exception:
        pass  # don't block startup

@app.get("/health")
async def health_check():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

@app.post("/tutor/session/start", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest, _=Depends(rate_limit_dependency)):
    session_id = f"session_{request.user_id}_{int(time.time())}"
    return SessionStartResponse(session_id=session_id, message="Tutoring session started successfully")

@app.post("/tutor/question", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest, _=Depends(rate_limit_dependency)):
    try:
        model_to_use = get_model_for_request(request.subject, request.level, request.model)
        answer = await tutor.get_answer(
            request.question,
            model=model_to_use,
            user_id=request.user_id,   # <-- pass user_id through
        )
        return AnswerResponse(model_used=model_to_use, answer=answer)
    except httpx.TimeoutException:
        raise HTTPException(status_code=502, detail="Ollama service timeout - please try again")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Unable to connect to Ollama service")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama service error: {e}")

@app.get("/dev/ollama")
async def dev_ollama_status():
    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{ollama_url}/api/tags")
            r.raise_for_status()
            return {"ok": True, "ollama_url": ollama_url, "tags": r.json()}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama unavailable at {ollama_url}: {e}")

@app.get("/dev/stats")
async def dev_stats():
    process = psutil.Process()
    return {
        "pid": os.getpid(),
        "cpu_percent": process.cpu_percent(),
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "time": datetime.utcnow().isoformat() + "Z",
    }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
