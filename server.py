import os
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn

from ergo_ai_tutor import ErgoAITutor, UserProfile, SubjectType, LearningStyle

app = FastAPI(
    title="Ergo AI Tutor API",
    description="AI-powered educational platform with Ollama backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ergo = ErgoAITutor()
rate_limit_store = {}

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

def check_rate_limit(user_id: str, limit_per_minute: int = 30) -> bool:
    current_minute = int(time.time() / 60)
    key = f"{user_id}_{current_minute}"
    
    if key not in rate_limit_store:
        rate_limit_store[key] = 0
    
    if rate_limit_store[key] >= limit_per_minute:
        return False
    
    rate_limit_store[key] += 1
    
    keys_to_remove = [k for k in rate_limit_store.keys() if not k.endswith(str(current_minute))]
    for k in keys_to_remove:
        del rate_limit_store[k]
    
    return True

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    user_id = request.headers.get("X-User-ID", "anonymous")
    
    if not check_rate_limit(user_id):
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Please try again in a minute."
        )
    
    response = await call_next(request)
    return response

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/tutor/session/start")
async def start_session(req: StartSessionRequest):
    try:
        user_profile = UserProfile(
            user_id=req.user.user_id,
            name=req.user.name,
            grade_level=req.user.grade_level,
            learning_style=LearningStyle(req.user.learning_style),
            mbti_type=req.user.mbti_type,
            strengths=req.user.strengths,
            weaknesses=req.user.weaknesses,
            current_subjects=req.user.current_subjects,
            progress=req.user.progress
        )
        
        session = await ergo.start_tutoring_session(
            user_profile, 
            SubjectType(req.subject), 
            req.subtopic
        )
        
        return {"success": True, "data": session}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")

@app.post("/tutor/question")
async def handle_question(req: QuestionRequest):
    try:
        answer = await ergo.handle_student_question(req.user_id, req.question)
        return {"success": True, "answer": answer}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")

@app.get("/dev/stats")
async def get_system_stats():
    return {
        "active_sessions": len(ergo.active_sessions),
        "conversation_history_users": len(ergo.ollama.conversation_history),
        "rate_limit_entries": len(rate_limit_store),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )
