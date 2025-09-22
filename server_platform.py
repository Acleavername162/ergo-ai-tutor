import os
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn

from ergo_platform_ai import ErgoPlatformAI
from platform_bridge import PlatformBridge

app = FastAPI(
    title="Ergo AI Tutor - Platform Integration",
    description="AI-powered educational platform with full data access",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize enhanced AI tutor
ergo = ErgoPlatformAI()
platform = PlatformBridge()

# Security
security = HTTPBearer()

# Request models
class PlatformSessionRequest(BaseModel):
    user_id: str
    subject: str
    subtopic: Optional[str] = None

class PlatformQuestionRequest(BaseModel):
    user_id: str
    question: str
    course_id: Optional[str] = None

class ProgressUpdateRequest(BaseModel):
    user_id: str
    course_id: str
    lesson_id: str
    completion: float

# Authentication middleware
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token (implement your auth logic)"""
    token = credentials.credentials
    # Add your token verification logic here
    # For now, accept any token
    return {"user_id": "verified"}

@app.get("/")
async def root():
    return {
        "message": "Ergo AI Tutor - Platform Integration API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/ai/session/start")
async def start_platform_session(req: PlatformSessionRequest):
    """Start AI tutoring session with platform context"""
    try:
        session = await ergo.start_personalized_session(
            req.user_id, 
            req.subject, 
            req.subtopic
        )
        return {"success": True, "data": session}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/question")
async def handle_platform_question(req: PlatformQuestionRequest):
    """Handle student question with platform context"""
    try:
        response = await ergo.handle_platform_question(
            req.user_id,
            req.question,
            req.course_id
        )
        return {"success": True, "answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/progress")
async def update_progress(req: ProgressUpdateRequest):
    """Update student learning progress"""
    try:
        await ergo.update_learning_progress(
            req.user_id,
            req.course_id,
            req.lesson_id,
            req.completion
        )
        return {"success": True, "message": "Progress updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai/analytics/{user_id}")
async def get_analytics(user_id: str):
    """Get learning analytics for user"""
    try:
        analytics = await ergo.get_learning_analytics(user_id)
        return {"success": True, "analytics": analytics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai/recommendations/{user_id}")
async def get_recommendations(user_id: str):
    """Get AI recommendations for user"""
    try:
        recommendations = await platform.get_recommended_content(user_id)
        return {"success": True, "recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting Ergo AI Platform Integration on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
