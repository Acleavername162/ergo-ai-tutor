#!/usr/bin/env python3
"""
Ergo AI Tutor - One-Command Setup Script
Creates the complete project structure with all files
"""

import os
import sys

def create_file(path, content):
    """Create a file with the given content"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Created {path}")

def setup_ergo_project():
    """Set up the complete Ergo AI Tutor project"""
    
    print("🎓 Setting up Ergo AI Tutor project...")
    print("=" * 50)
    
    # Create main AI tutor system - FIXED VERSION
    ergo_content = """import os
import time
import asyncio
import json
import requests
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import random
from datetime import datetime, timedelta
import hashlib
import uuid

class LearningStyle(Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING = "reading"

class DifficultyLevel(Enum):
    BEGINNER = 1
    ELEMENTARY = 2
    INTERMEDIATE = 3
    ADVANCED = 4
    EXPERT = 5

class SubjectType(Enum):
    MATH = "mathematics"
    SCIENCE = "science"
    ENGLISH = "english"
    HISTORY = "history"
    PROGRAMMING = "programming"

@dataclass
class UserProfile:
    user_id: str
    name: str
    grade_level: int
    learning_style: LearningStyle
    mbti_type: str
    strengths: List[str]
    weaknesses: List[str]
    current_subjects: List[str]
    progress: Dict[str, Dict]

class OllamaAIBackend:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1"):
        self.base_url = os.getenv("OLLAMA_URL", base_url)
        self.model = os.getenv("OLLAMA_MODEL", model)
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", "30"))
        self.conversation_history = {}
    
    async def _post(self, url, json_data, timeout):
        return await asyncio.to_thread(requests.post, url, json=json_data, timeout=timeout)
    
    async def chat(self, prompt: str, user_id: str, context: Optional[Dict] = None) -> str:
        full_prompt = self._build_context_prompt(prompt, user_id, context)
        
        for attempt in range(3):
            try:
                response = await self._post(
                    f"{self.base_url}/api/generate",
                    {
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_tokens": 1000
                        }
                    },
                    self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    ai_response = result.get('response', '')
                    self._update_conversation_history(user_id, prompt, ai_response)
                    return ai_response
                else:
                    if attempt == 2:
                        return "I'm having trouble connecting right now. Let's try again!"
                    
            except Exception as e:
                if attempt == 2:
                    print(f"Ollama API Error after 3 attempts: {e}")
                    return "I'm experiencing some technical difficulties. Please try again."
                
                await asyncio.sleep(0.5 * (2 ** attempt))
        
        return "I'm having trouble connecting right now. Let's try again!"
    
    def _build_context_prompt(self, prompt: str, user_id: str, context: Optional[Dict] = None) -> str:
        base_personality = '''You are Ergo, an AI tutor inspired by Albert Einstein. You are:
        - Patient, encouraging, and intellectually curious
        - A master at explaining complex concepts simply
        - Focused on helping students discover answers through guided questions
        - Always bringing lessons back to real-world applications
        - Context-aware: if a student asks off-topic questions, gently redirect to the current lesson
        
        Teaching Philosophy:
        1. Principle first (explain the "why")
        2. Practice through examples
        3. Application in real scenarios
        4. Turn mistakes into learning opportunities'''
        
        current_context = ""
        if context:
            current_lesson = context.get('current_lesson', '')
            subject = context.get('subject', '')
            difficulty = context.get('difficulty', '')
            
            current_context = f'''
            Current Context:
            - Subject: {subject}
            - Lesson: {current_lesson}
            - Difficulty Level: {difficulty}
            - Keep responses focused on this topic'''
        
        history = self._get_recent_history(user_id, limit=3)
        history_text = "\\n".join([f"Student: {h['user']}\\nErgo: {h['assistant']}" for h in history])
        
        full_prompt = f'''{base_personality}
        
        {current_context}
        
        Recent Conversation:
        {history_text}
        
        Current Student Question: {prompt}
        
        Respond as Ergo, keeping your response engaging, educational, and on-topic:'''
        
        return full_prompt
    
    def _update_conversation_history(self, user_id: str, user_msg: str, ai_response: str):
        if user_id.startswith("ergo_"):
            return
            
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        self.conversation_history[user_id].append({
            'timestamp': datetime.now().isoformat(),
            'user': user_msg,
            'assistant': ai_response
        })
        
        if len(self.conversation_history[user_id]) > 10:
            self.conversation_history[user_id] = self.conversation_history[user_id][-10:]
    
    def _get_recent_history(self, user_id: str, limit: int = 3) -> List[Dict]:
        if user_id not in self.conversation_history:
            return []
        return self.conversation_history[user_id][-limit:]

class ErgoAITutor:
    def __init__(self):
        self.ollama = OllamaAIBackend()
        self.active_sessions = {}
        self.progress_data = {}
    
    async def start_tutoring_session(self, user_profile: UserProfile, subject: SubjectType, subtopic: str) -> Dict:
        session_context = {
            'user_id': user_profile.user_id,
            'current_subject': subject,
            'subtopic': subtopic,
            'progress': 0,
            'start_time': datetime.now()
        }
        
        self.active_sessions[user_profile.user_id] = session_context
        
        welcome_prompt = f"Welcome {user_profile.name} to learning about {subtopic}! Let's start this exciting lesson together."
        
        welcome_message = await self.ollama.chat(
            welcome_prompt, 
            user_profile.user_id,
            context={
                'current_lesson': subtopic,
                'subject': subject.value,
                'difficulty': 'beginner'
            }
        )
        
        return {
            'session_context': session_context,
            'welcome_message': welcome_message,
            'session_id': user_profile.user_id
        }
    
    async def handle_student_question(self, user_id: str, question: str) -> str:
        if user_id not in self.active_sessions:
            return "Hi! Let's start a lesson first. What subject would you like to learn about today?"
        
        session = self.active_sessions[user_id]
        
        response = await self.ollama.chat(
            question,
            user_id,
            context={
                'current_lesson': session.get('subtopic', 'General'),
                'subject': session.get('current_subject', SubjectType.MATH).value if session.get('current_subject') else 'math',
                'difficulty': 'beginner'
            }
        )
        
        return response

async def main():
    ergo = ErgoAITutor()
    
    user_profile = UserProfile(
        user_id="student_123",
        name="Alex",
        grade_level=6,
        learning_style=LearningStyle.VISUAL,
        mbti_type="ENFP",
        strengths=["problem_solving"],
        weaknesses=["attention"],
        current_subjects=["mathematics"],
        progress={}
    )
    
    print("🎓 Starting Ergo AI Tutor Demo...")
    
    session = await ergo.start_tutoring_session(
        user_profile, 
        SubjectType.MATH, 
        "Basic Addition"
    )
    
    print(f"Welcome Message: {session['welcome_message']}")
    
    response = await ergo.handle_student_question(
        user_profile.user_id,
        "How do I add two numbers together?"
    )
    print(f"Ergo's Response: {response}")
    
    print("\\n✨ Ergo AI Tutor demo complete!")

if __name__ == "__main__":
    asyncio.run(main())
"""
    
    create_file("ergo_ai_tutor.py", ergo_content)
    
    # Create FastAPI server
    server_content = """import os
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
"""
    
    create_file("server.py", server_content)
    
    # Create other files
    create_file("requirements.txt", """fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
requests>=2.31.0
python-dotenv>=1.0.0
python-multipart>=0.0.6""")
    
    create_file(".env.example", """OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1
OLLAMA_TIMEOUT=30
PORT=8000
HOST=0.0.0.0
RATE_LIMIT_PER_MINUTE=30
ENVIRONMENT=development
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173""")
    
    create_file("README.md", """# Ergo AI Tutor 🎓

An AI-powered educational platform with Ollama backend.

## Quick Setup

1. Install Ollama: https://ollama.com/download
2. Install dependencies: `pip install -r requirements.txt`
3. Setup environment: `copy .env.example .env`
4. Start Ollama: `ollama serve` then `ollama pull llama3.1`
5. Run server: `python server.py`

## API Documentation

- Health Check: http://localhost:8000/health
- API Docs: http://localhost:8000/docs
- System Stats: http://localhost:8000/dev/stats

## Usage

The API will be available at http://localhost:8000

Ready to integrate with your frontend! 🚀
""")
    
    print("\n" + "=" * 50)
    print("🎉 Project setup complete!")
    print("=" * 50)
    print("\n📁 Created files:")
    print("  ✅ ergo_ai_tutor.py     - Core AI system")
    print("  ✅ server.py            - FastAPI server")
    print("  ✅ requirements.txt     - Dependencies")
    print("  ✅ .env.example         - Configuration")
    print("  ✅ README.md            - Documentation")
    
    print("\n🚀 Next steps:")
    print("  1. pip install -r requirements.txt")
    print("  2. copy .env.example .env")
    print("  3. Install Ollama from https://ollama.com/download")
    print("  4. ollama serve && ollama pull llama3.1")
    print("  5. python server.py")
    print("\n🌐 API will be available at: http://localhost:8000")
    print("📖 Documentation at: http://localhost:8000/docs")
    
    print("\n✨ Ready to integrate with your frontend! ✨")

if __name__ == "__main__":
    setup_ergo_project()