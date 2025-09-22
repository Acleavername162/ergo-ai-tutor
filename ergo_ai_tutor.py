import os
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
        history_text = "\n".join([f"Student: {h['user']}\nErgo: {h['assistant']}" for h in history])
        
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
    
    print("\n✨ Ergo AI Tutor demo complete!")

if __name__ == "__main__":
    asyncio.run(main())
