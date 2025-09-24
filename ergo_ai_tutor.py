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

@dataclass
class Lesson:
    lesson_id: str
    subject: SubjectType
    subtopic: str
    title: str
    grade_level: int
    difficulty: DifficultyLevel
    content: Dict
    prerequisites: List[str]
    learning_objectives: List[str]

@dataclass
class FlashCard:
    question: str
    answer: str
    explanation: str
    difficulty: DifficultyLevel
    subject: SubjectType
    hints: List[str]

@dataclass
class TutoringSession:
    user_id: str
    subject: str
    subtopic: str
    start_time: datetime
    last_interaction: datetime

class OllamaAIBackend:
    """Core Ollama integration for Ergo AI Tutor"""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "llama3.1:8b-instruct-q4_K_M"):
        # prefer env values
        self.base_url = os.getenv("OLLAMA_URL", os.getenv("OLLAMA_BASE_URL", base_url))
        self.model = os.getenv("OLLAMA_MODEL", model)
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", "60"))
        self.max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", "500"))  # -> num_predict
        self.temperature = float(os.getenv("OLLAMA_TEMP", "0.7"))
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}

    def _extract_first_json_block(self, text: str) -> Optional[dict]:
        """Extract JSON from LLM response if present"""
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        blob = m.group(1) if m else None
        if not blob:
            start, end = text.find("{"), text.rfind("}")
            blob = text[start:end+1] if (start != -1 and end > start) else None
        if not blob:
            return None
        try:
            return json.loads(blob)
        except Exception:
            return None

    async def _post(self, url, json_data, timeout):
        """Non-blocking HTTP POST wrapper"""
        return await asyncio.to_thread(requests.post, url, json=json_data, timeout=timeout)

    async def chat(self, prompt: str, user_id: str, context: Optional[Dict] = None) -> str:
        """Context-aware chat with retries"""
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
                            "temperature": self.temperature,
                            "top_p": 0.9,
                            "num_predict": self.max_tokens
                        },
                        "keep_alive": "5m"
                    },
                    self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    ai_response = result.get("response", "")
                    self._update_conversation_history(user_id, prompt, ai_response)
                    return ai_response
                else:
                    if attempt == 2:
                        return "I'm experiencing a hiccup. Try asking again or type /switch <subject> : <subtopic>."
            except Exception:
                if attempt == 2:
                    return "I'm experiencing a hiccup. Try asking again or type /switch <subject> : <subtopic>."
                await asyncio.sleep(0.5 * (2 ** attempt))

        return "I'm experiencing a hiccup. Try asking again or type /switch <subject> : <subtopic>."

    def _build_context_prompt(self, prompt: str, user_id: str, context: Optional[Dict] = None) -> str:
        base_personality = '''You are Ergo, an AI tutor inspired by Albert Einstein.

TEACHING STYLE:
- Patient and encouraging, never condescending
- Brief responses (2–3 sentences max) – no long monologues
- Step-by-step explanations using simple language
- Always check for understanding with a quick question

FOCUS DISCIPLINE:
- Stay laser-focused on the current subject and subtopic
- If asked off-topic, acknowledge briefly then redirect: "That's interesting! Let's focus on [current subtopic] for now. Type /switch to change."
- Prioritize depth over breadth in the current lesson

TEACHING RHYTHM:
1) Explain the core principle simply
2) Give a tiny, concrete example
3) Ask a check-for-understanding question
4) Connect to real-world application when relevant

SAFETY & BOUNDARIES:
- Refuse harmful/illicit requests politely
- Never reveal chain-of-thought
- Stay educational and age-appropriate'''

        current_context = ""
        if context:
            current_lesson = context.get("current_lesson", "")
            subject = context.get("subject", "")
            subtopic = context.get("subtopic", "")
            current_context = f"""
CURRENT LESSON CONTEXT:
- Subject: {subject}
- Subtopic: {subtopic}
- Lesson: {current_lesson}
- Keep responses focused on this topic"""

        history = self._get_recent_history(user_id, limit=3)
        history_text = "\n".join([f"Student: {h['user']}\nErgo: {h['assistant']}" for h in history])

        full_prompt = f"""{base_personality}

{current_context}

RECENT CONVERSATION:
{history_text}

CURRENT STUDENT INPUT: {prompt}

Respond as Ergo with a brief, focused answer:"""
        return full_prompt

    def _update_conversation_history(self, user_id: str, user_msg: str, ai_response: str):
        if user_id.startswith("ergo_"):
            return
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        self.conversation_history[user_id].append({
            "timestamp": datetime.now().isoformat(),
            "user": user_msg,
            "assistant": ai_response
        })
        if len(self.conversation_history[user_id]) > 3:
            self.conversation_history[user_id] = self.conversation_history[user_id][-3:]

    def _get_recent_history(self, user_id: str, limit: int = 3) -> List[Dict]:
        if user_id not in self.conversation_history:
            return []
        return self.conversation_history[user_id][-limit:]

class ErgoAITutor:
    """Main Ergo AI Tutor class with input router and robust handling"""

    def __init__(self):
        self.ollama = OllamaAIBackend()
        self.active_sessions: Dict[str, TutoringSession] = {}

    def _is_short(self, text: str) -> bool:
        tokens = text.strip().split()
        return len(tokens) <= 2 and "?" not in text

    def _is_off_topic(self, text: str, subject: str, subtopic: str) -> bool:
        """Conservative off-topic check:
           - Never mark short (≤2 tokens) as off-topic.
           - Otherwise, require zero overlap with subject/subtopic tokens.
        """
        t = set(re.findall(r"\w+", text.lower()))
        if len(t) <= 2:
            return False
        s = set(re.findall(r"\w+", str(subject).lower()))
        u = set(re.findall(r"\w+", str(subtopic).lower()))
        return len(t & (s | u)) == 0

    def _handle_command(self, command: str, user_id: str) -> Dict[str, Any]:
        command = command.strip()

        if command == "/help":
            return {
                "success": True,
                "answer": "Available commands:\n• /switch <subject> : <subtopic>\n• /help\n\nExample: /switch mathematics : fractions"
            }

        if command.startswith("/switch "):
            try:
                parts = command[8:].split(" : ")
                if len(parts) != 2:
                    return {
                        "success": True,
                        "answer": "Use: /switch <subject> : <subtopic>\nExample: /switch mathematics : fractions"
                    }
                subject = parts[0].strip()
                subtopic = parts[1].strip()
                self.active_sessions[user_id] = TutoringSession(
                    user_id=user_id,
                    subject=subject,
                    subtopic=subtopic,
                    start_time=datetime.now(),
                    last_interaction=datetime.now()
                )
                return {
                    "success": True,
                    "answer": f"Switched to {subject} : {subtopic}. What would you like to learn about this topic?"
                }
            except Exception:
                return {
                    "success": True,
                    "answer": "Use: /switch <subject> : <subtopic>\nExample: /switch mathematics : fractions"
                }

        return {
            "success": True,
            "answer": "Unknown command. Type /help to see available commands."
        }

    async def start_tutoring_session(self, user_profile: UserProfile, subject, subtopic: str) -> Dict:
        """Start a new session (replaces any existing session for this user)."""
        subj_value = subject.value if isinstance(subject, SubjectType) else str(subject)

        self.active_sessions[user_profile.user_id] = TutoringSession(
            user_id=user_profile.user_id,
            subject=subj_value,
            subtopic=subtopic,
            start_time=datetime.now(),
            last_interaction=datetime.now()
        )

        welcome_prompt = (
            f"Welcome {user_profile.name} to learning about {subtopic} in {subj_value}! "
            f"Introduce yourself as Ergo and ask what they'd like to start with."
        )

        try:
            welcome_message = await self.ollama.chat(
                welcome_prompt,
                user_profile.user_id,
                context={
                    "current_lesson": subtopic,
                    "subject": subj_value,
                    "subtopic": subtopic
                }
            )
        except Exception:
            welcome_message = (
                f"Hello {user_profile.name}! I'm Ergo, your AI tutor. "
                f"Let's explore {subtopic} in {subj_value}! What would you like to learn first?"
            )

        return {
            "session_context": {
                "user_id": user_profile.user_id,
                "current_subject": subj_value,
                "subtopic": subtopic,
                "progress": 0,
                "start_time": datetime.now().isoformat()
            },
            "welcome_message": welcome_message,
            "session_id": user_profile.user_id
        }

    async def handle_student_question(self, user_id: str, question: str) -> Dict[str, Any]:
        """Handle student inputs robustly; never throws."""
        try:
            if user_id not in self.active_sessions:
                return {
                    "success": True,
                    "answer": "Hi! I'm Ergo, your AI tutor.\nTry: /switch mathematics : addition  or  /switch science : atoms\nWhat subject interests you?"
                }

            session = self.active_sessions[user_id]
            session.last_interaction = datetime.now()

            question = question.strip()

            if question.startswith("/"):
                return self._handle_command(question, user_id)

            if self._is_short(question):
                return {
                    "success": True,
                    "answer": f"Let's dive deeper into {session.subtopic}! Try a specific question, or type /switch to change topics. What would you like to practice?"
                }

            if self._is_off_topic(question, session.subject, session.subtopic):
                return {
                    "success": True,
                    "answer": f"Interesting! We're focusing on {session.subtopic} in {session.subject}. "
                              f"Ask something about this, or type /switch <subject> : <subtopic> to change."
                }

            try:
                response = await self.ollama.chat(
                    question,
                    user_id,
                    context={
                        "current_lesson": session.subtopic,
                        "subject": session.subject,
                        "subtopic": session.subtopic
                    }
                )
                return {"success": True, "answer": response}
            except Exception:
                return {"success": True, "answer": "I'm experiencing a hiccup. Try again or /switch <subject> : <subtopic>."}

        except Exception:
            return {"success": True, "answer": "I'm experiencing a hiccup. Try again or /switch <subject> : <subtopic>."}

    async def generate_practice_flashcard(self, user_id: str) -> Optional[FlashCard]:
        if user_id not in self.active_sessions:
            return None
        session = self.active_sessions[user_id]
        return FlashCard(
            question=f"What is an important concept in {session.subtopic}?",
            answer=f"A key concept in {session.subtopic}",
            explanation=f"This relates to {session.subject} and specifically {session.subtopic}",
            difficulty=DifficultyLevel.BEGINNER,
            subject=SubjectType.MATH,
            hints=["Think about the basics", "Consider the fundamentals"]
        )

# Example usage
async def main():
    ergo = ErgoAITutor()
    user_profile = UserProfile(
        user_id="student_123",
        name="Alex",
        grade_level=6,
        learning_style=LearningStyle.VISUAL,
        mbti_type="ENFP",
        strengths=["problem_solving", "creativity"],
        weaknesses=["attention_to_detail"],
        current_subjects=["mathematics"],
        progress={}
    )

    print("Starting Enhanced Ergo AI Tutor Demo...")
    session = await ergo.start_tutoring_session(user_profile, SubjectType.MATH, "Basic Addition")
    print(f"Welcome Message: {session['welcome_message']}")

    test_inputs = ["hi", "What's your favorite color?", "/help", "/switch science : atoms", "How do atoms work?"]
    for test_input in test_inputs:
        print(f"\nStudent: {test_input}")
        response = await ergo.handle_student_question(user_profile.user_id, test_input)
        print(f"Ergo: {response['answer']}")

    print("\nEnhanced Ergo AI Tutor demo complete!")

if __name__ == "__main__":
    asyncio.run(main())
