import os
import asyncio
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

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
    """Core Ollama integration for Ergo AI Tutor"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1"):
        # Prefer env overrides
        self.base_url = os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_BASE_URL", base_url)
        self.model = os.getenv("OLLAMA_MODEL", model)
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", "60"))
        # Generation controls (Ollama expects num_predict, not max_tokens)
        self.num_predict = int(os.getenv("OLLAMA_MAX_TOKENS", "256"))
        self.temperature = float(os.getenv("OLLAMA_TEMP", "0.4"))
        self.conversation_history: Dict[str, List[Dict]] = {}

    async def _post(self, url, json_data, timeout):
        return await asyncio.to_thread(requests.post, url, json=json_data, timeout=timeout)

    def _get_recent_history(self, user_id: str, limit: int = 3) -> List[Dict]:
        if user_id not in self.conversation_history:
            return []
        return self.conversation_history[user_id][-limit:]

    def _update_conversation_history(self, user_id: str, user_msg: str, ai_response: str):
        # Skip storing synthetic/system users if needed
        if user_id.startswith("ergo_"):
            return
        self.conversation_history.setdefault(user_id, []).append({
            "timestamp": datetime.now().isoformat(),
            "user": user_msg,
            "assistant": ai_response
        })
        if len(self.conversation_history[user_id]) > 10:
            self.conversation_history[user_id] = self.conversation_history[user_id][-10:]

    def _build_context_prompt(self, prompt: str, user_id: str, context: Optional[Dict] = None) -> str:
        base_personality = '''You are Ergo, an AI tutor inspired by Albert Einstein.

        STYLE:
        - Patient, encouraging, and curious
        - Explain complex ideas simply with short steps and checks-for-understanding
        - Keep responses concise unless asked for more depth

        FOCUS RULES:
        - The lesson has a subject and subtopic. Stay on this track.
        - If the student asks something off-topic, acknowledge in ONE short sentence,
          then redirect with a guiding question tied to the current subtopic.
        - Offer a clear way to switch if they really want: "Type /switch to change the topic."

        TEACHING RHYTHM:
        1) Principle first ("why it works")
        2) Tiny example -> quick check
        3) Application to a real scenario
        4) Encourage reflection & next step'''

        subject = context.get('subject', '') if context else ''
        subtopic = context.get('current_lesson', '') if context else ''
        difficulty = context.get('difficulty', '') if context else ''
        current_context = ""
        if subject or subtopic or difficulty:
            current_context = f"""
            Current Context:
            - Subject: {subject}
            - Lesson:  {subtopic}
            - Difficulty: {difficulty}
            """.rstrip()

        web_block = ""
        if context and context.get('web_snippets'):
            web_block = f"""
            External Sources (use to verify facts and numbers, but do not overquote):
{context['web_snippets']}
            """.rstrip()

        history = self._get_recent_history(user_id, limit=3)
        history_text = "\n".join([f"Student: {h['user']}\nErgo: {h['assistant']}" for h in history])

        full_prompt = f'''{base_personality}

{current_context}
{web_block}

Recent Conversation:
{history_text}

Student message:
{prompt}

Your job:
- Stay on the current lesson unless the student explicitly asks to switch (e.g. "/switch").
- If off-topic, acknowledge briefly (<=1 sentence), then ask ONE guiding question on "{subtopic}".
- Give a small, correct answer with a check-for-understanding.
- Keep it friendly and motivating.
'''
        return full_prompt

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
                            "temperature": self.temperature,
                            "top_p": 0.9,
                            "num_predict": self.num_predict
                        },
                        "keep_alive": "5m"
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

class ErgoAITutor:
    """Main Ergo AI Tutor orchestrator"""

    def __init__(self):
        self.ollama = OllamaAIBackend()
        self.active_sessions: Dict[str, Dict] = {}
        self.progress_data: Dict[str, Dict] = {}

    async def start_tutoring_session(self, user_profile: UserProfile, subject: SubjectType, subtopic: str) -> Dict:
        session_context = {
            'user_id': user_profile.user_id,
            'current_subject': subject,
            'subtopic': subtopic,
            'progress': 0,
            'start_time': datetime.now()
        }
        # Set/overwrite active session for this user_id
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

    async def handle_student_question(self, user_id: str, question: str, web_snippets: Optional[List[dict]] = None) -> str:
        """Handle any question but keep focus on the current lesson. Optional web_snippets for light grounding."""
        if user_id not in self.active_sessions:
            return "Hi! Let’s start a lesson first. What subject would you like to learn about today?"

        session = self.active_sessions[user_id]
        subtopic = session.get('subtopic', 'General')
        subject = session.get('current_subject', None)
        subject_val = subject.value if (hasattr(subject, "value")) else (subject or "general")

        sources_text = ""
        if web_snippets:
            lines = []
            for i, s in enumerate(web_snippets, start=1):
                title = s.get("title", "")
                link = s.get("link", "")
                snippet = s.get("snippet", "")
                lines.append(f"[{i}] {title} — {snippet} ({link})")
            if lines:
                sources_text = "• " + "\n• ".join(lines)

        response = await self.ollama.chat(
            question,
            user_id,
            context={
                'current_lesson': subtopic,
                'subject': subject_val,
                'difficulty': 'beginner',
                'web_snippets': sources_text
            }
        )
        return response

# Optional demo main (unused in server mode)
if __name__ == "__main__":
    async def _demo():
        tutor = ErgoAITutor()
        user = UserProfile(
            user_id="student_123", name="Alex", grade_level=6,
            learning_style=LearningStyle.VISUAL, mbti_type="ENFP",
            strengths=["problem_solving"], weaknesses=["attention"],
            current_subjects=["mathematics"], progress={}
        )
        sess = await tutor.start_tutoring_session(user, SubjectType.MATH, "Basic Addition")
        print(sess["welcome_message"])
        ans = await tutor.handle_student_question(user.user_id, "How do I add two numbers?")
        print(ans)

    asyncio.run(_demo())
