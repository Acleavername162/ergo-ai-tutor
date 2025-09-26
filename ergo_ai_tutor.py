from __future__ import annotations

import os
import asyncio
import json
import requests
import re
import string
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
from pathlib import Path

# --- System prompt loading ---
_DEFAULT_PROMPT = (
    "You are Ergo, an AI tutor inspired by Albert Einstein. "
    "Keep answers brief (2–3 sentences), step-by-step, encouraging, and always end with a quick check question. "
    "Stay on the current subject/subtopic; gently redirect off-topic and mention /switch."
)

_PROMPT_CACHE: Optional[str] = None

def get_system_prompt() -> str:
    """Load prompt from file or env once; fallback to a safe default."""
    global _PROMPT_CACHE
    if _PROMPT_CACHE:
        return _PROMPT_CACHE

    override = os.getenv("ERGO_SYSTEM_PROMPT_PATH")
    search_order = []
    if override:
        search_order.append(Path(override))
    search_order.append(Path(__file__).with_name("ergo_system_prompt.txt"))
    search_order.append(Path.cwd() / "ergo_system_prompt.txt")

    for p in search_order:
        try:
            if p.exists():
                txt = p.read_text(encoding="utf-8").strip()
                if txt:
                    _PROMPT_CACHE = txt
                    return _PROMPT_CACHE
        except Exception:
            pass

    _PROMPT_CACHE = _DEFAULT_PROMPT
    return _PROMPT_CACHE


# -----------------------------
# Domain models
# -----------------------------

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
    content: Dict[str, Any]
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


# -----------------------------
# Ollama backend
# -----------------------------

class OllamaAIBackend:
    """Core Ollama integration for Ergo AI Tutor."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "llama3.1:8b-instruct-q4_K_M",
    ) -> None:
        self.base_url = os.getenv("OLLAMA_URL", os.getenv("OLLAMA_BASE_URL", base_url)).rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", model)

        # Tunables (safe CPU defaults)
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", "120"))
        self.max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", "160"))
        self.temperature = float(os.getenv("OLLAMA_TEMP", "0.6"))
        self.top_p = float(os.getenv("OLLAMA_TOP_P", "0.9"))

        # conversation_history: user_id -> List[{timestamp, question, answer}]
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}

    async def _post(self, url: str, json_data: Dict[str, Any], timeout: int):
        """Run requests.post in a thread to avoid blocking the event loop."""
        return await asyncio.to_thread(requests.post, url, json=json_data, timeout=timeout)

    def _build_context_prompt(self, prompt: str, user_id: str, context: Optional[Dict[str, Any]]) -> str:
        base_personality = get_system_prompt()

        current_context = ""
        if context:
            current_context = (
                "CURRENT LESSON CONTEXT:\n"
                f"- Subject: {context.get('subject','')}\n"
                f"- Subtopic: {context.get('subtopic','')}\n"
                f"- Lesson: {context.get('current_lesson','')}\n"
                "- Keep responses focused on this topic\n"
            )

        history = self._get_recent_history(user_id, limit=3)
        history_text = "\n".join([f"Student: {h['question']}\nErgo: {h['answer']}" for h in history]) or "(no recent turns)"

        full_prompt = (
            f"{base_personality}\n\n"
            f"{current_context}\n"
            f"RECENT CONVERSATION:\n{history_text}\n\n"
            f"CURRENT STUDENT INPUT: {prompt}\n\n"
            f"Respond as Ergo with a brief, focused answer:"
        )
        return full_prompt

    def _update_conversation_history(self, user_id: str, question: str, answer: str) -> None:
        self.conversation_history.setdefault(user_id, []).append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
        })
        if len(self.conversation_history[user_id]) > 3:
            self.conversation_history[user_id] = self.conversation_history[user_id][-3:]

    def _get_recent_history(self, user_id: str, limit: int = 3) -> List[Dict[str, str]]:
        return self.conversation_history.get(user_id, [])[-limit:]

    async def chat(self, prompt: str, user_id: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Main chat interface with context awareness, retries, and CPU-friendly options."""
        full_prompt = self._build_context_prompt(prompt, user_id, context)

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
            },
            "keep_alive": "10m",
        }

        for attempt in range(3):
            try:
                response = await self._post(f"{self.base_url}/api/generate", payload, self.timeout)
                if response.status_code == 200:
                    result = response.json()
                    ai_response = (result.get("response") or "").strip()
                    if not ai_response:
                        ai_response = "I’m here—try asking that again with a bit more detail."
                    self._update_conversation_history(user_id, prompt, ai_response)
                    return ai_response
            except Exception:
                await asyncio.sleep(0.5 * (2 ** attempt))

        return "I'm having a small hiccup or trouble connecting. Please try again, or type /switch <subject> : <subtopic>."


# -----------------------------
# Tutor orchestrator
# -----------------------------

class ErgoAITutor:
    """Main Ergo AI Tutor class with input router and robust handling."""

    def __init__(self) -> None:
        self.ai_backend = OllamaAIBackend()
        self.active_sessions: Dict[str, TutoringSession] = {}

        # Models to pre-warm so first real request isn't slow
        self.warm_models: List[str] = [
            m.strip() for m in os.getenv(
                "WARM_MODELS",
                "qwen2.5:7b-instruct-q4_0,llama3.1:8b-instruct-q4_K_M"
            ).split(",") if m.strip()
        ]

    # ---------- used by server.py on startup ----------
    async def warm_up(self) -> None:
        """Pre-hit Ollama with tiny prompts for each warm model. Best-effort, non-fatal."""
        for m in self.warm_models:
            try:
                payload = {
                    "model": m,
                    "prompt": "ready",
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {"num_predict": 8, "temperature": 0.0},
                }
                await self.ai_backend._post(f"{self.ai_backend.base_url}/api/generate", payload, timeout=30)
            except Exception:
                # ignore warmup failures; service should still run
                pass

    # ---------- compatibility wrapper for /tutor/question ----------
    async def get_answer(self, question: str, model: Optional[str] = None, user_id: str = "anonymous") -> str:
        """
        Optionally override the model for this call, then delegate to the backend chat
        while restoring the previous model afterwards.
        """
        prev_model = self.ai_backend.model
        try:
            if model:
                self.ai_backend.model = model
            # If you later add session context, pass it here:
            return await self.ai_backend.chat(question, user_id, context=None)
        finally:
            self.ai_backend.model = prev_model

    # ---------- helpers ----------

    def _is_short(self, text: str) -> bool:
        tokens = (text or "").strip().split()
        return len(tokens) <= 2 and "?" not in (text or "")

    def _normalize_tokens(self, text: str) -> set[str]:
        text = (text or "").lower().translate(str.maketrans("", "", string.punctuation))
        return set(text.split())

    def _is_off_topic(self, text: str, subject: str, subtopic: str) -> bool:
        text_tokens = self._normalize_tokens(text)
        subject_tokens = self._normalize_tokens(subject)
        subtopic_tokens = self._normalize_tokens(subtopic)
        return len(text_tokens & (subject_tokens | subtopic_tokens)) == 0

    def _subject_to_str(self, subject: Union[SubjectType, str]) -> str:
        if isinstance(subject, SubjectType):
            return subject.value
        return str(subject).strip().lower()

    def _handle_command(self, command: str, user_id: str) -> Dict[str, Any]:
        command = (command or "").strip()

        if command == "/help":
            return {
                "success": True,
                "answer": (
                    "Available commands:\n"
                    "• /switch <subject> : <subtopic> – Change your topic\n"
                    "• /help – Show this help\n\n"
                    "Example: /switch mathematics : fractions"
                ),
            }

        if command.startswith("/switch "):
            parts = command[8:].split(" : ")
            if len(parts) != 2:
                return {
                    "success": True,
                    "answer": (
                        "Use format: /switch <subject> : <subtopic>\n"
                        "Example: /switch mathematics : fractions"
                    ),
                }
            subject = parts[0].strip()
            subtopic = parts[1].strip()

            self.active_sessions[user_id] = TutoringSession(
                user_id=user_id,
                subject=subject,
                subtopic=subtopic,
                start_time=datetime.now(),
                last_interaction=datetime.now(),
            )
            return {
                "success": True,
                "answer": f"Switched to {subject} : {subtopic}. What would you like to learn?",
            }

        return {"success": True, "answer": "Unknown command. Type /help to see available commands."}

    # ---------- public API ----------

    async def start_tutoring_session(
        self,
        user_profile: UserProfile,
        subject: Union[SubjectType, str],
        subtopic: str,
    ) -> Dict[str, Any]:
        """Start a new tutoring session - replaces any existing session for this user."""
        subject_str = self._subject_to_str(subject)
        self.active_sessions[user_profile.user_id] = TutoringSession(
            user_id=user_profile.user_id,
            subject=subject_str,
            subtopic=subtopic,
            start_time=datetime.now(),
            last_interaction=datetime.now(),
        )

        welcome_prompt = (
            f"Welcome {user_profile.name} to learning about {subtopic} in {subject_str}! "
            f"Introduce yourself as Ergo and ask what they'd like to start with."
        )
        try:
            welcome_message = await self.ai_backend.chat(
                welcome_prompt,
                user_profile.user_id,
                context={
                    "current_lesson": subtopic,
                    "subject": subject_str,
                    "subtopic": subtopic,
                },
            )
        except Exception:
            welcome_message = (
                f"Hello {user_profile.name}! I'm Ergo, your AI tutor. "
                f"Let's explore {subtopic} together. What would you like to start with?"
            )

        session = self.active_sessions[user_profile.user_id]
        session_context = {
            "session_id": user_profile.user_id,
            "current_subject": session.subject,
            "current_subtopic": session.subtopic,
            "session_start": session.start_time.isoformat(),
            "last_activity": session.last_interaction.isoformat(),
            "user_profile": self._profile_to_public(user_profile),
        }
        return {"session_context": session_context, "welcome_message": welcome_message}

    async def handle_student_question(self, user_id: str, question: str) -> Dict[str, Any]:
        """Handle student questions with input router - never throws exceptions."""
        try:
            if user_id not in self.active_sessions:
                return {
                    "success": True,
                    "answer": (
                        "Hi! I'm Ergo, your AI tutor. Let's get you started.\n\n"
                        "Try: /switch mathematics : addition\n"
                        "Or: /switch science : atoms\n\n"
                        "What subject interests you?"
                    ),
                }

            session = self.active_sessions[user_id]
            session.last_interaction = datetime.now()

            question = (question or "").strip()
            if not question:
                return {
                    "success": True,
                    "answer": (
                        f"Tell me something specific about {session.subtopic}, "
                        "or type /help for options."
                    ),
                }

            if question.startswith("/"):
                return self._handle_command(question, user_id)

            if self._is_short(question):
                return {
                    "success": True,
                    "answer": (
                        f"Let’s dive deeper into {session.subtopic}. "
                        "Ask a specific question or type /switch to change topics. "
                        "What would you like to practice?"
                    ),
                }

            if self._is_off_topic(question, session.subject, session.subtopic):
                return {
                    "success": True,
                    "answer": (
                        f"That's interesting! Right now we're focusing on {session.subtopic} in {session.subject}. "
                        f"Let’s keep focusing on this topic for now, or type /switch <subject> : <subtopic> to change topics."
                    ),
                }

            try:
                response = await self.ai_backend.chat(
                    question,
                    user_id,
                    context={
                        "current_lesson": session.subtopic,
                        "subject": session.subject,
                        "subtopic": session.subtopic,
                    },
                )
                return {"success": True, "answer": response}
            except Exception:
                return {
                    "success": True,
                    "answer": "I'm experiencing a hiccup or trouble connecting. Try again, or type /switch <subject> : <subtopic>.",
                }
        except Exception:
            return {
                "success": True,
                "answer": "I'm experiencing a hiccup or trouble connecting. Try again, or type /switch <subject> : <subtopic>.",
            }

    def get_session_status(self, user_id: str) -> Dict[str, Any]:
        """Small helper your tests rely on."""
        s = self.active_sessions.get(user_id)
        if not s:
            return {
                "current_subject": None,
                "current_subtopic": None,
                "session_start": None,
                "last_activity": None,
            }
        return {
            "current_subject": s.subject,
            "current_subtopic": s.subtopic,
            "session_start": s.start_time.isoformat(),
            "last_activity": s.last_interaction.isoformat(),
        }

    async def generate_practice_flashcard(self, user_id: str) -> Optional[FlashCard]:
        """Generate a practice flashcard for current lesson (placeholder)."""
        if user_id not in self.active_sessions:
            return None
        session = self.active_sessions[user_id]
        return FlashCard(
            question=f"What is an important concept in {session.subtopic}?",
            answer=f"A key concept in {session.subtopic}",
            explanation=f"This relates to {session.subject} and specifically {session.subtopic}",
            difficulty=DifficultyLevel.BEGINNER,
            subject=SubjectType.MATH,
            hints=["Think about the basics", "Consider the fundamentals"],
        )

    # ---------- utilities ----------

    def _profile_to_public(self, profile: UserProfile) -> Dict[str, Any]:
        """Serialize UserProfile with enum values expanded."""
        data = asdict(profile)
        data["learning_style"] = profile.learning_style.value
        return data


__all__ = [
    "LearningStyle",
    "DifficultyLevel",
    "SubjectType",
    "UserProfile",
    "Lesson",
    "FlashCard",
    "TutoringSession",
    "OllamaAIBackend",
    "ErgoAITutor",
]
