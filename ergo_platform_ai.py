"""
Enhanced Ergo AI Tutor with Platform Integration
Connects to user data, course content, and learning analytics
"""

import asyncio
from typing import Dict, List, Optional
from ergo_ai_tutor import ErgoAITutor, UserProfile, SubjectType, LearningStyle
from platform_bridge import PlatformBridge

class ErgoPlatformAI(ErgoAITutor):
    """Enhanced Ergo AI with full platform integration"""
    
    def __init__(self):
        super().__init__()
        self.platform = PlatformBridge()
    
    async def start_personalized_session(self, user_id: str, subject: str, subtopic: str = None) -> Dict:
        """Start session with full platform context"""
        
        # Get user profile from platform
        user_data = await self.platform.get_user_profile(user_id)
        if not user_data:
            raise Exception(f"User {user_id} not found in platform")
        
        # Get user's learning progress
        progress = await self.platform.get_user_progress(user_id)
        
        # Create enhanced user profile
        user_profile = UserProfile(
            user_id=user_id,
            name=user_data["name"],
            grade_level=user_data["grade_level"],
            learning_style=LearningStyle(user_data["learning_style"]),
            mbti_type="ENFP",  # Default or from platform
            strengths=[],
            weaknesses=[],
            current_subjects=[subject],
            progress={p["course_id"]: {"completion": p["completion_percentage"]} for p in progress}
        )
        
        # Start tutoring session with context
        session = await self.start_tutoring_session(user_profile, SubjectType(subject), subtopic or subject)
        
        # Add platform-specific context
        session["user_progress"] = progress
        session["recommendations"] = await self.platform.get_recommended_content(user_id)
        
        return session
    
    async def handle_platform_question(self, user_id: str, question: str, course_id: str = None) -> str:
        """Handle question with full platform context"""
        
        # Get platform context
        user_data = await self.platform.get_user_profile(user_id)
        course_content = None
        
        if course_id:
            course_content = await self.platform.get_course_content(course_id)
        
        # Build enhanced context
        context = {}
        if course_content:
            context.update({
                "current_course": course_content["title"],
                "course_content": course_content["content"],
                "difficulty": course_content["difficulty"]
            })
        
        if user_data:
            context.update({
                "user_name": user_data["name"],
                "grade_level": user_data["grade_level"],
                "learning_style": user_data["learning_style"]
            })
        
        # Get AI response
        response = await self.handle_student_question(user_id, question)
        
        # Save interaction to platform
        await self.platform.save_chat_interaction(user_id, question, response, context)
        
        return response
    
    async def update_learning_progress(self, user_id: str, course_id: str, lesson_id: str, completion: float):
        """Update user's learning progress"""
        await self.platform.update_user_progress(user_id, course_id, lesson_id, completion)
    
    async def get_learning_analytics(self, user_id: str) -> Dict:
        """Get comprehensive learning analytics"""
        progress = await self.platform.get_user_progress(user_id)
        recommendations = await self.platform.get_recommended_content(user_id)
        
        # Calculate analytics
        total_courses = len(progress)
        completed_courses = len([p for p in progress if p["completion_percentage"] >= 90])
        avg_completion = sum(p["completion_percentage"] for p in progress) / max(len(progress), 1)
        
        return {
            "total_courses": total_courses,
            "completed_courses": completed_courses,
            "completion_rate": completed_courses / max(total_courses, 1) * 100,
            "average_progress": avg_completion,
            "recommendations": recommendations,
            "recent_activity": progress[:5]  # Last 5 accessed
        }
