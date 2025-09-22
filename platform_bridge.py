"""
Platform Bridge - Connects Ollama AI to platform data
Handles user authentication, data access, and platform integration
"""

import os
import requests
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import sqlite3
import asyncio

class PlatformBridge:
    """Bridge between Ollama AI and platform data"""
    
    def __init__(self):
        self.platform_url = os.getenv("PLATFORM_URL", "https://your-platform.com")
        self.api_key = os.getenv("PLATFORM_API_KEY", "")
        self.db_path = os.getenv("PLATFORM_DB_PATH", "./platform.db")
        self.init_database()
    
    def init_database(self):
        """Initialize platform database connection"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for platform data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT,
                name TEXT,
                grade_level INTEGER,
                learning_style TEXT,
                subscription_tier TEXT,
                created_at TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS courses (
                id TEXT PRIMARY KEY,
                title TEXT,
                subject TEXT,
                difficulty TEXT,
                content TEXT,
                prerequisites TEXT,
                created_at TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_progress (
                user_id TEXT,
                course_id TEXT,
                lesson_id TEXT,
                completion_percentage REAL,
                last_accessed TIMESTAMP,
                PRIMARY KEY (user_id, course_id, lesson_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                message TEXT,
                ai_response TEXT,
                context TEXT,
                timestamp TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile and learning data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, email, name, grade_level, learning_style, subscription_tier
            FROM users WHERE id = ?
        ''', (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "user_id": result[0],
                "email": result[1],
                "name": result[2],
                "grade_level": result[3],
                "learning_style": result[4],
                "subscription_tier": result[5]
            }
        return None
    
    async def get_user_progress(self, user_id: str) -> List[Dict]:
        """Get user's learning progress across all courses"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT p.course_id, p.lesson_id, p.completion_percentage, 
                   p.last_accessed, c.title, c.subject
            FROM user_progress p
            JOIN courses c ON p.course_id = c.id
            WHERE p.user_id = ?
            ORDER BY p.last_accessed DESC
        ''', (user_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{
            "course_id": row[0],
            "lesson_id": row[1],
            "completion_percentage": row[2],
            "last_accessed": row[3],
            "course_title": row[4],
            "subject": row[5]
        } for row in results]
    
    async def get_course_content(self, course_id: str) -> Optional[Dict]:
        """Get course content and structure"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, title, subject, difficulty, content, prerequisites
            FROM courses WHERE id = ?
        ''', (course_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "course_id": result[0],
                "title": result[1],
                "subject": result[2],
                "difficulty": result[3],
                "content": json.loads(result[4]) if result[4] else {},
                "prerequisites": result[5].split(',') if result[5] else []
            }
        return None
    
    async def save_chat_interaction(self, user_id: str, message: str, ai_response: str, context: Dict = None):
        """Save chat interaction for learning analytics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        chat_id = f"{user_id}_{datetime.now().timestamp()}"
        
        cursor.execute('''
            INSERT INTO chat_history (id, user_id, message, ai_response, context, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            chat_id,
            user_id,
            message,
            ai_response,
            json.dumps(context) if context else "{}",
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
    
    async def update_user_progress(self, user_id: str, course_id: str, lesson_id: str, completion: float):
        """Update user's progress in a course"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_progress 
            (user_id, course_id, lesson_id, completion_percentage, last_accessed)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, course_id, lesson_id, completion, datetime.now()))
        
        conn.commit()
        conn.close()
    
    async def get_recommended_content(self, user_id: str) -> List[Dict]:
        """Get AI-recommended content based on user progress"""
        user_profile = await self.get_user_profile(user_id)
        user_progress = await self.get_user_progress(user_id)
        
        if not user_profile:
            return []
        
        # Simple recommendation logic - you can enhance this
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find courses matching user's grade level that they haven't completed
        completed_courses = {p["course_id"] for p in user_progress if p["completion_percentage"] >= 90}
        
        cursor.execute('''
            SELECT id, title, subject, difficulty
            FROM courses 
            WHERE id NOT IN ({})
            ORDER BY title
            LIMIT 5
        '''.format(','.join(['?' for _ in completed_courses])), list(completed_courses))
        
        recommendations = cursor.fetchall()
        conn.close()
        
        return [{
            "course_id": row[0],
            "title": row[1],
            "subject": row[2],
            "difficulty": row[3],
            "reason": "Recommended based on your learning progress"
        } for row in recommendations]
