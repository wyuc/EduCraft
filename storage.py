import os
import json
import logging
import sqlite3
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum, auto
from config import BASE_DIR

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Enum for possible task statuses"""
    CREATED = "created"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    
    def __str__(self):
        return self.value


class ScriptStorage:
    """
    A lightweight SQLite-based storage system for lecture generation scripts.
    This provides an interface for storing and retrieving script data,
    with real-time updates and progress tracking.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ScriptStorage, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, db_path=None):
        """
        Initialize the storage with SQLite connection.
        Uses a singleton pattern to ensure one connection.
        """
        if self._initialized:
            return
        
        # Use default path if not provided
        if db_path is None:
            db_path = BASE_DIR / 'buffer' / 'lecgen.db'
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        
        # Connect to SQLite database
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        
        # Create tables if they don't exist
        self._create_tables()
        
        logger.debug(f"Connected to SQLite database at {db_path}")
        self._initialized = True
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Tasks table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            model_provider TEXT NOT NULL,
            model_name TEXT,
            algo TEXT,
            status TEXT NOT NULL,
            error_message TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            slide_count INTEGER DEFAULT 0,
            slides_completed INTEGER DEFAULT 0
        )
        ''')
       # Slides table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS slides (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            slide_num INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(task_id, slide_num)
        )
        ''')
        
        self.conn.commit()
    
    def create_task(self, file_name: str, model_provider: str, model_name: Optional[str] = None, algo: Optional[str] = None) -> str:
        """
        Create a new lecture generation task and return its ID.
        
        Args:
            file_name: Name of the file
            model_provider: Model provider (e.g., 'claude', 'gemini')
            model_name: Specific model name
            algo: Algorithm used for generation (e.g., 'vlm')
            
        Returns:
            task_id: Unique identifier for the task
        """
        task_id = f"{file_name}_{model_provider}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        current_time = datetime.now().isoformat()
        
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            INSERT INTO tasks 
            (task_id, file_name, model_provider, model_name, algo, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (task_id, file_name, model_provider, model_name, algo, TaskStatus.CREATED.value, current_time, current_time)
        )
        self.conn.commit()
        
        logger.debug(f"Created task {task_id} for {file_name}")
        return task_id
    
    def update_task_status(self, task_id: str, status: TaskStatus, slide_count: Optional[int] = None):
        """
        Update the status of a task.
        
        Args:
            task_id: Unique task identifier
            status: New status (TaskStatus enum)
            slide_count: Total number of slides (if known)
        """
        current_time = datetime.now().isoformat()
        
        cursor = self.conn.cursor()
        if slide_count is not None:
            cursor.execute(
                '''
                UPDATE tasks 
                SET status = ?, updated_at = ?, slide_count = ?
                WHERE task_id = ?
                ''',
                (status.value, current_time, slide_count, task_id)
            )
        else:
            cursor.execute(
                '''
                UPDATE tasks 
                SET status = ?, updated_at = ?
                WHERE task_id = ?
                ''',
                (status.value, current_time, task_id)
            )
        self.conn.commit()
        
        logger.debug(f"Updated task {task_id} status to {status}")
    
    def save_slide(self, task_id: str, slide_num: int, content: str):
        """
        Save a slide's content.
        
        Args:
            task_id: Unique task identifier
            slide_num: Slide number (1-indexed)
            content: Slide content
        """
        current_time = datetime.now().isoformat()
        
        cursor = self.conn.cursor()
        
        # Insert or replace slide
        cursor.execute(
            '''
            INSERT OR REPLACE INTO slides 
            (task_id, slide_num, content, created_at)
            VALUES (?, ?, ?, ?)
            ''',
            (task_id, slide_num, content, current_time)
        )
        
        # Count slides completed
        cursor.execute(
            '''
            SELECT COUNT(*) FROM slides WHERE task_id = ?
            ''',
            (task_id,)
        )
        slides_completed = cursor.fetchone()[0]
        
        # Update task with slides_completed count
        cursor.execute(
            '''
            UPDATE tasks 
            SET updated_at = ?, slides_completed = ?
            WHERE task_id = ?
            ''',
            (current_time, slides_completed, task_id)
        )
        
        self.conn.commit()
        
        logger.debug(f"Saved slide {slide_num} for task {task_id}")
    
    def set_error(self, task_id: str, error_message: str):
        """
        Set an error state for a task.
        
        Args:
            task_id: Unique task identifier
            error_message: Error message
        """
        current_time = datetime.now().isoformat()
        
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            UPDATE tasks 
            SET status = ?, error_message = ?, updated_at = ?
            WHERE task_id = ?
            ''',
            (TaskStatus.ERROR.value, error_message, current_time, task_id)
        )
        self.conn.commit()
        
        logger.error(f"Task {task_id} error: {error_message}")
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a task's details.
        
        Args:
            task_id: Unique task identifier
            
        Returns:
            Task details or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            SELECT * FROM tasks WHERE task_id = ?
            ''',
            (task_id,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        
        return None
    
    def get_slide(self, task_id: str, slide_num: int) -> Optional[Dict[str, Any]]:
        """
        Get a slide's content.
        
        Args:
            task_id: Unique task identifier
            slide_num: Slide number (1-indexed)
            
        Returns:
            Slide details or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            SELECT * FROM slides WHERE task_id = ? AND slide_num = ?
            ''',
            (task_id, slide_num)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        
        return None
    
    def get_all_slides(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Get all slides for a task.
        
        Args:
            task_id: Unique task identifier
            
        Returns:
            List of slides, sorted by slide number
        """
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            SELECT * FROM slides WHERE task_id = ? ORDER BY slide_num
            ''',
            (task_id,)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        Get all tasks.
        
        Returns:
            List of tasks
        """
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            SELECT * FROM tasks ORDER BY created_at DESC
            '''
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows] 

    def get_latest_completed_task(self, file_name: str, algo: str, model_provider: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest completed task matching the given parameters.
        
        Args:
            file_name: Name of the file
            algo: Algorithm used for generation
            model_provider: Model provider name
            
        Returns:
            Latest task details or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            SELECT * FROM tasks 
            WHERE file_name = ? 
            AND algo = ? 
            AND model_provider = ? 
            AND status = ?
            ORDER BY created_at DESC
            LIMIT 1
            ''',
            (file_name, algo, model_provider, TaskStatus.COMPLETED.value)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        
        return None