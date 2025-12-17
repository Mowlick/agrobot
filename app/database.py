"""
Database Module for AgroBot
Handles user authentication, storage, and voice query logging
Uses SQLite for local storage
"""

import os
import sqlite3
import hashlib
import uuid
from datetime import datetime
from functools import wraps

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'instance', 'agrobot.db')

class Database:
    """Database handler for AgroBot"""
    
    def __init__(self):
        self.db_path = DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
    
    def _get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                preferred_language TEXT DEFAULT 'en',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # Voice queries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS voice_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                transcription TEXT NOT NULL,
                language TEXT DEFAULT 'en',
                confidence REAL,
                detected_disease TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Chat history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                response TEXT,
                language TEXT DEFAULT 'en',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("[DB] Database initialized successfully")
    
    def _hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, name, email, password, preferred_language='en'):
        """Create a new user"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if email already exists
            cursor.execute('SELECT id FROM users WHERE email = ?', (email.lower(),))
            if cursor.fetchone():
                conn.close()
                return {'success': False, 'error': 'Email already registered'}
            
            user_id = str(uuid.uuid4())
            password_hash = self._hash_password(password)
            
            cursor.execute('''
                INSERT INTO users (id, name, email, password_hash, preferred_language)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, name, email.lower(), password_hash, preferred_language))
            
            conn.commit()
            conn.close()
            
            print(f"[DB] User created: {email}")
            return {
                'success': True,
                'user_id': user_id,
                'name': name,
                'email': email.lower()
            }
            
        except Exception as e:
            print(f"[DB] Error creating user: {e}")
            return {'success': False, 'error': str(e)}
    
    def authenticate_user(self, email, password):
        """Authenticate a user"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            password_hash = self._hash_password(password)
            
            cursor.execute('''
                SELECT id, name, email, preferred_language
                FROM users
                WHERE email = ? AND password_hash = ?
            ''', (email.lower(), password_hash))
            
            user = cursor.fetchone()
            
            if user:
                # Update last login
                cursor.execute(
                    'UPDATE users SET last_login = ? WHERE id = ?',
                    (datetime.now(), user['id'])
                )
                conn.commit()
                conn.close()
                
                print(f"[DB] User authenticated: {email}")
                return {
                    'success': True,
                    'user_id': user['id'],
                    'name': user['name'],
                    'email': user['email'],
                    'preferred_language': user['preferred_language']
                }
            else:
                conn.close()
                return {'success': False, 'error': 'Invalid email or password'}
                
        except Exception as e:
            print(f"[DB] Auth error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_user(self, user_id):
        """Get user by ID"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, email, preferred_language, created_at, last_login
                FROM users
                WHERE id = ?
            ''', (user_id,))
            
            user = cursor.fetchone()
            conn.close()
            
            if user:
                return {
                    'id': user['id'],
                    'name': user['name'],
                    'email': user['email'],
                    'preferred_language': user['preferred_language'],
                    'created_at': user['created_at'],
                    'last_login': user['last_login']
                }
            return {'name': 'User', 'email': '', 'preferred_language': 'en'}
            
        except Exception as e:
            print(f"[DB] Error getting user: {e}")
            return {'name': 'User', 'email': '', 'preferred_language': 'en'}
    
    def update_user_language(self, user_id, language):
        """Update user's preferred language"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                'UPDATE users SET preferred_language = ? WHERE id = ?',
                (language, user_id)
            )
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"[DB] Error updating language: {e}")
            return False
    
    def log_voice_query(self, user_id, transcription, language, confidence, detected_disease=None):
        """Log a voice query"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO voice_queries (user_id, transcription, language, confidence, detected_disease)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, transcription, language, confidence, detected_disease))
            
            conn.commit()
            conn.close()
            
            print(f"[DB] Voice query logged for user: {user_id}")
            return True
            
        except Exception as e:
            print(f"[DB] Error logging voice query: {e}")
            return False
    
    def get_user_voice_history(self, user_id, limit=10):
        """Get user's voice query history"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT transcription, language, confidence, detected_disease, created_at
                FROM voice_queries
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (user_id, limit))
            
            queries = cursor.fetchall()
            conn.close()
            
            return {
                'success': True,
                'queries': [
                    {
                        'transcription': q['transcription'],
                        'language': q['language'],
                        'confidence': q['confidence'],
                        'detected_disease': q['detected_disease'],
                        'created_at': q['created_at']
                    }
                    for q in queries
                ]
            }
            
        except Exception as e:
            print(f"[DB] Error getting voice history: {e}")
            return {'success': False, 'error': str(e)}
    
    def log_chat(self, user_id, message, response, language='en'):
        """Log a chat message"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO chat_history (user_id, message, response, language)
                VALUES (?, ?, ?, ?)
            ''', (user_id, message, response, language))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"[DB] Error logging chat: {e}")
            return False


# Create global database instance
db = Database()

# Export convenience functions
def create_user(name, email, password, preferred_language='en'):
    """Create a new user"""
    return db.create_user(name, email, password, preferred_language)

def authenticate_user(email, password):
    """Authenticate a user"""
    return db.authenticate_user(email, password)

def get_user(user_id):
    """Get user by ID"""
    return db.get_user(user_id)

def log_voice_query(user_id, transcription, language, confidence, detected_disease=None):
    """Log a voice query"""
    return db.log_voice_query(user_id, transcription, language, confidence, detected_disease)
