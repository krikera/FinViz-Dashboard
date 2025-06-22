"""
Database Layer for FinViz Dashboard
Supports SQLite and PostgreSQL with user management and financial data models
"""

import sqlite3
import pandas as pd
import hashlib
import secrets
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

try:
    from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
    from sqlalchemy.orm import sessionmaker, declarative_base, relationship
    from sqlalchemy.exc import SQLAlchemyError
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    print("SQLAlchemy not available. Using SQLite with pandas.")

from config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class User:
    """User data model"""
    id: Optional[int] = None
    username: str = ""
    email: str = ""
    password_hash: str = ""
    salt: str = ""
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    preferences: Dict[str, Any] = None

    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}

@dataclass
class FinancialTransaction:
    """Financial transaction data model"""
    id: Optional[int] = None
    user_id: int = 0
    date: datetime = None
    description: str = ""
    amount: float = 0.0
    transaction_type: str = "withdrawal"  # withdrawal, deposit
    category: str = "uncategorized"
    subcategory: str = ""
    balance: float = 0.0
    account_name: str = "default"
    tags: str = ""
    notes: str = ""
    is_recurring: bool = False
    merchant: str = ""
    location: str = ""
    sentiment_score: Optional[float] = None
    anomaly_score: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class DatabaseManager:
    """Comprehensive database manager with user support"""
    
    def __init__(self):
        self.config = get_config()
        self.engine = None
        self.session_factory = None
        self.connection = None
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and tables"""
        try:
            if SQLALCHEMY_AVAILABLE and self.config.database.type != "sqlite":
                self._init_sqlalchemy()
            else:
                self._init_sqlite()
                
            self._create_tables()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _init_sqlalchemy(self):
        """Initialize SQLAlchemy engine"""
        database_url = self.config.get_database_url()
        self.engine = create_engine(
            database_url,
            pool_size=self.config.database.pool_size,
            max_overflow=self.config.database.max_overflow,
            echo=False
        )
        self.session_factory = sessionmaker(bind=self.engine)
    
    def _init_sqlite(self):
        """Initialize SQLite connection"""
        db_path = Path(self.config.database.database)
        db_path.parent.mkdir(exist_ok=True)
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
    
    def _create_tables(self):
        """Create database tables"""
        if SQLALCHEMY_AVAILABLE and self.engine:
            self._create_sqlalchemy_tables()
        else:
            self._create_sqlite_tables()
    
    def _create_sqlalchemy_tables(self):
        """Create tables using SQLAlchemy"""
        metadata = MetaData()
        
        # Users table
        users_table = Table(
            'users', metadata,
            Column('id', Integer, primary_key=True),
            Column('username', String(80), unique=True, nullable=False),
            Column('email', String(120), unique=True, nullable=False),
            Column('password_hash', String(255), nullable=False),
            Column('salt', String(32), nullable=False),
            Column('created_at', DateTime, default=datetime.utcnow),
            Column('last_login', DateTime),
            Column('is_active', Boolean, default=True),
            Column('failed_login_attempts', Integer, default=0),
            Column('locked_until', DateTime),
            Column('preferences', Text)
        )
        
        # Financial transactions table
        transactions_table = Table(
            'financial_transactions', metadata,
            Column('id', Integer, primary_key=True),
            Column('user_id', Integer, ForeignKey('users.id'), nullable=False),
            Column('date', DateTime, nullable=False),
            Column('description', Text, nullable=False),
            Column('amount', Float, nullable=False),
            Column('transaction_type', String(20), nullable=False),
            Column('category', String(50)),
            Column('subcategory', String(50)),
            Column('balance', Float),
            Column('account_name', String(100)),
            Column('tags', Text),
            Column('notes', Text),
            Column('is_recurring', Boolean, default=False),
            Column('merchant', String(100)),
            Column('location', String(100)),
            Column('sentiment_score', Float),
            Column('anomaly_score', Float),
            Column('created_at', DateTime, default=datetime.utcnow),
            Column('updated_at', DateTime, default=datetime.utcnow)
        )
        
        metadata.create_all(self.engine)
    
    def _create_sqlite_tables(self):
        """Create tables using SQLite"""
        cursor = self.connection.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                failed_login_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP,
                preferences TEXT
            )
        ''')
        
        # Financial transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS financial_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                date TIMESTAMP NOT NULL,
                description TEXT NOT NULL,
                amount REAL NOT NULL,
                transaction_type TEXT NOT NULL,
                category TEXT,
                subcategory TEXT,
                balance REAL,
                account_name TEXT,
                tags TEXT,
                notes TEXT,
                is_recurring BOOLEAN DEFAULT 0,
                merchant TEXT,
                location TEXT,
                sentiment_score REAL,
                anomaly_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON financial_transactions(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON financial_transactions(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON financial_transactions(category)')
        
        self.connection.commit()
    
    # User Management Methods
    def create_user(self, username: str, email: str, password: str) -> Optional[int]:
        """Create a new user"""
        try:
            # Generate salt and hash password
            salt = secrets.token_hex(16)
            password_hash = self._hash_password(password, salt)
            
            user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                salt=salt,
                created_at=datetime.utcnow()
            )
            
            if SQLALCHEMY_AVAILABLE and self.engine:
                return self._create_user_sqlalchemy(user)
            else:
                return self._create_user_sqlite(user)
                
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user credentials"""
        try:
            user = self.get_user_by_username(username)
            if not user:
                return None
            
            # Check if account is locked
            if user.locked_until and user.locked_until > datetime.utcnow():
                return None
            
            # Verify password
            if self._verify_password(password, user.password_hash, user.salt):
                # Reset failed attempts and update last login
                self._update_user_login(user.id)
                return user
            else:
                # Increment failed attempts
                self._increment_failed_attempts(user.id)
                return None
                
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            if SQLALCHEMY_AVAILABLE and self.engine:
                return self._get_user_by_username_sqlalchemy(username)
            else:
                return self._get_user_by_username_sqlite(username)
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        try:
            if SQLALCHEMY_AVAILABLE and self.engine:
                return self._get_user_by_id_sqlalchemy(user_id)
            else:
                return self._get_user_by_id_sqlite(user_id)
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    def update_user_preferences(self, user_id: int, preferences: Dict[str, Any]) -> bool:
        """Update user preferences"""
        try:
            preferences_json = json.dumps(preferences)
            
            if SQLALCHEMY_AVAILABLE and self.engine:
                with self.engine.connect() as conn:
                    conn.execute(
                        text("UPDATE users SET preferences = :prefs WHERE id = :user_id"),
                        {"prefs": preferences_json, "user_id": user_id}
                    )
                    conn.commit()
            else:
                cursor = self.connection.cursor()
                cursor.execute(
                    "UPDATE users SET preferences = ? WHERE id = ?",
                    (preferences_json, user_id)
                )
                self.connection.commit()
            
            return True
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            return False
    
    # Transaction Management Methods
    def add_transaction(self, transaction: FinancialTransaction) -> Optional[int]:
        """Add a financial transaction"""
        try:
            if SQLALCHEMY_AVAILABLE and self.engine:
                return self._add_transaction_sqlalchemy(transaction)
            else:
                return self._add_transaction_sqlite(transaction)
        except Exception as e:
            logger.error(f"Error adding transaction: {e}")
            return None
    
    def get_transactions(self, user_id: int, limit: int = None, offset: int = 0) -> List[FinancialTransaction]:
        """Get transactions for a user"""
        try:
            if SQLALCHEMY_AVAILABLE and self.engine:
                return self._get_transactions_sqlalchemy(user_id, limit, offset)
            else:
                return self._get_transactions_sqlite(user_id, limit, offset)
        except Exception as e:
            logger.error(f"Error getting transactions: {e}")
            return []
    
    def get_transactions_dataframe(self, user_id: int, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Get transactions as pandas DataFrame"""
        try:
            if SQLALCHEMY_AVAILABLE and self.engine:
                query = """
                    SELECT * FROM financial_transactions 
                    WHERE user_id = :user_id
                """
                params = {"user_id": user_id}
                
                if start_date:
                    query += " AND date >= :start_date"
                    params["start_date"] = start_date
                
                if end_date:
                    query += " AND date <= :end_date"
                    params["end_date"] = end_date
                
                query += " ORDER BY date DESC"
                
                return pd.read_sql(query, self.engine, params=params)
            else:
                query = """
                    SELECT * FROM financial_transactions 
                    WHERE user_id = ?
                """
                params = [user_id]
                
                if start_date:
                    query += " AND date >= ?"
                    params.append(start_date.isoformat())
                
                if end_date:
                    query += " AND date <= ?"
                    params.append(end_date.isoformat())
                
                query += " ORDER BY date DESC"
                
                return pd.read_sql(query, self.connection, params=params)
                
        except Exception as e:
            logger.error(f"Error getting transactions dataframe: {e}")
            return pd.DataFrame()
    
    def update_transaction(self, transaction_id: int, updates: Dict[str, Any]) -> bool:
        """Update a transaction"""
        try:
            updates['updated_at'] = datetime.utcnow()
            
            set_clause = ", ".join([f"{key} = :{key}" for key in updates.keys()])
            query = f"UPDATE financial_transactions SET {set_clause} WHERE id = :transaction_id"
            
            updates['transaction_id'] = transaction_id
            
            if SQLALCHEMY_AVAILABLE and self.engine:
                with self.engine.connect() as conn:
                    conn.execute(text(query), updates)
                    conn.commit()
            else:
                # Convert to SQLite format
                sqlite_query = query.replace(":", "")
                sqlite_params = [updates[key.replace(":", "")] for key in updates.keys()]
                
                cursor = self.connection.cursor()
                cursor.execute(sqlite_query, sqlite_params)
                self.connection.commit()
            
            return True
        except Exception as e:
            logger.error(f"Error updating transaction: {e}")
            return False
    
    def delete_transaction(self, transaction_id: int, user_id: int) -> bool:
        """Delete a transaction (with user verification)"""
        try:
            if SQLALCHEMY_AVAILABLE and self.engine:
                with self.engine.connect() as conn:
                    result = conn.execute(
                        text("DELETE FROM financial_transactions WHERE id = :tid AND user_id = :uid"),
                        {"tid": transaction_id, "uid": user_id}
                    )
                    conn.commit()
                    return result.rowcount > 0
            else:
                cursor = self.connection.cursor()
                cursor.execute(
                    "DELETE FROM financial_transactions WHERE id = ? AND user_id = ?",
                    (transaction_id, user_id)
                )
                self.connection.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error deleting transaction: {e}")
            return False
    
    def bulk_import_transactions(self, user_id: int, transactions_df: pd.DataFrame) -> int:
        """Bulk import transactions from DataFrame"""
        try:
            transactions_df['user_id'] = user_id
            transactions_df['created_at'] = datetime.utcnow()
            transactions_df['updated_at'] = datetime.utcnow()
            
            if SQLALCHEMY_AVAILABLE and self.engine:
                count = transactions_df.to_sql(
                    'financial_transactions', 
                    self.engine, 
                    if_exists='append', 
                    index=False
                )
            else:
                count = transactions_df.to_sql(
                    'financial_transactions',
                    self.connection,
                    if_exists='append',
                    index=False
                )
            
            return count
        except Exception as e:
            logger.error(f"Error bulk importing transactions: {e}")
            return 0
    
    # Helper Methods
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt"""
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
    
    def _verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash"""
        return self._hash_password(password, salt) == password_hash
    
    def _update_user_login(self, user_id: int):
        """Update user login timestamp and reset failed attempts"""
        try:
            if SQLALCHEMY_AVAILABLE and self.engine:
                with self.engine.connect() as conn:
                    conn.execute(
                        text("UPDATE users SET last_login = :now, failed_login_attempts = 0, locked_until = NULL WHERE id = :user_id"),
                        {"now": datetime.utcnow(), "user_id": user_id}
                    )
                    conn.commit()
            else:
                cursor = self.connection.cursor()
                cursor.execute(
                    "UPDATE users SET last_login = ?, failed_login_attempts = 0, locked_until = NULL WHERE id = ?",
                    (datetime.utcnow(), user_id)
                )
                self.connection.commit()
        except Exception as e:
            logger.error(f"Error updating user login: {e}")
    
    def _increment_failed_attempts(self, user_id: int):
        """Increment failed login attempts and lock account if necessary"""
        try:
            max_attempts = self.config.security.max_login_attempts
            lock_duration = timedelta(minutes=15)  # Lock for 15 minutes
            
            if SQLALCHEMY_AVAILABLE and self.engine:
                with self.engine.connect() as conn:
                    # Get current failed attempts
                    result = conn.execute(
                        text("SELECT failed_login_attempts FROM users WHERE id = :user_id"),
                        {"user_id": user_id}
                    )
                    current_attempts = result.fetchone()[0] + 1
                    
                    # Update attempts and lock if necessary
                    locked_until = None
                    if current_attempts >= max_attempts:
                        locked_until = datetime.utcnow() + lock_duration
                    
                    conn.execute(
                        text("UPDATE users SET failed_login_attempts = :attempts, locked_until = :locked WHERE id = :user_id"),
                        {"attempts": current_attempts, "locked": locked_until, "user_id": user_id}
                    )
                    conn.commit()
            else:
                cursor = self.connection.cursor()
                
                # Get current failed attempts
                cursor.execute("SELECT failed_login_attempts FROM users WHERE id = ?", (user_id,))
                current_attempts = cursor.fetchone()[0] + 1
                
                # Update attempts and lock if necessary
                locked_until = None
                if current_attempts >= max_attempts:
                    locked_until = datetime.utcnow() + lock_duration
                
                cursor.execute(
                    "UPDATE users SET failed_login_attempts = ?, locked_until = ? WHERE id = ?",
                    (current_attempts, locked_until, user_id)
                )
                self.connection.commit()
                
        except Exception as e:
            logger.error(f"Error incrementing failed attempts: {e}")
    
    # SQLAlchemy-specific implementations
    def _create_user_sqlalchemy(self, user: User) -> Optional[int]:
        """Create user using SQLAlchemy"""
        with self.engine.connect() as conn:
            result = conn.execute(
                text('''
                    INSERT INTO users (username, email, password_hash, salt, created_at, preferences)
                    VALUES (:username, :email, :password_hash, :salt, :created_at, :preferences)
                '''),
                {
                    'username': user.username,
                    'email': user.email,
                    'password_hash': user.password_hash,
                    'salt': user.salt,
                    'created_at': user.created_at,
                    'preferences': json.dumps(user.preferences)
                }
            )
            conn.commit()
            return result.lastrowid
    
    def _get_user_by_username_sqlalchemy(self, username: str) -> Optional[User]:
        """Get user by username using SQLAlchemy"""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM users WHERE username = :username"),
                {"username": username}
            )
            row = result.fetchone()
            
            if row:
                return User(
                    id=row.id,
                    username=row.username,
                    email=row.email,
                    password_hash=row.password_hash,
                    salt=row.salt,
                    created_at=row.created_at,
                    last_login=row.last_login,
                    is_active=row.is_active,
                    failed_login_attempts=row.failed_login_attempts,
                    locked_until=row.locked_until,
                    preferences=json.loads(row.preferences) if row.preferences else {}
                )
            return None
    
    def _get_user_by_id_sqlalchemy(self, user_id: int) -> Optional[User]:
        """Get user by ID using SQLAlchemy"""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM users WHERE id = :user_id"),
                {"user_id": user_id}
            )
            row = result.fetchone()
            
            if row:
                return User(
                    id=row.id,
                    username=row.username,
                    email=row.email,
                    password_hash=row.password_hash,
                    salt=row.salt,
                    created_at=row.created_at,
                    last_login=row.last_login,
                    is_active=row.is_active,
                    failed_login_attempts=row.failed_login_attempts,
                    locked_until=row.locked_until,
                    preferences=json.loads(row.preferences) if row.preferences else {}
                )
            return None
    
    # SQLite-specific implementations
    def _create_user_sqlite(self, user: User) -> Optional[int]:
        """Create user using SQLite"""
        cursor = self.connection.cursor()
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, salt, created_at, preferences)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user.username,
            user.email,
            user.password_hash,
            user.salt,
            user.created_at,
            json.dumps(user.preferences)
        ))
        self.connection.commit()
        return cursor.lastrowid
    
    def _get_user_by_username_sqlite(self, username: str) -> Optional[User]:
        """Get user by username using SQLite"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        
        if row:
            return User(
                id=row['id'],
                username=row['username'],
                email=row['email'],
                password_hash=row['password_hash'],
                salt=row['salt'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                last_login=datetime.fromisoformat(row['last_login']) if row['last_login'] else None,
                is_active=bool(row['is_active']),
                failed_login_attempts=row['failed_login_attempts'],
                locked_until=datetime.fromisoformat(row['locked_until']) if row['locked_until'] else None,
                preferences=json.loads(row['preferences']) if row['preferences'] else {}
            )
        return None
    
    def _get_user_by_id_sqlite(self, user_id: int) -> Optional[User]:
        """Get user by ID using SQLite"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        
        if row:
            return User(
                id=row['id'],
                username=row['username'],
                email=row['email'],
                password_hash=row['password_hash'],
                salt=row['salt'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                last_login=datetime.fromisoformat(row['last_login']) if row['last_login'] else None,
                is_active=bool(row['is_active']),
                failed_login_attempts=row['failed_login_attempts'],
                locked_until=datetime.fromisoformat(row['locked_until']) if row['locked_until'] else None,
                preferences=json.loads(row['preferences']) if row['preferences'] else {}
            )
        return None
    
    def _add_transaction_sqlite(self, transaction: FinancialTransaction) -> Optional[int]:
        """Add transaction using SQLite"""
        cursor = self.connection.cursor()
        cursor.execute('''
            INSERT INTO financial_transactions (
                user_id, date, description, amount, transaction_type,
                category, subcategory, balance, account_name, tags,
                notes, is_recurring, merchant, location,
                sentiment_score, anomaly_score, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            transaction.user_id,
            transaction.date.isoformat() if transaction.date else None,
            transaction.description,
            transaction.amount,
            transaction.transaction_type,
            transaction.category,
            transaction.subcategory,
            transaction.balance,
            transaction.account_name,
            transaction.tags,
            transaction.notes,
            transaction.is_recurring,
            transaction.merchant,
            transaction.location,
            transaction.sentiment_score,
            transaction.anomaly_score,
            datetime.utcnow().isoformat(),
            datetime.utcnow().isoformat()
        ))
        self.connection.commit()
        return cursor.lastrowid
    
    def _get_transactions_sqlite(self, user_id: int, limit: int = None, offset: int = 0) -> List[FinancialTransaction]:
        """Get transactions using SQLite"""
        cursor = self.connection.cursor()
        
        query = "SELECT * FROM financial_transactions WHERE user_id = ? ORDER BY date DESC"
        params = [user_id]
        
        if limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        transactions = []
        for row in rows:
            transactions.append(FinancialTransaction(
                id=row['id'],
                user_id=row['user_id'],
                date=datetime.fromisoformat(row['date']) if row['date'] else None,
                description=row['description'],
                amount=row['amount'],
                transaction_type=row['transaction_type'],
                category=row['category'],
                subcategory=row['subcategory'],
                balance=row['balance'],
                account_name=row['account_name'],
                tags=row['tags'],
                notes=row['notes'],
                is_recurring=bool(row['is_recurring']),
                merchant=row['merchant'],
                location=row['location'],
                sentiment_score=row['sentiment_score'],
                anomaly_score=row['anomaly_score'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
            ))
        
        return transactions
    
    def close(self):
        """Close database connections"""
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()

# Global database instance
_db_instance: Optional[DatabaseManager] = None

def get_database() -> DatabaseManager:
    """Get database instance (singleton)"""
    global _db_instance
    
    if _db_instance is None:
        _db_instance = DatabaseManager()
    
    return _db_instance

def close_database():
    """Close database instance"""
    global _db_instance
    
    if _db_instance:
        _db_instance.close()
        _db_instance = None 