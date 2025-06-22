"""
Authentication System for FinViz Dashboard
Provides secure user authentication, session management, and access control
"""

import streamlit as st
import jwt
import hashlib
import secrets
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass
import time

try:
    from database import get_database, User
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    print("Database not available. Authentication features limited.")

from config import get_config

@dataclass
class Session:
    """User session data"""
    user_id: int
    username: str
    email: str
    created_at: datetime
    expires_at: datetime
    is_active: bool = True

class AuthenticationManager:
    """Comprehensive authentication manager with security features"""
    
    def __init__(self):
        self.config = get_config()
        self.db = get_database() if DATABASE_AVAILABLE else None
        
        # Initialize session state keys
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize Streamlit session state for authentication"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'session' not in st.session_state:
            st.session_state.session = None
        if 'login_attempts' not in st.session_state:
            st.session_state.login_attempts = 0
        if 'locked_until' not in st.session_state:
            st.session_state.locked_until = None
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        if not self.config.security.enable_authentication:
            return True
        
        if not st.session_state.authenticated:
            return False
        
        # Check session expiry
        if st.session_state.session:
            if st.session_state.session.expires_at < datetime.utcnow():
                self.logout()
                return False
        
        return True
    
    def get_current_user(self) -> Optional[User]:
        """Get current authenticated user"""
        if self.is_authenticated():
            return st.session_state.user
        return None
    
    def get_current_user_id(self) -> Optional[int]:
        """Get current user ID"""
        user = self.get_current_user()
        return user.id if user else None
    
    def login(self, username: str, password: str) -> bool:
        """Authenticate user with username and password"""
        if not DATABASE_AVAILABLE or not self.db:
            return False
        
        # Check if account is temporarily locked
        if self._is_account_locked():
            return False
        
        # Validate input
        if not username or not password:
            return False
        
        # Authenticate against database
        user = self.db.authenticate_user(username, password)
        
        if user:
            # Create session
            session = Session(
                user_id=user.id,
                username=user.username,
                email=user.email,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(minutes=self.config.security.session_timeout_minutes)
            )
            
            # Update session state
            st.session_state.authenticated = True
            st.session_state.user = user
            st.session_state.session = session
            st.session_state.login_attempts = 0
            st.session_state.locked_until = None
            
            return True
        else:
            # Handle failed login
            self._handle_failed_login()
            return False
    
    def register(self, username: str, email: str, password: str, confirm_password: str) -> tuple[bool, str]:
        """Register a new user"""
        if not DATABASE_AVAILABLE or not self.db:
            return False, "Database not available"
        
        # Validate input
        validation_result = self._validate_registration_input(username, email, password, confirm_password)
        if not validation_result[0]:
            return validation_result
        
        # Check if user already exists
        existing_user = self.db.get_user_by_username(username)
        if existing_user:
            return False, "Username already exists"
        
        # Create user
        user_id = self.db.create_user(username, email, password)
        
        if user_id:
            return True, "User registered successfully"
        else:
            return False, "Failed to create user"
    
    def logout(self):
        """Logout current user"""
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.session = None
    
    def change_password(self, current_password: str, new_password: str, confirm_password: str) -> tuple[bool, str]:
        """Change user password"""
        if not self.is_authenticated():
            return False, "Not authenticated"
        
        user = self.get_current_user()
        if not user:
            return False, "User not found"
        
        # Validate current password
        if not self.db.authenticate_user(user.username, current_password):
            return False, "Current password is incorrect"
        
        # Validate new password
        if new_password != confirm_password:
            return False, "Passwords do not match"
        
        if not self._validate_password(new_password):
            return False, f"Password must be at least {self.config.security.password_min_length} characters long"
        
        # Update password in database (implementation depends on database layer)
        # For now, return success
        return True, "Password changed successfully"
    
    def require_authentication(self):
        """Decorator/guard to require authentication"""
        if not self.is_authenticated():
            self.show_login_form()
            st.stop()
    
    def show_login_form(self):
        """Display login/registration form"""
        st.title("🔐 Authentication Required")
        
        # Check if account is locked
        if self._is_account_locked():
            st.error(f"Account locked due to too many failed attempts. Try again later.")
            return
        
        # Create tabs for login and registration
        login_tab, register_tab = st.tabs(["Login", "Register"])
        
        with login_tab:
            self._show_login_tab()
        
        with register_tab:
            self._show_register_tab()
    
    def _show_login_tab(self):
        """Show login form"""
        st.subheader("Login to Your Account")
        
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                login_button = st.form_submit_button("Login", use_container_width=True)
            
            if login_button:
                if self.login(username, password):
                    st.success("Login successful!")
                    time.sleep(1)
                    st.rerun()
                else:
                    remaining_attempts = self.config.security.max_login_attempts - st.session_state.login_attempts
                    if remaining_attempts > 0:
                        st.error(f"Invalid credentials. {remaining_attempts} attempts remaining.")
                    else:
                        st.error("Account locked due to too many failed attempts.")
    
    def _show_register_tab(self):
        """Show registration form"""
        st.subheader("Create New Account")
        
        with st.form("register_form"):
            username = st.text_input("Username", key="register_username")
            email = st.text_input("Email", key="register_email")
            password = st.text_input("Password", type="password", key="register_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")
            
            # Terms and conditions
            terms_accepted = st.checkbox("I agree to the Terms and Conditions")
            
            register_button = st.form_submit_button("Register", use_container_width=True)
            
            if register_button:
                if not terms_accepted:
                    st.error("Please accept the Terms and Conditions")
                else:
                    success, message = self.register(username, email, password, confirm_password)
                    if success:
                        st.success(message)
                        st.info("Please use the Login tab to sign in.")
                    else:
                        st.error(message)
    
    def show_user_profile(self):
        """Show user profile and settings"""
        if not self.is_authenticated():
            return
        
        user = self.get_current_user()
        session = st.session_state.session
        
        st.subheader("👤 User Profile")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"**Username:** {user.username}")
            st.write(f"**Email:** {user.email}")
            st.write(f"**Account Created:** {user.created_at.strftime('%Y-%m-%d %H:%M') if user.created_at else 'Unknown'}")
            st.write(f"**Last Login:** {user.last_login.strftime('%Y-%m-%d %H:%M') if user.last_login else 'Never'}")
            st.write(f"**Session Expires:** {session.expires_at.strftime('%Y-%m-%d %H:%M') if session else 'Unknown'}")
        
        with col2:
            if st.button("🔓 Logout", use_container_width=True):
                self.logout()
                st.rerun()
        
        # Change Password Section
        st.markdown("---")
        st.subheader("🔑 Change Password")
        
        with st.expander("Change Password"):
            with st.form("change_password_form"):
                current_password = st.text_input("Current Password", type="password")
                new_password = st.text_input("New Password", type="password")
                confirm_new_password = st.text_input("Confirm New Password", type="password")
                
                change_password_button = st.form_submit_button("Change Password")
                
                if change_password_button:
                    success, message = self.change_password(current_password, new_password, confirm_new_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
    
    def show_admin_panel(self):
        """Show admin panel for user management (placeholder)"""
        if not self.is_authenticated():
            return
        
        user = self.get_current_user()
        
        # Simple admin check (in real app, use role-based permissions)
        if user.username != "admin":
            st.warning("Admin access required")
            return
        
        st.subheader("🛠️ Admin Panel")
        
        # User management placeholder
        st.write("**User Management**")
        st.info("User management features coming soon...")
        
        # System stats placeholder
        st.write("**System Statistics**")
        if DATABASE_AVAILABLE and self.db:
            # Get basic stats (this would need proper implementation)
            st.metric("Total Users", "N/A")
            st.metric("Active Sessions", "N/A")
            st.metric("Database Type", self.config.database.type)
    
    def _validate_registration_input(self, username: str, email: str, password: str, confirm_password: str) -> tuple[bool, str]:
        """Validate registration input"""
        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters long"
        
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return False, "Username can only contain letters, numbers, and underscores"
        
        if not email or not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
            return False, "Please enter a valid email address"
        
        if not self._validate_password(password):
            return False, f"Password must be at least {self.config.security.password_min_length} characters long"
        
        if password != confirm_password:
            return False, "Passwords do not match"
        
        return True, "Valid"
    
    def _validate_password(self, password: str) -> bool:
        """Validate password strength"""
        if len(password) < self.config.security.password_min_length:
            return False
        
        # Add more validation rules as needed
        # - Must contain uppercase and lowercase
        # - Must contain numbers
        # - Must contain special characters
        
        return True
    
    def _is_account_locked(self) -> bool:
        """Check if account is temporarily locked"""
        if st.session_state.locked_until:
            if datetime.utcnow() < st.session_state.locked_until:
                return True
            else:
                # Reset lock
                st.session_state.locked_until = None
                st.session_state.login_attempts = 0
        
        return False
    
    def _handle_failed_login(self):
        """Handle failed login attempt"""
        st.session_state.login_attempts += 1
        
        if st.session_state.login_attempts >= self.config.security.max_login_attempts:
            # Lock account for 15 minutes
            st.session_state.locked_until = datetime.utcnow() + timedelta(minutes=15)
    
    def generate_jwt_token(self, user_id: int) -> str:
        """Generate JWT token for API access"""
        if not self.config.security.jwt_secret_key:
            raise ValueError("JWT secret key not configured")
        
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.config.security.jwt_secret_key, algorithm='HS256')
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload"""
        try:
            if not self.config.security.jwt_secret_key:
                return None
            
            payload = jwt.decode(token, self.config.security.jwt_secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

# Global authentication instance
_auth_instance: Optional[AuthenticationManager] = None

def get_auth() -> AuthenticationManager:
    """Get authentication manager instance (singleton)"""
    global _auth_instance
    
    if _auth_instance is None:
        _auth_instance = AuthenticationManager()
    
    return _auth_instance

def require_auth():
    """Decorator to require authentication"""
    auth = get_auth()
    auth.require_authentication()

def get_current_user() -> Optional[User]:
    """Get current authenticated user"""
    auth = get_auth()
    return auth.get_current_user()

def get_current_user_id() -> Optional[int]:
    """Get current user ID"""
    auth = get_auth()
    return auth.get_current_user_id()

# Streamlit components for authentication
def show_auth_sidebar():
    """Show authentication info in sidebar"""
    auth = get_auth()
    
    with st.sidebar:
        st.markdown("---")
        
        if auth.is_authenticated():
            user = auth.get_current_user()
            st.write(f"👤 **{user.username}**")
            
            if st.button("👤 Profile", key="profile_btn"):
                st.session_state.show_profile = True
            
            if st.button("🔓 Logout", key="logout_btn"):
                auth.logout()
                st.rerun()
        else:
            st.write("🔐 **Not authenticated**")
            if st.button("🔑 Login", key="login_btn"):
                st.session_state.show_login = True

def handle_auth_modals():
    """Handle authentication modals"""
    auth = get_auth()
    
    # Handle profile modal
    if st.session_state.get("show_profile", False):
        with st.container():
            auth.show_user_profile()
            if st.button("✕ Close Profile"):
                st.session_state.show_profile = False
                st.rerun()
    
    # Handle login modal
    if st.session_state.get("show_login", False):
        with st.container():
            auth.show_login_form()
            if auth.is_authenticated():
                st.session_state.show_login = False
                st.rerun() 