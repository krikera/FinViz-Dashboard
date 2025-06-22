"""
Configuration Management System for FinViz Dashboard
Supports multiple environments and secure credential management
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import streamlit as st

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    type: str = "sqlite"  # sqlite, postgresql, mysql
    host: str = "localhost"
    port: int = 5432
    database: str = "finviz.db"
    username: str = ""
    password: str = ""
    pool_size: int = 5
    max_overflow: int = 10

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    enable_authentication: bool = True
    session_timeout_minutes: int = 30
    max_login_attempts: int = 3
    password_min_length: int = 8
    enable_2fa: bool = False
    jwt_secret_key: str = ""
    encryption_key: str = ""

@dataclass
class CacheConfig:
    """Caching configuration settings"""
    enable_redis: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl_seconds: int = 3600
    enable_memory_cache: bool = True

@dataclass
class APIConfig:
    """External API configuration"""
    enable_real_time_data: bool = False
    alpha_vantage_api_key: str = ""
    plaid_client_id: str = ""
    plaid_secret: str = ""
    yfinance_enabled: bool = True
    rate_limit_per_minute: int = 60

@dataclass
class MLConfig:
    """Machine Learning configuration"""
    enable_advanced_models: bool = True
    model_cache_path: str = "./models"
    auto_retrain_days: int = 7
    anomaly_threshold: float = 0.05
    prediction_horizon_days: int = 30

@dataclass
class UIConfig:
    """User Interface configuration"""
    theme: str = "light"  # light, dark, auto
    default_chart_type: str = "plotly"
    enable_animations: bool = True
    max_file_size_mb: int = 100
    supported_formats: list = None

    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ["csv", "xlsx", "json"]

class Config:
    """Main configuration class with environment support"""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.config_dir = Path(__file__).parent / "config"
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize configurations
        self.database = DatabaseConfig()
        self.security = SecurityConfig()
        self.cache = CacheConfig()
        self.api = APIConfig()
        self.ml = MLConfig()
        self.ui = UIConfig()
        
        # Load environment-specific configurations
        self.load_config()
        self.validate_config()
    
    def load_config(self):
        """Load configuration from environment variables and config files"""
        # Load from environment variables
        self._load_from_env()
        
        # Load from config file
        config_file = self.config_dir / f"{self.environment}.json"
        if config_file.exists():
            self._load_from_file(config_file)
        
        # Load secrets from separate file (not committed to git)
        secrets_file = self.config_dir / "secrets.json"
        if secrets_file.exists():
            self._load_secrets(secrets_file)
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Database
        self.database.type = os.getenv("DB_TYPE", self.database.type)
        self.database.host = os.getenv("DB_HOST", self.database.host)
        self.database.port = int(os.getenv("DB_PORT", self.database.port))
        self.database.database = os.getenv("DB_NAME", self.database.database)
        self.database.username = os.getenv("DB_USERNAME", self.database.username)
        self.database.password = os.getenv("DB_PASSWORD", self.database.password)
        
        # Security
        self.security.jwt_secret_key = os.getenv("JWT_SECRET_KEY", self.security.jwt_secret_key)
        self.security.encryption_key = os.getenv("ENCRYPTION_KEY", self.security.encryption_key)
        
        # APIs
        self.api.alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY", self.api.alpha_vantage_api_key)
        self.api.plaid_client_id = os.getenv("PLAID_CLIENT_ID", self.api.plaid_client_id)
        self.api.plaid_secret = os.getenv("PLAID_SECRET", self.api.plaid_secret)
    
    def _load_from_file(self, config_file: Path):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations with file data
            if "database" in config_data:
                self._update_dataclass(self.database, config_data["database"])
            if "security" in config_data:
                self._update_dataclass(self.security, config_data["security"])
            if "cache" in config_data:
                self._update_dataclass(self.cache, config_data["cache"])
            if "api" in config_data:
                self._update_dataclass(self.api, config_data["api"])
            if "ml" in config_data:
                self._update_dataclass(self.ml, config_data["ml"])
            if "ui" in config_data:
                self._update_dataclass(self.ui, config_data["ui"])
                
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    def _load_secrets(self, secrets_file: Path):
        """Load secrets from separate file"""
        try:
            with open(secrets_file, 'r') as f:
                secrets = json.load(f)
            
            # Update sensitive configurations
            if "jwt_secret_key" in secrets:
                self.security.jwt_secret_key = secrets["jwt_secret_key"]
            if "encryption_key" in secrets:
                self.security.encryption_key = secrets["encryption_key"]
            if "database_password" in secrets:
                self.database.password = secrets["database_password"]
            if "api_keys" in secrets:
                api_keys = secrets["api_keys"]
                self.api.alpha_vantage_api_key = api_keys.get("alpha_vantage", "")
                self.api.plaid_client_id = api_keys.get("plaid_client_id", "")
                self.api.plaid_secret = api_keys.get("plaid_secret", "")
                
        except Exception as e:
            print(f"Warning: Could not load secrets file: {e}")
    
    def _update_dataclass(self, obj, data: Dict[str, Any]):
        """Update dataclass with dictionary data"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def validate_config(self):
        """Validate configuration settings"""
        if self.security.enable_authentication and not self.security.jwt_secret_key:
            print("Warning: JWT secret key not set. Authentication may not work properly.")
        
        if self.api.enable_real_time_data and not any([
            self.api.alpha_vantage_api_key,
            self.api.plaid_client_id
        ]):
            print("Warning: Real-time data enabled but no API keys configured.")
        
        if self.database.type == "postgresql" and not all([
            self.database.host,
            self.database.username,
            self.database.password
        ]):
            print("Warning: PostgreSQL configured but missing connection details.")
    
    def save_config(self):
        """Save current configuration to file"""
        config_file = self.config_dir / f"{self.environment}.json"
        config_data = {
            "database": asdict(self.database),
            "security": asdict(self.security),
            "cache": asdict(self.cache),
            "api": asdict(self.api),
            "ml": asdict(self.ml),
            "ui": asdict(self.ui)
        }
        
        # Remove sensitive data from config file
        config_data["database"]["password"] = ""
        config_data["security"]["jwt_secret_key"] = ""
        config_data["security"]["encryption_key"] = ""
        config_data["api"]["alpha_vantage_api_key"] = ""
        config_data["api"]["plaid_client_id"] = ""
        config_data["api"]["plaid_secret"] = ""
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        if self.database.type == "sqlite":
            return f"sqlite:///{self.database.database}"
        elif self.database.type == "postgresql":
            return f"postgresql://{self.database.username}:{self.database.password}@{self.database.host}:{self.database.port}/{self.database.database}"
        elif self.database.type == "mysql":
            return f"mysql://{self.database.username}:{self.database.password}@{self.database.host}:{self.database.port}/{self.database.database}"
        else:
            raise ValueError(f"Unsupported database type: {self.database.type}")

# Global configuration instance
_config_instance: Optional[Config] = None

def get_config(environment: str = None) -> Config:
    """Get configuration instance (singleton)"""
    global _config_instance
    
    if _config_instance is None:
        if environment is None:
            environment = os.getenv("FINVIZ_ENV", "development")
        _config_instance = Config(environment)
    
    return _config_instance

def reset_config():
    """Reset configuration instance (useful for testing)"""
    global _config_instance
    _config_instance = None

# Streamlit integration for configuration management
def display_config_ui():
    """Display configuration UI in Streamlit sidebar"""
    if "config_expanded" not in st.session_state:
        st.session_state.config_expanded = False
    
    with st.sidebar:
        st.markdown("---")
        if st.button("⚙️ Settings", key="config_toggle"):
            st.session_state.config_expanded = not st.session_state.config_expanded
        
        if st.session_state.config_expanded:
            config = get_config()
            
            st.subheader("Application Settings")
            
            # UI Settings
            with st.expander("🎨 Interface"):
                config.ui.theme = st.selectbox(
                    "Theme", ["light", "dark", "auto"], 
                    index=["light", "dark", "auto"].index(config.ui.theme)
                )
                config.ui.default_chart_type = st.selectbox(
                    "Default Chart Type", ["plotly", "matplotlib"], 
                    index=["plotly", "matplotlib"].index(config.ui.default_chart_type)
                )
                config.ui.enable_animations = st.checkbox(
                    "Enable Animations", config.ui.enable_animations
                )
            
            # Security Settings
            with st.expander("🔒 Security"):
                config.security.enable_authentication = st.checkbox(
                    "Enable Authentication", config.security.enable_authentication
                )
                config.security.session_timeout_minutes = st.number_input(
                    "Session Timeout (minutes)", 
                    min_value=5, max_value=480, 
                    value=config.security.session_timeout_minutes
                )
            
            # ML Settings
            with st.expander("🤖 AI/ML"):
                config.ml.enable_advanced_models = st.checkbox(
                    "Enable Advanced Models", config.ml.enable_advanced_models
                )
                config.ml.anomaly_threshold = st.slider(
                    "Anomaly Detection Threshold", 
                    min_value=0.01, max_value=0.1, 
                    value=config.ml.anomaly_threshold, step=0.01
                )
                config.ml.prediction_horizon_days = st.number_input(
                    "Prediction Horizon (days)", 
                    min_value=7, max_value=365, 
                    value=config.ml.prediction_horizon_days
                )
            
            # API Settings
            with st.expander("🌐 APIs"):
                config.api.enable_real_time_data = st.checkbox(
                    "Enable Real-time Data", config.api.enable_real_time_data
                )
                config.api.yfinance_enabled = st.checkbox(
                    "Enable Yahoo Finance", config.api.yfinance_enabled
                )
            
            if st.button("💾 Save Settings"):
                config.save_config()
                st.success("Settings saved successfully!")
                st.rerun() 