"""
Settings Management Module for Attendance System Pipeline V1

Centralized configuration management with environment variable support,
validation, and different deployment profiles.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AIModelConfig:
    """AI Model configuration"""
    model_pack: str = 'buffalo_l'
    detection_threshold: float = 0.5
    recognition_threshold: float = 0.65
    detection_size: tuple = (640, 640)
    use_gpu: bool = True
    ctx_id: Optional[int] = None


@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: str = 'sqlite'  # 'sqlite' or 'postgresql'
    sqlite_path: str = 'attendance_system.db'
    postgresql_url: Optional[str] = None
    enable_wal: bool = True
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    retention_days: int = 30


@dataclass
class BusinessLogicConfig:
    """Business logic configuration"""
    cooldown_minutes: int = 30
    work_hours_start: str = '07:00'
    work_hours_end: str = '19:00'
    weekend_enabled: bool = True
    timezone: str = 'UTC'
    face_quality_threshold: float = 0.3
    max_faces_per_frame: int = 10


@dataclass
class CameraConfig:
    """Camera configuration"""
    default_resolution: tuple = (640, 480)
    default_fps: int = 30
    buffer_size: int = 1
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 3
    frame_skip_enabled: bool = True


@dataclass
class NotificationConfig:
    """Notification configuration"""
    enabled: bool = True
    console_enabled: bool = True
    slack_enabled: bool = False
    teams_enabled: bool = False
    email_enabled: bool = False
    webhook_enabled: bool = False
    
    # Slack settings
    slack_webhook_url: str = ''
    slack_channel: str = '#attendance'
    slack_username: str = 'AttendanceBot'
    
    # Teams settings
    teams_webhook_url: str = ''
    
    # Email settings
    smtp_server: str = ''
    smtp_port: int = 587
    smtp_username: str = ''
    smtp_password: str = ''
    email_from: str = ''
    email_to: list = field(default_factory=list)


@dataclass
class SecurityConfig:
    """Security configuration"""
    anti_spoofing_enabled: bool = False
    data_encryption_enabled: bool = False
    access_log_enabled: bool = True
    session_timeout_minutes: int = 60
    max_login_attempts: int = 3


@dataclass
class PerformanceConfig:
    """Performance configuration"""
    enable_monitoring: bool = True
    enable_caching: bool = True
    cache_size_mb: int = 512
    enable_batch_processing: bool = True
    max_batch_size: int = 10
    enable_gpu_optimization: bool = True


class AttendanceConfig:
    """Central configuration manager for the attendance system"""
    
    def __init__(self, config_file: Optional[str] = None, profile: str = 'default'):
        """
        Initialize configuration
        
        Args:
            config_file: Path to configuration file (YAML or JSON)
            profile: Configuration profile ('default', 'demo', 'production')
        """
        self.profile = profile
        self.config_file = config_file
        
        # Initialize with default configurations
        self.ai_models = AIModelConfig()
        self.database = DatabaseConfig()
        self.business_logic = BusinessLogicConfig()
        self.camera = CameraConfig()
        self.notifications = NotificationConfig()
        self.security = SecurityConfig()
        self.performance = PerformanceConfig()
        
        # Load profile-specific configurations
        self._load_profile_config()
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Override with environment variables
        self._load_from_environment()
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"🔧 Configuration loaded: profile={profile}")
    
    def _load_profile_config(self):
        """Load profile-specific configurations"""
        if self.profile == 'demo':
            # Demo profile - optimized for quick setup
            self.ai_models.model_pack = 'buffalo_s'  # Faster model
            self.database.sqlite_path = 'demo_attendance.db'
            self.business_logic.cooldown_minutes = 5  # Shorter cooldown for demo
            self.notifications.console_enabled = True
            self.performance.enable_monitoring = True
            
        elif self.profile == 'production':
            # Production profile - full features
            self.ai_models.model_pack = 'buffalo_l'  # Best accuracy
            self.database.type = 'postgresql'
            self.database.backup_enabled = True
            self.business_logic.cooldown_minutes = 30
            self.security.anti_spoofing_enabled = True
            self.security.data_encryption_enabled = True
            self.notifications.enabled = True
            
        elif self.profile == 'colab':
            # Google Colab profile
            self.ai_models.use_gpu = True
            self.database.sqlite_path = '/content/attendance_colab.db'
            self.camera.default_resolution = (640, 480)  # Colab-friendly
            self.performance.enable_gpu_optimization = True
            
        elif self.profile == 'local':
            # Local development profile
            self.ai_models.model_pack = 'buffalo_s'
            self.database.sqlite_path = 'local_attendance.db'
            self.notifications.console_enabled = True
            self.performance.enable_monitoring = True
    
    def load_from_file(self, config_file: str):
        """Load configuration from YAML or JSON file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_file}")
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    logger.error(f"Unsupported config file format: {config_path.suffix}")
                    return
            
            self._update_from_dict(config_data)
            logger.info(f"✅ Configuration loaded from: {config_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load config file {config_file}: {e}")
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            # AI Models
            'ATTENDANCE_MODEL_PACK': ('ai_models', 'model_pack'),
            'ATTENDANCE_DETECTION_THRESHOLD': ('ai_models', 'detection_threshold', float),
            'ATTENDANCE_RECOGNITION_THRESHOLD': ('ai_models', 'recognition_threshold', float),
            'ATTENDANCE_USE_GPU': ('ai_models', 'use_gpu', bool),
            
            # Database
            'ATTENDANCE_DB_TYPE': ('database', 'type'),
            'ATTENDANCE_SQLITE_PATH': ('database', 'sqlite_path'),
            'ATTENDANCE_POSTGRESQL_URL': ('database', 'postgresql_url'),
            
            # Business Logic
            'ATTENDANCE_COOLDOWN_MINUTES': ('business_logic', 'cooldown_minutes', int),
            'ATTENDANCE_WORK_START': ('business_logic', 'work_hours_start'),
            'ATTENDANCE_WORK_END': ('business_logic', 'work_hours_end'),
            
            # Notifications
            'ATTENDANCE_SLACK_WEBHOOK': ('notifications', 'slack_webhook_url'),
            'ATTENDANCE_TEAMS_WEBHOOK': ('notifications', 'teams_webhook_url'),
            'ATTENDANCE_EMAIL_SMTP': ('notifications', 'smtp_server'),
            
            # Security
            'ATTENDANCE_ANTI_SPOOFING': ('security', 'anti_spoofing_enabled', bool),
            'ATTENDANCE_DATA_ENCRYPTION': ('security', 'data_encryption_enabled', bool),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Parse value type
                    if len(config_path) == 3:
                        section, key, value_type = config_path
                        if value_type == bool:
                            value = value.lower() in ('true', '1', 'yes', 'on')
                        elif value_type == int:
                            value = int(value)
                        elif value_type == float:
                            value = float(value)
                    else:
                        section, key = config_path
                    
                    # Set configuration value
                    config_section = getattr(self, section)
                    setattr(config_section, key, value)
                    logger.debug(f"Environment override: {env_var} -> {section}.{key} = {value}")
                    
                except Exception as e:
                    logger.warning(f"Failed to parse environment variable {env_var}: {e}")
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section_name, section_data in config_dict.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                config_section = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(config_section, key):
                        setattr(config_section, key, value)
                    else:
                        logger.warning(f"Unknown config key: {section_name}.{key}")
            else:
                logger.warning(f"Unknown config section: {section_name}")
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Validate AI model settings
        if self.ai_models.detection_threshold < 0 or self.ai_models.detection_threshold > 1:
            errors.append("detection_threshold must be between 0 and 1")
        
        if self.ai_models.recognition_threshold < 0 or self.ai_models.recognition_threshold > 1:
            errors.append("recognition_threshold must be between 0 and 1")
        
        # Validate business logic
        if self.business_logic.cooldown_minutes < 0:
            errors.append("cooldown_minutes must be positive")
        
        # Validate work hours format
        try:
            time.fromisoformat(self.business_logic.work_hours_start)
            time.fromisoformat(self.business_logic.work_hours_end)
        except ValueError:
            errors.append("work_hours must be in HH:MM format")
        
        # Validate database settings
        if self.database.type not in ['sqlite', 'postgresql']:
            errors.append("database type must be 'sqlite' or 'postgresql'")
        
        if self.database.type == 'postgresql' and not self.database.postgresql_url:
            errors.append("postgresql_url required when using PostgreSQL")
        
        # Log validation results
        if errors:
            for error in errors:
                logger.error(f"❌ Configuration error: {error}")
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
        else:
            logger.info("✅ Configuration validation passed")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'ai_models': self.ai_models.__dict__,
            'database': self.database.__dict__,
            'business_logic': self.business_logic.__dict__,
            'camera': self.camera.__dict__,
            'notifications': self.notifications.__dict__,
            'security': self.security.__dict__,
            'performance': self.performance.__dict__,
        }
    
    def save_to_file(self, file_path: str, format: str = 'yaml'):
        """Save configuration to file"""
        config_dict = self.to_dict()
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"✅ Configuration saved to: {file_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get AI model configuration for initialization"""
        return {
            'model_pack': self.ai_models.model_pack,
            'ctx_id': self.ai_models.ctx_id,
            'det_size': self.ai_models.detection_size,
            'det_thresh': self.ai_models.detection_threshold
        }
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        if self.database.type == 'sqlite':
            return {
                'db_path': self.database.sqlite_path,
                'enable_wal': self.database.enable_wal
            }
        else:
            return {
                'connection_string': self.database.postgresql_url
            }
    
    def get_notification_config(self) -> Dict[str, Any]:
        """Get notification configuration"""
        return {
            'enabled': self.notifications.enabled,
            'channels': {
                'console': self.notifications.console_enabled,
                'slack': self.notifications.slack_enabled,
                'teams': self.notifications.teams_enabled,
                'email': self.notifications.email_enabled,
            },
            'slack': {
                'webhook_url': self.notifications.slack_webhook_url,
                'channel': self.notifications.slack_channel,
                'username': self.notifications.slack_username,
            },
            'teams': {
                'webhook_url': self.notifications.teams_webhook_url,
            },
            'email': {
                'smtp_server': self.notifications.smtp_server,
                'smtp_port': self.notifications.smtp_port,
                'username': self.notifications.smtp_username,
                'password': self.notifications.smtp_password,
                'from_email': self.notifications.email_from,
                'to_emails': self.notifications.email_to,
            }
        }


# Global configuration instance
_global_config: Optional[AttendanceConfig] = None


def get_config() -> AttendanceConfig:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = AttendanceConfig()
    return _global_config


def set_config(config: AttendanceConfig):
    """Set global configuration instance"""
    global _global_config
    _global_config = config


def load_config(config_file: str = None, profile: str = 'default') -> AttendanceConfig:
    """Load and set global configuration"""
    config = AttendanceConfig(config_file=config_file, profile=profile)
    set_config(config)
    return config 