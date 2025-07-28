"""
AI Attendance System Pipeline V1

A comprehensive face recognition attendance system using SCRFD detection
and ArcFace recognition with SQLite database for rapid demo deployment.

Repository: hoangh-e/auto-face-attendance/tree/pipeline-v1.0/
Architecture: Camera → Frigate → MQTT → AI Service → Database → Notifications
"""

__version__ = "1.0.0"
__author__ = "AI Attendance System Team"
__description__ = "AI-powered attendance system with SCRFD face detection and ArcFace recognition"

# Core imports for easy access
from .core.ai_models import AttendanceAIModels
from .core.database import AttendanceDatabaseSQLite
from .core.processor import AttendanceProcessor
from .core.utils import *

# Integration imports
from .integrations.camera import CameraManager
from .integrations.notifications import NotificationManager

# Configuration
from .config.settings import AttendanceConfig
from .config.constants import *

__all__ = [
    'AttendanceAIModels',
    'AttendanceDatabaseSQLite', 
    'AttendanceProcessor',
    'CameraManager',
    'NotificationManager',
    'AttendanceConfig'
] 