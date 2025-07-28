"""
System Constants for AI Attendance System Pipeline V1

Central location for all system-wide constants, including model names,
file paths, error codes, and configuration defaults.
"""

from pathlib import Path
import os

# ============================================================================
# VERSION AND METADATA
# ============================================================================

VERSION = "1.0.0"
SYSTEM_NAME = "AI Attendance System"
PIPELINE_VERSION = "Pipeline V1"
REPOSITORY_URL = "https://github.com/hoangh-e/auto-face-attendance"
BRANCH_NAME = "pipeline-v1.0"

# ============================================================================
# AI MODEL CONSTANTS
# ============================================================================

# Available InsightFace model packs
AVAILABLE_MODEL_PACKS = [
    'buffalo_l',    # Large model - best accuracy
    'buffalo_m',    # Medium model - balanced
    'buffalo_s',    # Small model - fastest
    'buffalo_sc',   # Small compact - detection + recognition only
]

# Default model configurations
DEFAULT_MODEL_PACK = 'buffalo_l'
DEFAULT_DETECTION_SIZE = (640, 640)
DEFAULT_DETECTION_THRESHOLD = 0.5
DEFAULT_RECOGNITION_THRESHOLD = 0.65

# Face embedding dimensions
FACE_EMBEDDING_DIMENSION = 512

# Maximum number of faces to process per frame
MAX_FACES_PER_FRAME = 10

# Face quality thresholds
MIN_FACE_QUALITY = 0.3
RECOMMENDED_FACE_QUALITY = 0.7

# Face size requirements (pixels)
MIN_FACE_SIZE = 50
RECOMMENDED_FACE_SIZE = 100

# ============================================================================
# DATABASE CONSTANTS
# ============================================================================

# Database types
DB_TYPE_SQLITE = 'sqlite'
DB_TYPE_POSTGRESQL = 'postgresql'

# Default database names
DEFAULT_SQLITE_DB = 'attendance_system.db'
DEFAULT_DEMO_DB = 'attendance_demo.db'
DEFAULT_TEST_DB = 'test_attendance.db'

# Table names
TABLE_EMPLOYEES = 'employees'
TABLE_ATTENDANCE_LOGS = 'attendance_logs'
TABLE_FACE_REGISTRATIONS = 'face_registrations'
TABLE_SYSTEM_CONFIG = 'system_config'
TABLE_PERFORMANCE_METRICS = 'performance_metrics'

# Employee status
EMPLOYEE_STATUS_ACTIVE = 1
EMPLOYEE_STATUS_INACTIVE = 0

# Attendance event types
EVENT_TYPE_CHECK_IN = 'check_in'
EVENT_TYPE_CHECK_OUT = 'check_out'
VALID_EVENT_TYPES = [EVENT_TYPE_CHECK_IN, EVENT_TYPE_CHECK_OUT]

# Database performance settings
DEFAULT_CACHE_SIZE = 10000
DEFAULT_WAL_MODE = True
DEFAULT_SYNCHRONOUS_MODE = 'NORMAL'

# ============================================================================
# BUSINESS LOGIC CONSTANTS
# ============================================================================

# Work hours (24-hour format)
DEFAULT_WORK_START = '07:00'
DEFAULT_WORK_END = '19:00'

# Cooldown period between attendance records (minutes)
DEFAULT_COOLDOWN_MINUTES = 30
MIN_COOLDOWN_MINUTES = 1
MAX_COOLDOWN_MINUTES = 480  # 8 hours

# Timezone settings
DEFAULT_TIMEZONE = 'UTC'

# Weekend handling
WEEKEND_DAYS = [5, 6]  # Saturday, Sunday (0=Monday)

# ============================================================================
# CAMERA AND VIDEO CONSTANTS
# ============================================================================

# Default camera settings
DEFAULT_CAMERA_RESOLUTION = (640, 480)
DEFAULT_CAMERA_FPS = 30
DEFAULT_BUFFER_SIZE = 1

# Supported video formats
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

# Camera backends (OpenCV)
CAMERA_BACKENDS = {
    'DSHOW': 700,      # DirectShow (Windows)
    'V4L2': 200,       # Video4Linux2 (Linux)
    'GSTREAMER': 1800, # GStreamer
    'ANY': 0           # Auto-detect
}

# Performance targets
TARGET_REAL_TIME_FPS = 30
TARGET_FRAME_TIME_MS = 33.3  # 30 FPS = 33.3ms per frame

# ============================================================================
# FILE SYSTEM CONSTANTS
# ============================================================================

# Default directories
DEFAULT_DATA_DIR = 'data'
DEFAULT_LOGS_DIR = 'logs'
DEFAULT_BACKUP_DIR = 'backups'
DEFAULT_SNAPSHOTS_DIR = 'snapshots'
DEFAULT_UPLOADS_DIR = 'uploads'
DEFAULT_EXPORTS_DIR = 'exports'

# File extensions
LOG_FILE_EXTENSION = '.log'
CONFIG_FILE_EXTENSIONS = ['.yaml', '.yml', '.json']
BACKUP_FILE_EXTENSION = '.backup'

# File size limits
MAX_UPLOAD_FILE_SIZE_MB = 100
MAX_VIDEO_FILE_SIZE_MB = 500
MAX_IMAGE_FILE_SIZE_MB = 10

# ============================================================================
# NOTIFICATION CONSTANTS
# ============================================================================

# Notification channels
NOTIFICATION_CONSOLE = 'console'
NOTIFICATION_SLACK = 'slack'
NOTIFICATION_TEAMS = 'teams'
NOTIFICATION_EMAIL = 'email'
NOTIFICATION_WEBHOOK = 'webhook'

AVAILABLE_NOTIFICATION_CHANNELS = [
    NOTIFICATION_CONSOLE,
    NOTIFICATION_SLACK,
    NOTIFICATION_TEAMS,
    NOTIFICATION_EMAIL,
    NOTIFICATION_WEBHOOK
]

# Notification templates
NOTIFICATION_TEMPLATES = {
    EVENT_TYPE_CHECK_IN: '✅ {name} checked in at {time}',
    EVENT_TYPE_CHECK_OUT: '🚪 {name} checked out at {time}',
    'unknown_face': '❓ Unknown person detected at {time}',
    'system_alert': '⚠️ System Alert: {message}',
    'system_error': '❌ System Error: {message}',
    'system_info': 'ℹ️ System Info: {message}'
}

# Severity levels
SEVERITY_INFO = 'info'
SEVERITY_WARNING = 'warning'
SEVERITY_ERROR = 'error'
SEVERITY_CRITICAL = 'critical'

SEVERITY_LEVELS = [SEVERITY_INFO, SEVERITY_WARNING, SEVERITY_ERROR, SEVERITY_CRITICAL]

# ============================================================================
# PERFORMANCE CONSTANTS
# ============================================================================

# Performance thresholds
EXCELLENT_LATENCY_MS = 50
GOOD_LATENCY_MS = 100
ACCEPTABLE_LATENCY_MS = 200

# Memory limits
DEFAULT_MEMORY_LIMIT_MB = 2048
GPU_MEMORY_THRESHOLD_MB = 1024

# Batch processing
DEFAULT_BATCH_SIZE = 8
MAX_BATCH_SIZE = 32

# Cache settings
DEFAULT_CACHE_SIZE_MB = 512
MAX_CACHE_SIZE_MB = 2048

# ============================================================================
# SECURITY CONSTANTS
# ============================================================================

# Session settings
DEFAULT_SESSION_TIMEOUT_MINUTES = 60
MAX_SESSION_TIMEOUT_MINUTES = 480

# Authentication
MAX_LOGIN_ATTEMPTS = 3
LOCKOUT_DURATION_MINUTES = 15

# Data retention
DEFAULT_RETENTION_DAYS = 30
MIN_RETENTION_DAYS = 7
MAX_RETENTION_DAYS = 365

# ============================================================================
# ERROR CODES
# ============================================================================

# Success codes
SUCCESS = 0

# General error codes
ERROR_UNKNOWN = 1000
ERROR_INVALID_INPUT = 1001
ERROR_FILE_NOT_FOUND = 1002
ERROR_PERMISSION_DENIED = 1003
ERROR_CONFIGURATION_ERROR = 1004

# AI model error codes
ERROR_MODEL_LOAD_FAILED = 2000
ERROR_MODEL_INFERENCE_FAILED = 2001
ERROR_GPU_NOT_AVAILABLE = 2002
ERROR_INVALID_MODEL_PACK = 2003

# Database error codes
ERROR_DATABASE_CONNECTION = 3000
ERROR_DATABASE_QUERY = 3001
ERROR_EMPLOYEE_NOT_FOUND = 3002
ERROR_DUPLICATE_EMPLOYEE = 3003
ERROR_INVALID_ATTENDANCE_DATA = 3004

# Camera error codes
ERROR_CAMERA_NOT_FOUND = 4000
ERROR_CAMERA_CONNECTION_FAILED = 4001
ERROR_FRAME_CAPTURE_FAILED = 4002
ERROR_INVALID_VIDEO_FORMAT = 4003

# Processing error codes
ERROR_FACE_DETECTION_FAILED = 5000
ERROR_FACE_RECOGNITION_FAILED = 5001
ERROR_POOR_FACE_QUALITY = 5002
ERROR_MULTIPLE_FACES = 5003

# Notification error codes
ERROR_NOTIFICATION_SEND_FAILED = 6000
ERROR_INVALID_NOTIFICATION_CONFIG = 6001

# ============================================================================
# LOGGING CONSTANTS
# ============================================================================

# Log levels
LOG_LEVEL_DEBUG = 'DEBUG'
LOG_LEVEL_INFO = 'INFO'
LOG_LEVEL_WARNING = 'WARNING'
LOG_LEVEL_ERROR = 'ERROR'
LOG_LEVEL_CRITICAL = 'CRITICAL'

# Log formats
LOG_FORMAT_DETAILED = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
LOG_FORMAT_SIMPLE = '%(asctime)s - %(levelname)s - %(message)s'

# Log file settings
DEFAULT_LOG_FILE = 'attendance_system.log'
MAX_LOG_FILE_SIZE_MB = 10
MAX_LOG_FILES = 5

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================

# Environment types
ENV_COLAB = 'colab'
ENV_LOCAL = 'local'
ENV_DOCKER = 'docker'
ENV_PRODUCTION = 'production'

# Environment detection patterns
COLAB_PATTERNS = ['google.colab', '/content']
DOCKER_PATTERNS = ['/.dockerenv', '/proc/1/cgroup']

# ============================================================================
# EXPORT FORMATS
# ============================================================================

# Supported export formats
EXPORT_FORMAT_JSON = 'json'
EXPORT_FORMAT_CSV = 'csv'
EXPORT_FORMAT_EXCEL = 'xlsx'
EXPORT_FORMAT_PDF = 'pdf'

SUPPORTED_EXPORT_FORMATS = [
    EXPORT_FORMAT_JSON,
    EXPORT_FORMAT_CSV,
    EXPORT_FORMAT_EXCEL,
    EXPORT_FORMAT_PDF
]

# ============================================================================
# SYSTEM PATHS
# ============================================================================

# Get base directory
BASE_DIR = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / DEFAULT_DATA_DIR
LOGS_DIR = BASE_DIR / DEFAULT_LOGS_DIR
BACKUP_DIR = BASE_DIR / DEFAULT_BACKUP_DIR
SNAPSHOTS_DIR = BASE_DIR / DEFAULT_SNAPSHOTS_DIR
UPLOADS_DIR = BASE_DIR / DEFAULT_UPLOADS_DIR
EXPORTS_DIR = BASE_DIR / DEFAULT_EXPORTS_DIR

# Configuration paths
CONFIG_DIR = BASE_DIR / 'config'
DEFAULT_CONFIG_FILE = CONFIG_DIR / 'attendance_config.yaml'

# ============================================================================
# VALIDATION PATTERNS
# ============================================================================

# Email validation pattern
EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

# Employee code pattern (alphanumeric with underscore/dash)
EMPLOYEE_CODE_PATTERN = r'^[A-Za-z0-9_-]{3,20}$'

# Time format pattern (HH:MM)
TIME_FORMAT_PATTERN = r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$'

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_version_info():
    """Get formatted version information"""
    return {
        'version': VERSION,
        'system_name': SYSTEM_NAME,
        'pipeline_version': PIPELINE_VERSION,
        'repository_url': REPOSITORY_URL,
        'branch_name': BRANCH_NAME
    }

def is_colab_environment():
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return '/content' in os.getcwd()

def is_docker_environment():
    """Check if running in Docker container"""
    return (
        os.path.exists('/.dockerenv') or 
        os.path.exists('/proc/1/cgroup') and 
        'docker' in open('/proc/1/cgroup').read()
    )

def get_environment_type():
    """Detect current environment type"""
    if is_colab_environment():
        return ENV_COLAB
    elif is_docker_environment():
        return ENV_DOCKER
    elif os.getenv('ATTENDANCE_ENV') == 'production':
        return ENV_PRODUCTION
    else:
        return ENV_LOCAL


# Export commonly used constants
__all__ = [
    # Version info
    'VERSION', 'SYSTEM_NAME', 'PIPELINE_VERSION',
    
    # Model constants
    'AVAILABLE_MODEL_PACKS', 'DEFAULT_MODEL_PACK', 'FACE_EMBEDDING_DIMENSION',
    
    # Database constants
    'DB_TYPE_SQLITE', 'DB_TYPE_POSTGRESQL', 'DEFAULT_SQLITE_DB',
    'EVENT_TYPE_CHECK_IN', 'EVENT_TYPE_CHECK_OUT',
    
    # Business logic
    'DEFAULT_WORK_START', 'DEFAULT_WORK_END', 'DEFAULT_COOLDOWN_MINUTES',
    
    # Camera and video
    'DEFAULT_CAMERA_RESOLUTION', 'DEFAULT_CAMERA_FPS', 'SUPPORTED_VIDEO_FORMATS',
    
    # File system
    'DATA_DIR', 'LOGS_DIR', 'BACKUP_DIR', 'SNAPSHOTS_DIR',
    
    # Notifications
    'AVAILABLE_NOTIFICATION_CHANNELS', 'NOTIFICATION_TEMPLATES',
    
    # Performance
    'EXCELLENT_LATENCY_MS', 'GOOD_LATENCY_MS', 'TARGET_REAL_TIME_FPS',
    
    # Error codes
    'SUCCESS', 'ERROR_UNKNOWN', 'ERROR_MODEL_LOAD_FAILED',
    
    # Utility functions
    'get_version_info', 'is_colab_environment', 'get_environment_type'
] 