"""
Utility Functions for Attendance System Pipeline V1

Common utility functions for image processing, validation, formatting,
and other helper functions used across the system.
"""

import cv2
import numpy as np
import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_image(image: Union[np.ndarray, str, Path]) -> Optional[np.ndarray]:
    """
    Validate and load image from various sources
    
    Args:
        image: Image as numpy array, file path, or PIL Image
        
    Returns:
        Validated numpy array in BGR format or None if invalid
    """
    try:
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                return image
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # Convert RGBA to RGB
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:
                logger.warning("Invalid image shape")
                return None
        
        elif isinstance(image, (str, Path)):
            if not os.path.exists(image):
                logger.warning(f"Image file not found: {image}")
                return None
            
            img = cv2.imread(str(image))
            if img is None:
                logger.warning(f"Cannot read image: {image}")
                return None
            return img
        
        elif hasattr(image, 'mode'):  # PIL Image
            img_array = np.array(image)
            if image.mode == 'RGB':
                return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif image.mode == 'RGBA':
                return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else:
                return img_array
        
        else:
            logger.warning(f"Unsupported image type: {type(image)}")
            return None
            
    except Exception as e:
        logger.error(f"Image validation error: {e}")
        return None


def resize_image_with_aspect_ratio(image: np.ndarray, target_size: Tuple[int, int], 
                                 maintain_ratio: bool = True) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        target_size: Target (width, height)
        maintain_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    try:
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        if not maintain_ratio:
            return cv2.resize(image, target_size)
        
        # Calculate scale factor
        scale_width = target_width / width
        scale_height = target_height / height
        scale = min(scale_width, scale_height)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height))
        
        # Create canvas with target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Center the resized image on canvas
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return canvas
        
    except Exception as e:
        logger.error(f"Image resize error: {e}")
        return image


def crop_face_with_margin(image: np.ndarray, bbox: List[float], 
                         margin: float = 0.2) -> Optional[np.ndarray]:
    """
    Crop face from image with additional margin
    
    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        margin: Additional margin as ratio of face size
        
    Returns:
        Cropped face image or None
    """
    try:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = [int(x) for x in bbox[:4]]
        
        # Calculate margin
        face_width = x2 - x1
        face_height = y2 - y1
        margin_x = int(face_width * margin)
        margin_y = int(face_height * margin)
        
        # Expand bounding box with margin
        x1_expanded = max(0, x1 - margin_x)
        y1_expanded = max(0, y1 - margin_y)
        x2_expanded = min(width, x2 + margin_x)
        y2_expanded = min(height, y2 + margin_y)
        
        # Crop face
        face_crop = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
        
        if face_crop.size == 0:
            return None
        
        return face_crop
        
    except Exception as e:
        logger.error(f"Face crop error: {e}")
        return None


def draw_face_annotations(image: np.ndarray, face_results: List[Dict], 
                         show_landmarks: bool = False, 
                         show_attributes: bool = True) -> np.ndarray:
    """
    Draw face detection and recognition annotations
    
    Args:
        image: Input image
        face_results: List of face detection/recognition results
        show_landmarks: Whether to draw facial landmarks
        show_attributes: Whether to show age/gender attributes
        
    Returns:
        Annotated image
    """
    annotated = image.copy()
    
    try:
        for i, face in enumerate(face_results):
            bbox = face.get('bbox', [])
            if len(bbox) < 4:
                continue
            
            x1, y1, x2, y2 = [int(x) for x in bbox[:4]]
            
            # Determine color based on recognition status
            if face.get('employee_found', False):
                color = (0, 255, 0)  # Green for recognized
                name = face.get('employee_name', 'Unknown')
                similarity = face.get('recognition_similarity', 0.0)
                label = f"{name} ({similarity:.2f})"
            else:
                color = (0, 0, 255)  # Red for unknown
                conf = face.get('detection_confidence', 0.0)
                label = f"Unknown ({conf:.2f})"
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw quality score
            quality = face.get('face_quality_score', 0.0)
            quality_text = f"Q: {quality:.2f}"
            cv2.putText(annotated, quality_text, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw attributes if available and enabled
            if show_attributes:
                y_offset = 40
                
                if face.get('age') is not None:
                    age_text = f"Age: {face['age']}"
                    cv2.putText(annotated, age_text, (x1, y2 + y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    y_offset += 15
                
                if face.get('gender') is not None:
                    gender_text = f"Gender: {face['gender']}"
                    cv2.putText(annotated, gender_text, (x1, y2 + y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    y_offset += 15
            
            # Draw landmarks if available and enabled
            if show_landmarks and face.get('landmarks') is not None:
                landmarks = face['landmarks']
                if isinstance(landmarks, np.ndarray) and landmarks.shape[0] >= 5:
                    for point in landmarks:
                        cv2.circle(annotated, (int(point[0]), int(point[1])), 2, color, -1)
        
        return annotated
        
    except Exception as e:
        logger.error(f"Annotation error: {e}")
        return image


def format_timestamp(timestamp: Union[str, datetime] = None, 
                    format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Format timestamp to string
    
    Args:
        timestamp: Timestamp to format (None for current time)
        format_str: Format string
        
    Returns:
        Formatted timestamp string
    """
    try:
        if timestamp is None:
            timestamp = datetime.now()
        elif isinstance(timestamp, str):
            return timestamp
        
        return timestamp.strftime(format_str)
        
    except Exception as e:
        logger.error(f"Timestamp formatting error: {e}")
        return datetime.now().strftime(format_str)


def parse_timestamp(timestamp_str: str, 
                   format_str: str = '%Y-%m-%d %H:%M:%S') -> Optional[datetime]:
    """
    Parse timestamp string to datetime object
    
    Args:
        timestamp_str: Timestamp string
        format_str: Expected format
        
    Returns:
        Datetime object or None if parsing fails
    """
    try:
        return datetime.strptime(timestamp_str, format_str)
    except Exception as e:
        logger.error(f"Timestamp parsing error: {e}")
        return None


def calculate_time_difference(time1: str, time2: str, 
                            format_str: str = '%Y-%m-%d %H:%M:%S') -> Optional[timedelta]:
    """
    Calculate time difference between two timestamps
    
    Args:
        time1: First timestamp string
        time2: Second timestamp string
        format_str: Timestamp format
        
    Returns:
        Time difference or None if parsing fails
    """
    try:
        dt1 = parse_timestamp(time1, format_str)
        dt2 = parse_timestamp(time2, format_str)
        
        if dt1 and dt2:
            return abs(dt2 - dt1)
        return None
        
    except Exception as e:
        logger.error(f"Time difference calculation error: {e}")
        return None


def encode_image_base64(image: np.ndarray, 
                       format: str = '.jpg', 
                       quality: int = 90) -> Optional[str]:
    """
    Encode image to base64 string
    
    Args:
        image: Input image
        format: Image format ('.jpg', '.png', etc.)
        quality: JPEG quality (0-100)
        
    Returns:
        Base64 encoded string or None
    """
    try:
        encode_params = []
        if format.lower() in ['.jpg', '.jpeg']:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif format.lower() == '.png':
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - (quality // 10)]
        
        success, buffer = cv2.imencode(format, image, encode_params)
        if success:
            encoded = base64.b64encode(buffer).decode('utf-8')
            return encoded
        return None
        
    except Exception as e:
        logger.error(f"Image encoding error: {e}")
        return None


def decode_image_base64(encoded_string: str) -> Optional[np.ndarray]:
    """
    Decode base64 string to image
    
    Args:
        encoded_string: Base64 encoded image string
        
    Returns:
        Decoded image as numpy array or None
    """
    try:
        # Remove data URL prefix if present
        if ',' in encoded_string:
            encoded_string = encoded_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(encoded_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
        
    except Exception as e:
        logger.error(f"Image decoding error: {e}")
        return None


def calculate_image_hash(image: np.ndarray, algorithm: str = 'md5') -> Optional[str]:
    """
    Calculate hash of image for deduplication
    
    Args:
        image: Input image
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hash string or None
    """
    try:
        # Encode image to bytes
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            return None
        
        # Calculate hash
        if algorithm == 'md5':
            hash_obj = hashlib.md5()
        elif algorithm == 'sha1':
            hash_obj = hashlib.sha1()
        elif algorithm == 'sha256':
            hash_obj = hashlib.sha256()
        else:
            logger.warning(f"Unsupported hash algorithm: {algorithm}")
            return None
        
        hash_obj.update(buffer.tobytes())
        return hash_obj.hexdigest()
        
    except Exception as e:
        logger.error(f"Image hash calculation error: {e}")
        return None


def validate_employee_data(employee_data: Dict) -> Tuple[bool, List[str]]:
    """
    Validate employee registration data
    
    Args:
        employee_data: Employee data dictionary
        
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # Required fields
    required_fields = ['employee_code', 'name', 'email']
    for field in required_fields:
        if field not in employee_data or not employee_data[field]:
            errors.append(f"Missing required field: {field}")
    
    # Email validation (basic)
    email = employee_data.get('email', '')
    if email and '@' not in email:
        errors.append("Invalid email format")
    
    # Employee code validation
    employee_code = employee_data.get('employee_code', '')
    if employee_code and len(employee_code) < 3:
        errors.append("Employee code must be at least 3 characters")
    
    # Name validation
    name = employee_data.get('name', '')
    if name and len(name) < 2:
        errors.append("Name must be at least 2 characters")
    
    return len(errors) == 0, errors


def create_directory_structure(base_path: str, subdirs: List[str]) -> bool:
    """
    Create directory structure for the system
    
    Args:
        base_path: Base directory path
        subdirs: List of subdirectories to create
        
    Returns:
        True if successful, False otherwise
    """
    try:
        base_path = Path(base_path)
        base_path.mkdir(exist_ok=True)
        
        for subdir in subdirs:
            (base_path / subdir).mkdir(exist_ok=True)
        
        logger.info(f"Directory structure created: {base_path}")
        return True
        
    except Exception as e:
        logger.error(f"Directory creation error: {e}")
        return False


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    
    if i >= len(size_names):
        i = len(size_names) - 1
    
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


def cleanup_old_files(directory: str, max_age_days: int, pattern: str = "*") -> int:
    """
    Clean up old files in directory
    
    Args:
        directory: Directory path
        max_age_days: Maximum age in days
        pattern: File pattern to match
        
    Returns:
        Number of files deleted
    """
    try:
        directory_path = Path(directory)
        if not directory_path.exists():
            return 0
        
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        deleted_count = 0
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old files from {directory}")
        return deleted_count
        
    except Exception as e:
        logger.error(f"File cleanup error: {e}")
        return 0


def get_system_info() -> Dict:
    """
    Get system information
    
    Returns:
        Dictionary with system information
    """
    try:
        import psutil
        import platform
        
        info = {
            'platform': platform.platform(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_free_gb': psutil.disk_usage('/').free / (1024**3),
        }
        
        # GPU information if available
        try:
            import torch
            if torch.cuda.is_available():
                info['gpu_available'] = True
                info['gpu_name'] = torch.cuda.get_device_name(0)
                info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            else:
                info['gpu_available'] = False
        except:
            info['gpu_available'] = False
        
        return info
        
    except Exception as e:
        logger.error(f"System info error: {e}")
        return {}


# Export utility functions
__all__ = [
    'validate_image',
    'resize_image_with_aspect_ratio',
    'crop_face_with_margin',
    'draw_face_annotations',
    'format_timestamp',
    'parse_timestamp',
    'calculate_time_difference',
    'encode_image_base64',
    'decode_image_base64',
    'calculate_image_hash',
    'validate_employee_data',
    'create_directory_structure',
    'format_file_size',
    'cleanup_old_files',
    'get_system_info'
] 