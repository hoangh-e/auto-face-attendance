#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Attendance System - Improved Local Camera Demo
Production-ready version with comprehensive error handling
Pipeline V1: SCRFD Detection + ArcFace Recognition + SQLite Database
"""

# Standard library imports
import sys
import os
import time
import json
import sqlite3
import logging
import threading
import queue
import gc
import atexit
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import platform

# Third-party imports (with dependency checking)
def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'sklearn': 'scikit-learn',
        'insightface': 'insightface',
        'torch': 'torch',
        'PIL': 'pillow',
        'tqdm': 'tqdm',
        'psutil': 'psutil'
    }
    
    missing = []
    available = []
    
    print("🔍 DEPENDENCY CHECK:")
    print("=" * 30)
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
            available.append(package_name)
        except ImportError:
            print(f"❌ {package_name} - NOT INSTALLED")
            missing.append(package_name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing)}")
        print("\nOr install all at once:")
        print("pip install -r requirements.txt")
        return False
    
    print(f"\n✅ All {len(available)} dependencies satisfied!")
    return True

# Only proceed if dependencies are available
if not check_dependencies():
    print("\n🚨 Please install missing dependencies before running!")
    sys.exit(1)

# Now safe to import everything
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from sklearn.metrics.pairwise import cosine_similarity
import insightface
from insightface.app import FaceAnalysis
from PIL import Image
from tqdm import tqdm
import psutil

# System info
print(f"\n⚡ AI ATTENDANCE SYSTEM - IMPROVED VERSION")
print("=" * 50)
print(f"🔍 SYSTEM INFO:")
print(f"├─ Platform: {platform.system()} {platform.release()}")
print(f"├─ Python: {sys.version.split()[0]}")
print(f"├─ Working Directory: {os.getcwd()}")
print(f"├─ PyTorch: {torch.__version__}")
print(f"├─ OpenCV: {cv2.__version__}")

# GPU Detection
gpu_available = torch.cuda.is_available()
if gpu_available:
    print(f"├─ GPU: ✅ {torch.cuda.get_device_name(0)}")
else:
    print(f"├─ GPU: ❌ CPU mode")

print(f"└─ Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('attendance_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Directory setup
base_dir = Path.cwd()
employees_dir = base_dir / 'employees'
data_dir = base_dir / 'data'
snapshots_dir = base_dir / 'snapshots'
logs_dir = base_dir / 'logs'

# Create directories
for directory in [employees_dir, data_dir, snapshots_dir, logs_dir]:
    directory.mkdir(exist_ok=True)

print(f"\n📂 DIRECTORY STRUCTURE:")
print(f"├─ Employees: {employees_dir}")
print(f"├─ Data: {data_dir}")
print(f"├─ Snapshots: {snapshots_dir}")
print(f"└─ Logs: {logs_dir}")


class SafeAISystem:
    """Enhanced AI system with robust loading and error handling"""
    
    def __init__(self):
        self.app = None
        self.model_loaded = False
        self.performance_stats = {
            'total_inferences': 0,
            'avg_latency_ms': 0.0,
            'total_time': 0.0,
            'errors': 0
        }
        self.model_lock = threading.Lock()
        self._init_models()
    
    def _init_models(self):
        """Initialize AI models with retry logic and fallbacks"""
        max_retries = 3
        models_to_try = ['buffalo_s', 'buffalo_l']  # Try lightweight first
        
        print("\n🤖 AI MODEL INITIALIZATION:")
        print("=" * 30)
        
        for model_name in models_to_try:
            for attempt in range(max_retries):
                try:
                    print(f"Loading {model_name} model (attempt {attempt + 1}/{max_retries})...")
                    
                    # Determine providers
                    if gpu_available:
                        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                        ctx_id = 0
                        print(f"🚀 Using GPU acceleration")
                    else:
                        providers = ['CPUExecutionProvider']
                        ctx_id = -1
                        print(f"💻 Using CPU processing")
                    
                    # Initialize model
                    self.app = FaceAnalysis(name=model_name, providers=providers)
                    
                    # Prepare with timeout simulation
                    det_size = (640, 640) if gpu_available else (320, 320)
                    self.app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=0.5)
                    
                    # Verify with test image
                    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
                    test_faces = self.app.get(test_image)
                    
                    print(f"✅ Model {model_name} loaded successfully!")
                    print(f"├─ Detection size: {det_size}")
                    print(f"├─ Context: {'GPU' if ctx_id >= 0 else 'CPU'}")
                    print(f"└─ Providers: {providers}")
                    
                    self.model_loaded = True
                    return
                    
                except Exception as e:
                    print(f"❌ Attempt {attempt + 1} failed: {str(e)[:50]}...")
                    if attempt < max_retries - 1:
                        print("Retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        print(f"Failed to load {model_name}, trying next model...")
                        break
        
        # If all models failed
        raise RuntimeError("Failed to load any AI model. Please check InsightFace installation.")
    
    def detect_and_recognize(self, image: np.ndarray) -> List[Dict]:
        """Thread-safe face detection and recognition"""
        if not self.model_loaded:
            raise RuntimeError("AI model not loaded")
        
        start_time = time.time()
        
        try:
            with self.model_lock:
                faces = self.app.get(image)
            
            results = []
            for face in faces:
                result = {
                    'bbox': face.bbox.tolist(),
                    'det_score': float(face.det_score),
                    'landmarks': getattr(face, 'kps', None),
                    'embedding': face.embedding,
                    'age': getattr(face, 'age', None),
                    'gender': getattr(face, 'gender', None),
                    'quality_score': self._calculate_quality(face)
                }
                results.append(result)
            
            # Update performance stats
            processing_time = (time.time() - start_time) * 1000
            self.performance_stats['total_inferences'] += 1
            self.performance_stats['total_time'] += processing_time
            self.performance_stats['avg_latency_ms'] = (
                self.performance_stats['total_time'] / self.performance_stats['total_inferences']
            )
            
            return results
            
        except Exception as e:
            self.performance_stats['errors'] += 1
            logger.error(f"Face detection error: {e}")
            return []
    
    def _calculate_quality(self, face) -> float:
        """Calculate face quality score"""
        try:
            # Basic quality metrics
            det_score = face.det_score
            
            # Check if landmarks are available for pose estimation
            if hasattr(face, 'kps') and face.kps is not None:
                # Simple pose estimation based on landmarks
                landmarks = face.kps
                # Calculate face symmetry and frontality
                pose_score = min(1.0, det_score * 1.2)  # Simple heuristic
            else:
                pose_score = det_score
            
            # Combine scores
            quality = (det_score * 0.7 + pose_score * 0.3)
            return min(1.0, quality)
            
        except Exception:
            return float(face.det_score) if hasattr(face, 'det_score') else 0.5


class SafeDatabase:
    """Thread-safe database operations"""
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or (data_dir / 'attendance.db')
        self.connection_lock = threading.Lock()
        self._init_database()
    
    def _get_connection(self):
        """Get thread-safe database connection"""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_database(self):
        """Initialize database tables"""
        with self.connection_lock:
            conn = self._get_connection()
            try:
                # Employees table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS employees (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        embedding BLOB NOT NULL,
                        photos_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Attendance logs table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS attendance_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        employee_id INTEGER,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        confidence REAL,
                        image_path TEXT,
                        FOREIGN KEY (employee_id) REFERENCES employees (id)
                    )
                ''')
                
                # Performance logs table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS performance_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        processing_time_ms REAL,
                        faces_detected INTEGER,
                        memory_usage_mb REAL
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
            except Exception as e:
                logger.error(f"Database initialization error: {e}")
                raise
            finally:
                conn.close()
    
    def save_employee(self, name: str, embedding: np.ndarray, photos_count: int = 0) -> int:
        """Save employee embedding to database"""
        with self.connection_lock:
            conn = self._get_connection()
            try:
                # Convert embedding to bytes
                embedding_bytes = embedding.tobytes()
                
                # Insert or update employee
                conn.execute('''
                    INSERT OR REPLACE INTO employees (name, embedding, photos_count, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ''', (name, embedding_bytes, photos_count))
                
                employee_id = conn.lastrowid
                conn.commit()
                
                logger.info(f"Employee {name} saved with ID {employee_id}")
                return employee_id
                
            except Exception as e:
                logger.error(f"Error saving employee {name}: {e}")
                raise
            finally:
                conn.close()
    
    def get_all_employees(self) -> List[Dict]:
        """Get all employees with their embeddings"""
        with self.connection_lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute('''
                    SELECT id, name, embedding, photos_count, created_at
                    FROM employees
                    ORDER BY name
                ''')
                
                employees = []
                for row in cursor:
                    # Convert bytes back to numpy array
                    embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                    
                    employees.append({
                        'id': row['id'],
                        'name': row['name'],
                        'embedding': embedding,
                        'photos_count': row['photos_count'],
                        'created_at': row['created_at']
                    })
                
                return employees
                
            except Exception as e:
                logger.error(f"Error getting employees: {e}")
                return []
            finally:
                conn.close()
    
    def log_attendance(self, employee_id: int, confidence: float, image_path: str = None) -> bool:
        """Log attendance event"""
        with self.connection_lock:
            conn = self._get_connection()
            try:
                conn.execute('''
                    INSERT INTO attendance_logs (employee_id, confidence, image_path)
                    VALUES (?, ?, ?)
                ''', (employee_id, confidence, image_path))
                
                conn.commit()
                logger.info(f"Attendance logged for employee {employee_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error logging attendance: {e}")
                return False
            finally:
                conn.close()


class SafeCameraManager:
    """Enhanced camera management with robust error handling"""
    
    def __init__(self):
        self.camera = None
        self.camera_active = False
        self.camera_thread = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.camera_stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'fps': 0,
            'last_fps_time': time.time(),
            'dropped_frames': 0
        }
    
    def detect_cameras(self) -> List[Dict]:
        """Detect available cameras with detailed info"""
        print("\n📷 CAMERA DETECTION:")
        print("=" * 25)
        
        available_cameras = []
        
        # Test camera indices 0-3
        for i in range(4):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        camera_info = {
                            'index': i,
                            'name': f'Camera {i}',
                            'resolution': f'{width}x{height}',
                            'fps': fps,
                            'working': True
                        }
                        available_cameras.append(camera_info)
                        
                        print(f"✅ Camera {i}: {width}x{height} @ {fps:.1f} FPS")
                    cap.release()
                else:
                    print(f"❌ Camera {i}: Not accessible")
                    
            except Exception as e:
                print(f"❌ Camera {i}: Error - {e}")
        
        if not available_cameras:
            print("⚠️ No cameras detected!")
            print("💡 Troubleshooting:")
            print("  • Check camera connections")
            print("  • Verify camera permissions")
            print("  • Try connecting a USB webcam")
        
        return available_cameras
    
    def initialize_camera(self, camera_index: int = 0) -> bool:
        """Initialize specific camera with optimal settings"""
        print(f"\n📷 Initializing camera {camera_index}...")
        
        try:
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                logger.error(f"Cannot open camera {camera_index}")
                return False
            
            # Set optimal settings for face recognition
            settings = [
                (cv2.CAP_PROP_FRAME_WIDTH, 640),
                (cv2.CAP_PROP_FRAME_HEIGHT, 480),
                (cv2.CAP_PROP_FPS, 30),
                (cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
            ]
            
            for prop, value in settings:
                self.camera.set(prop, value)
            
            # Test capture
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                logger.error(f"Cannot capture from camera {camera_index}")
                self.camera.release()
                return False
            
            # Get actual settings
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            print(f"✅ Camera initialized:")
            print(f"├─ Resolution: {width}x{height}")
            print(f"├─ Target FPS: {fps}")
            print(f"└─ Buffer size: 1 (real-time)")
            
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            return False
    
    def start_camera_thread(self) -> bool:
        """Start threaded camera capture"""
        if not self.camera or not self.camera.isOpened():
            logger.error("Camera not initialized")
            return False
        
        self.camera_active = True
        self.camera_thread = threading.Thread(
            target=self._camera_capture_loop, 
            daemon=True,
            name="CameraThread"
        )
        self.camera_thread.start()
        
        print("🎥 Camera thread started")
        return True
    
    def _camera_capture_loop(self):
        """Camera capture loop with error recovery"""
        logger.info("Camera capture loop started")
        
        while self.camera_active:
            try:
                ret, frame = self.camera.read()
                
                if not ret or frame is None:
                    logger.warning("Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                # Update stats
                self.camera_stats['frames_captured'] += 1
                
                # Update latest frame (thread-safe)
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                
                # Add to queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Remove oldest frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                        self.camera_stats['dropped_frames'] += 1
                    except queue.Empty:
                        pass
                
                # Calculate FPS
                current_time = time.time()
                if current_time - self.camera_stats['last_fps_time'] >= 1.0:
                    self.camera_stats['fps'] = self.camera_stats['frames_captured']
                    self.camera_stats['frames_captured'] = 0
                    self.camera_stats['last_fps_time'] = current_time
                
                # Small delay to prevent CPU overload
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Camera capture error: {e}")
                time.sleep(0.5)  # Longer delay on error
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get latest frame (thread-safe)"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def stop_camera(self):
        """Stop camera capture safely"""
        print("🔴 Stopping camera...")
        self.camera_active = False
        
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        print("✅ Camera stopped successfully")


class ImprovedEmployeeManager:
    """Enhanced employee management with better scanning and processing"""
    
    def __init__(self, ai_system: SafeAISystem, database: SafeDatabase):
        self.ai_system = ai_system
        self.database = database
        self.employees_data = {}
        self.processing_stats = {
            'total_employees': 0,
            'total_photos': 0,
            'successful_extractions': 0,
            'failed_extractions': 0
        }
    
    def scan_employee_folders(self) -> Dict:
        """Scan employee folders and process images"""
        print("\n👥 EMPLOYEE PROCESSING:")
        print("=" * 25)
        
        if not employees_dir.exists():
            print(f"❌ Employees directory not found: {employees_dir}")
            return {}
        
        employee_folders = [d for d in employees_dir.iterdir() if d.is_dir()]
        
        if not employee_folders:
            print("❌ No employee folders found!")
            print("💡 Create folders like: employees/John_Doe/")
            print("   Add 2+ photos per employee")
            return {}
        
        print(f"Found {len(employee_folders)} employee folders")
        
        for folder in tqdm(employee_folders, desc="Processing employees"):
            employee_name = folder.name
            
            try:
                result = self._process_employee_folder(folder)
                
                if result['success']:
                    self.employees_data[employee_name] = result
                    print(f"✅ {employee_name}: {result['photos_processed']} photos processed")
                else:
                    print(f"❌ {employee_name}: {result['error']}")
                    
            except Exception as e:
                print(f"❌ {employee_name}: Unexpected error - {e}")
                self.processing_stats['failed_extractions'] += 1
        
        # Save to database
        self._save_to_database()
        
        # Show summary
        self._show_processing_summary()
        
        return self.employees_data
    
    def _process_employee_folder(self, folder: Path) -> Dict:
        """Process all images in an employee folder"""
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            f for f in folder.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if len(image_files) < 2:
            return {
                'success': False,
                'error': f'Need at least 2 photos (found {len(image_files)})',
                'photos_processed': 0
            }
        
        # Process each image
        embeddings = []
        processed_count = 0
        
        for image_file in image_files:
            try:
                # Load and validate image
                image = cv2.imread(str(image_file))
                if image is None:
                    continue
                
                # Get face embeddings
                faces = self.ai_system.detect_and_recognize(image)
                
                if faces:
                    # Use the best quality face
                    best_face = max(faces, key=lambda f: f['quality_score'])
                    
                    if best_face['quality_score'] > 0.3:  # Quality threshold
                        embeddings.append(best_face['embedding'])
                        processed_count += 1
                
            except Exception as e:
                logger.warning(f"Error processing {image_file}: {e}")
                continue
        
        if not embeddings:
            return {
                'success': False,
                'error': 'No valid faces found in photos',
                'photos_processed': processed_count
            }
        
        # Calculate average embedding
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Normalize embedding
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        self.processing_stats['total_employees'] += 1
        self.processing_stats['total_photos'] += len(image_files)
        self.processing_stats['successful_extractions'] += processed_count
        
        return {
            'success': True,
            'embedding': avg_embedding,
            'photos_processed': processed_count,
            'photos_total': len(image_files),
            'quality_scores': [np.mean([f['quality_score'] for f in faces]) for faces in [faces]],
            'error': None
        }
    
    def _save_to_database(self):
        """Save all processed employees to database"""
        print("\n💾 Saving to database...")
        
        saved_count = 0
        for name, data in self.employees_data.items():
            if data['success']:
                try:
                    self.database.save_employee(
                        name=name,
                        embedding=data['embedding'],
                        photos_count=data['photos_processed']
                    )
                    saved_count += 1
                except Exception as e:
                    logger.error(f"Failed to save {name}: {e}")
        
        print(f"✅ Saved {saved_count} employees to database")
    
    def _show_processing_summary(self):
        """Show processing statistics"""
        stats = self.processing_stats
        
        print(f"\n📊 PROCESSING SUMMARY:")
        print("=" * 25)
        print(f"├─ Total employees: {stats['total_employees']}")
        print(f"├─ Total photos: {stats['total_photos']}")
        print(f"├─ Successful extractions: {stats['successful_extractions']}")
        print(f"├─ Failed extractions: {stats['failed_extractions']}")
        
        if stats['total_photos'] > 0:
            success_rate = (stats['successful_extractions'] / stats['total_photos']) * 100
            print(f"└─ Success rate: {success_rate:.1f}%")


class RealTimeProcessor:
    """Real-time attendance processing"""
    
    def __init__(self, ai_system: SafeAISystem, database: SafeDatabase, camera_manager: SafeCameraManager):
        self.ai_system = ai_system
        self.database = database
        self.camera_manager = camera_manager
        self.known_employees = []
        self.recognition_threshold = 0.4
        self.cooldown_period = 30  # seconds
        self.last_recognition = {}
        self.processing_active = False
        self.processing_thread = None
        
        # Load known employees
        self._load_known_employees()
    
    def _load_known_employees(self):
        """Load known employees from database"""
        self.known_employees = self.database.get_all_employees()
        print(f"📚 Loaded {len(self.known_employees)} known employees")
    
    def recognize_face(self, face_embedding: np.ndarray) -> Tuple[str, float]:
        """Recognize face against known employees"""
        if not self.known_employees:
            return "Unknown", 0.0
        
        best_match = "Unknown"
        best_similarity = 0.0
        
        for employee in self.known_employees:
            # Calculate cosine similarity
            similarity = cosine_similarity(
                face_embedding.reshape(1, -1),
                employee['embedding'].reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = employee['name']
        
        # Apply threshold
        if best_similarity < self.recognition_threshold:
            return "Unknown", best_similarity
        
        return best_match, best_similarity
    
    def start_real_time_processing(self):
        """Start real-time attendance processing"""
        if not self.camera_manager.camera_active:
            print("❌ Camera not active")
            return False
        
        self.processing_active = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="ProcessingThread"
        )
        self.processing_thread.start()
        
        print("🔄 Real-time processing started")
        return True
    
    def _processing_loop(self):
        """Main processing loop"""
        logger.info("Real-time processing loop started")
        
        while self.processing_active:
            try:
                frame = self.camera_manager.get_latest_frame()
                
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Process frame
                self._process_frame(frame)
                
                # Update camera stats
                self.camera_manager.camera_stats['frames_processed'] += 1
                
                # Control processing rate
                time.sleep(0.1)  # 10 FPS processing
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                time.sleep(1.0)
    
    def _process_frame(self, frame: np.ndarray):
        """Process a single frame for attendance"""
        try:
            # Detect faces
            faces = self.ai_system.detect_and_recognize(frame)
            
            current_time = time.time()
            
            for face in faces:
                # Check quality threshold
                if face['quality_score'] < 0.5:
                    continue
                
                # Recognize face
                name, confidence = self.recognize_face(face['embedding'])
                
                if name != "Unknown":
                    # Check cooldown
                    last_seen = self.last_recognition.get(name, 0)
                    
                    if current_time - last_seen > self.cooldown_period:
                        # Log attendance
                        employee = next((e for e in self.known_employees if e['name'] == name), None)
                        
                        if employee:
                            # Save snapshot
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            snapshot_path = snapshots_dir / f"{name}_{timestamp}.jpg"
                            cv2.imwrite(str(snapshot_path), frame)
                            
                            # Log to database
                            self.database.log_attendance(
                                employee_id=employee['id'],
                                confidence=confidence,
                                image_path=str(snapshot_path)
                            )
                            
                            # Update last recognition
                            self.last_recognition[name] = current_time
                            
                            logger.info(f"Attendance logged: {name} (confidence: {confidence:.3f})")
                            print(f"✅ {name} detected (confidence: {confidence:.3f})")
                
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
    
    def stop_processing(self):
        """Stop real-time processing"""
        print("🔴 Stopping real-time processing...")
        self.processing_active = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        print("✅ Real-time processing stopped")


# Global cleanup function
def cleanup():
    """Cleanup function called on exit"""
    try:
        if 'camera_manager' in globals():
            camera_manager.stop_camera()
        
        if 'processor' in globals():
            processor.stop_processing()
        
        # Force garbage collection
        gc.collect()
        
        print("✅ Cleanup completed")
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

# Register cleanup function
atexit.register(cleanup)


# Main execution
if __name__ == "__main__":
    print("\n🚀 STARTING IMPROVED ATTENDANCE SYSTEM")
    print("=" * 50)
    
    try:
        # Initialize components
        print("Initializing AI system...")
        ai_system = SafeAISystem()
        
        print("Initializing database...")
        database = SafeDatabase()
        
        print("Initializing camera manager...")
        camera_manager = SafeCameraManager()
        
        print("Initializing employee manager...")
        employee_manager = ImprovedEmployeeManager(ai_system, database)
        
        # Step 1: Scan employees
        print("\n" + "="*50)
        print("STEP 1: EMPLOYEE REGISTRATION")
        print("="*50)
        employees = employee_manager.scan_employee_folders()
        
        if not employees:
            print("\n⚠️ No employees processed!")
            print("Please add employee photos and try again.")
            sys.exit(1)
        
        # Step 2: Setup cameras
        print("\n" + "="*50)
        print("STEP 2: CAMERA SETUP")
        print("="*50)
        cameras = camera_manager.detect_cameras()
        
        if not cameras:
            print("\n❌ No cameras found!")
            print("Please connect a camera and try again.")
            sys.exit(1)
        
        # Initialize first available camera
        if camera_manager.initialize_camera(cameras[0]['index']):
            if camera_manager.start_camera_thread():
                print("✅ Camera system ready!")
            else:
                print("❌ Failed to start camera thread")
                sys.exit(1)
        else:
            print("❌ Failed to initialize camera")
            sys.exit(1)
        
        # Step 3: Real-time processing
        print("\n" + "="*50)
        print("STEP 3: REAL-TIME PROCESSING")
        print("="*50)
        
        processor = RealTimeProcessor(ai_system, database, camera_manager)
        
        if processor.start_real_time_processing():
            print("✅ Real-time processing started!")
            
            print("\n🎉 SYSTEM FULLY OPERATIONAL!")
            print("=" * 30)
            print("📊 System Status:")
            print(f"├─ Employees registered: {len(employees)}")
            print(f"├─ Camera active: {camera_manager.camera_active}")
            print(f"├─ Processing active: {processor.processing_active}")
            print(f"└─ Recognition threshold: {processor.recognition_threshold}")
            
            print("\n💡 Controls:")
            print("├─ Press Ctrl+C to stop")
            print("├─ Check 'snapshots/' for captured images")
            print("├─ Check 'attendance_system.log' for logs")
            print("└─ Database: data/attendance.db")
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
                    
                    # Show stats every 30 seconds
                    if int(time.time()) % 30 == 0:
                        stats = camera_manager.camera_stats
                        print(f"\n📊 Stats: FPS={stats['fps']}, Processed={stats['frames_processed']}, Dropped={stats['dropped_frames']}")
                    
            except KeyboardInterrupt:
                print("\n🔴 Stopping system...")
                
        else:
            print("❌ Failed to start real-time processing")
            
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"❌ System error: {e}")
        
    finally:
        cleanup()
        print("\n👋 System shutdown complete!") 