#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Attendance System - Local Camera Demo (Fixed)
Real-time Camera Processing with Employee Folder Auto-Detection
Pipeline V1: SCRFD Detection + ArcFace Recognition + SQLite Database
"""

import sys
import subprocess
import time
import os
from pathlib import Path
import platform

print("⚡ AI ATTENDANCE SYSTEM - LOCAL CAMERA DEMO")
print("=" * 50)

# Environment Detection
print("🔍 ENVIRONMENT DETECTION:")
print(f"├─ Platform: {platform.system()} {platform.release()}")
print(f"├─ Python: {sys.version.split()[0]}")
print(f"├─ Working Directory: {os.getcwd()}")
print(f"└─ Architecture: {platform.machine()}")

# GPU Detection
try:
    import torch
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"🚀 GPU: ✅ {torch.cuda.get_device_name(0)}")
    else:
        print(f"💻 GPU: ❌ CPU mode (slower but functional)")
except ImportError:
    print(f"📦 PyTorch: Not installed, using CPU mode")
    gpu_available = False

# Install packages
print("\n📦 INSTALLING DEPENDENCIES:")

# Core packages for local demo
packages = [
    'torch', 'torchvision', 'insightface', 'opencv-python',
    'scikit-learn', 'matplotlib', 'pandas', 'tqdm', 'pillow',
    'plotly', 'psutil'
]

for pkg in packages:
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], 
                      timeout=60, check=True)
        print(f"├─ ✅ {pkg}")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        print(f"├─ ⚠️ {pkg} (skipped)")

# Core imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sqlite3
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm

# Try to import CV2 with better error handling
try:
    import cv2
    print("✅ OpenCV imported successfully")
except ImportError as e:
    print("❌ OpenCV import failed. Installing opencv-python...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'opencv-python', '--upgrade'])
        import cv2
        print("✅ OpenCV installed and imported successfully")
    except Exception as install_error:
        print(f"❌ Failed to install OpenCV: {install_error}")
        print("Please manually install: pip install opencv-python")
        raise

# AI and ML imports with error handling
try:
    import insightface
    from insightface.app import FaceAnalysis
    from sklearn.metrics.pairwise import cosine_similarity
    import psutil
    print("✅ AI libraries imported successfully")
except ImportError as e:
    print(f"❌ AI library import failed: {e}")
    print("Installing missing AI libraries...")
    packages = ['insightface', 'scikit-learn', 'psutil']
    for pkg in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
        except:
            print(f"⚠️ Failed to install {pkg}")
    
    # Try importing again
    import insightface
    from insightface.app import FaceAnalysis
    from sklearn.metrics.pairwise import cosine_similarity
    import psutil
    print("✅ AI libraries installed and imported")

# Create directory structure
base_dir = Path.cwd()
employees_dir = base_dir / 'employees'
data_dir = base_dir / 'data'
snapshots_dir = base_dir / 'snapshots'

for directory in [employees_dir, data_dir, snapshots_dir]:
    directory.mkdir(exist_ok=True)

print(f"\n📂 Directory structure created:")
print(f"├─ Employees: {employees_dir}")
print(f"├─ Data: {data_dir}")
print(f"└─ Snapshots: {snapshots_dir}")

print("🤖 AI SYSTEM & DATABASE SETUP")
print("=" * 35)

# Initialize AI Models for Local Processing
class LocalAISystem:
    def __init__(self):
        print("Initializing Local AI System...")
        self.app = None
        self.performance_stats = {
            'total_inferences': 0,
            'avg_latency_ms': 0.0,
            'total_time': 0.0
        }
        self._init_models()
    
    def _init_models(self):
        try:
            # Setup providers
            try:
                import torch
                if torch.cuda.is_available():
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    print(f"🚀 GPU Mode: {torch.cuda.get_device_name(0)}")
                else:
                    providers = ['CPUExecutionProvider']
                    print("💻 CPU Mode: Slower but functional")
            except ImportError:
                providers = ['CPUExecutionProvider']
                print("💻 CPU Mode: PyTorch not available, using CPU")
            
            # Initialize with lighter model for local demo
            model_pack = 'buffalo_s'  # Faster for local processing
            print(f"📦 Loading {model_pack} model pack...")
            
            self.app = FaceAnalysis(name=model_pack, providers=providers)
            try:
                import torch
                ctx_id = 0 if torch.cuda.is_available() else -1
            except ImportError:
                ctx_id = -1
            
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=0.5)
            
            print(f"✅ {model_pack} loaded successfully!")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print("Trying with basic CPU configuration...")
            try:
                self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
                self.app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.5)
                print("✅ Basic CPU model loaded successfully!")
            except Exception as e2:
                print(f"❌ Even basic model loading failed: {e2}")
                raise
    
    def detect_and_recognize(self, image):
        """Process image and return face data"""
        start_time = time.time()
        
        try:
            faces = self.app.get(image)
            
            results = []
            for face in faces:
                result = {
                    'bbox': face.bbox,
                    'det_score': face.det_score,
                    'landmarks': getattr(face, 'kps', None),
                    'embedding': face.embedding,
                    'age': getattr(face, 'age', None),
                    'gender': getattr(face, 'gender', None)
                }
                results.append(result)
            
            # Update performance stats
            processing_time = (time.time() - start_time) * 1000
            self.performance_stats['total_inferences'] += 1
            self.performance_stats['total_time'] += processing_time
            self.performance_stats['avg_latency_ms'] = (
                self.performance_stats['total_time'] / 
                self.performance_stats['total_inferences']
            )
            
            return results
            
        except Exception as e:
            print(f"Processing error: {e}")
            return []

# Initialize Local Database
class LocalDatabase:
    def __init__(self, db_path="local_attendance.db"):
        self.db_path = data_dir / db_path
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        print(f"🗄️ Local database: {self.db_path}")
    
    def _create_tables(self):
        cursor = self.conn.cursor()
        
        # Employees table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            department TEXT,
            position TEXT,
            face_embeddings TEXT,
            folder_path TEXT,
            face_count INTEGER DEFAULT 0,
            avg_quality REAL DEFAULT 0.0,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Attendance logs
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER,
            event_type TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            confidence REAL NOT NULL,
            snapshot_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (employee_id) REFERENCES employees (id)
        )
        """)
        
        self.conn.commit()
    
    def register_employee(self, employee_data, face_embeddings, folder_path):
        """Register employee with face embeddings"""
        cursor = self.conn.cursor()
        
        try:
            # Calculate average embedding
            if face_embeddings:
                avg_embedding = np.mean(face_embeddings, axis=0)
                avg_quality = np.mean([self._calculate_quality(emb) for emb in face_embeddings])
            else:
                return None
            
            cursor.execute("""
            INSERT INTO employees 
            (employee_code, name, email, department, position, face_embeddings, 
             folder_path, face_count, avg_quality)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                employee_data['employee_code'],
                employee_data['name'],
                employee_data['email'],
                employee_data.get('department', ''),
                employee_data.get('position', ''),
                json.dumps(avg_embedding.tolist()),
                str(folder_path),
                len(face_embeddings),
                avg_quality
            ))
            
            employee_id = cursor.lastrowid
            self.conn.commit()
            
            return employee_id
            
        except Exception as e:
            self.conn.rollback()
            print(f"Registration error: {e}")
            return None
    
    def _calculate_quality(self, embedding):
        """Simple embedding quality metric"""
        return min(np.linalg.norm(embedding) / 1.0, 1.0)
    
    def find_employee_by_embedding(self, query_embedding, threshold=0.65):
        """Find employee by face embedding"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT id, employee_code, name, email, face_embeddings
        FROM employees WHERE is_active = 1 AND face_embeddings IS NOT NULL
        """)
        
        best_match = None
        best_similarity = 0.0
        
        for row in cursor.fetchall():
            try:
                stored_embedding = np.array(json.loads(row['face_embeddings']))
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    stored_embedding.reshape(1, -1)
                )[0][0]
                
                if similarity > best_similarity and similarity > threshold:
                    best_similarity = similarity
                    best_match = {
                        'id': row['id'],
                        'employee_code': row['employee_code'],
                        'name': row['name'],
                        'email': row['email'],
                        'similarity': similarity
                    }
            except:
                continue
        
        return best_match
    
    def record_attendance(self, employee_id, event_type, confidence, snapshot_path=None):
        """Record attendance event"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT INTO attendance_logs (employee_id, event_type, timestamp, confidence, snapshot_path)
        VALUES (?, ?, ?, ?, ?)
        """, (employee_id, event_type, timestamp, confidence, snapshot_path))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_statistics(self):
        """Get database statistics"""
        cursor = self.conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM employees WHERE is_active = 1")
        stats['total_employees'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM attendance_logs")
        stats['total_logs'] = cursor.fetchone()[0]
        
        cursor.execute("""
        SELECT COUNT(*) FROM attendance_logs 
        WHERE DATE(timestamp) = DATE('now')
        """)
        stats['today_logs'] = cursor.fetchone()[0]
        
        return stats

# Employee Registration & Camera Setup
import glob
from collections import defaultdict
import threading
import queue

print("📁 EMPLOYEE REGISTRATION & CAMERA SETUP")
print("=" * 45)

class LocalEmployeeManager:
    """Smart employee folder scanning and registration"""
    
    def __init__(self, ai_system, database):
        self.ai_system = ai_system
        self.db = database
        self.registered_employees = {}
        self.scan_results = []
    
    def scan_employee_folders(self, auto_register=True):
        """Scan and optionally auto-register employees from folders"""
        print(f"🔍 Scanning employee folders in: {employees_dir}")
        
        if not employees_dir.exists():
            employees_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created employees directory: {employees_dir}")
            print(f"💡 Add employee folders: employees/John_Doe/photo1.jpg, photo2.jpg, ...")
            return
        
        # Find employee folders
        employee_folders = [f for f in employees_dir.iterdir() 
                           if f.is_dir() and not f.name.startswith('.') and not f.name.startswith('_')]
        
        if not employee_folders:
            print("⚠️ No employee folders found")
            print("Create folders like: employees/John_Doe/, employees/Jane_Smith/")
            return
        
        print(f"📁 Found {len(employee_folders)} employee folders:")
        
        for folder in employee_folders:
            print(f"  ├─ {folder.name}")
        
        # Process each folder
        total_registered = 0
        
        for folder in tqdm(employee_folders, desc="Processing employees"):
            result = self._process_employee_folder(folder, auto_register)
            if result:
                self.scan_results.append(result)
                if result['registered']:
                    total_registered += 1
        
        print(f"\n✅ Scan completed:")
        print(f"├─ Folders processed: {len(employee_folders)}")
        print(f"├─ Successfully registered: {total_registered}")
        print(f"└─ Database employees: {self.db.get_statistics()['total_employees']}")
        
        # Show summary table
        if self.scan_results:
            self._show_scan_summary()
    
    def _process_employee_folder(self, folder_path, auto_register=True):
        """Process single employee folder"""
        employee_name = folder_path.name.replace('_', ' ').title()
        
        # Find image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(folder_path.glob(ext))
        
        if not image_files:
            print(f"  ⚠️ {employee_name}: No images found")
            return None
        
        # Process images
        face_embeddings = []
        quality_scores = []
        valid_images = 0
        
        for img_file in image_files:
            try:
                # Load image
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                
                # Detect faces
                faces = self.ai_system.detect_and_recognize(image)
                
                if len(faces) == 1:  # Exactly one face
                    face_data = faces[0]
                    
                    if face_data['det_score'] > 0.6:  # Good quality threshold
                        face_embeddings.append(face_data['embedding'])
                        quality_scores.append(face_data['det_score'])
                        valid_images += 1
                
            except Exception as e:
                continue
        
        # Create result
        result = {
            'name': employee_name,
            'folder_path': str(folder_path),
            'total_images': len(image_files),
            'valid_images': valid_images,
            'avg_quality': np.mean(quality_scores) if quality_scores else 0,
            'registered': False,
            'employee_id': None
        }
        
        # Auto-register if sufficient quality faces
        if auto_register and len(face_embeddings) >= 1:
            employee_data = {
                'employee_code': employee_name.upper().replace(' ', '_'),
                'name': employee_name,
                'email': f"{employee_name.lower().replace(' ', '.')}@company.com",
                'department': 'Auto-Registered',
                'position': 'Employee'
            }
            
            employee_id = self.db.register_employee(
                employee_data, face_embeddings, folder_path
            )
            
            if employee_id:
                result['registered'] = True
                result['employee_id'] = employee_id
                self.registered_employees[employee_id] = result
                print(f"  ✅ {employee_name}: Registered ({valid_images} faces, avg quality: {result['avg_quality']:.3f})")
            else:
                print(f"  ❌ {employee_name}: Registration failed")
        else:
            print(f"  ⚠️ {employee_name}: Insufficient quality faces ({valid_images})")
        
        return result
    
    def _show_scan_summary(self):
        """Show employee scan summary table"""
        print(f"\n📋 EMPLOYEE SCAN SUMMARY")
        print("=" * 50)
        
        # Create summary DataFrame
        summary_data = []
        for result in self.scan_results:
            summary_data.append({
                'Employee': result['name'],
                'Total Images': result['total_images'],
                'Valid Faces': result['valid_images'],
                'Avg Quality': f"{result['avg_quality']:.3f}",
                'Registered': '✅' if result['registered'] else '❌',
                'Employee ID': result['employee_id'] or 'N/A'
            })
        
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))

class LocalCameraManager:
    """Enhanced camera management for local demo"""
    
    def __init__(self):
        self.camera = None
        self.camera_active = False
        self.camera_thread = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.latest_frame = None
        self.camera_stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'fps': 0,
            'last_fps_time': time.time()
        }
    
    def detect_cameras(self):
        """Detect available cameras"""
        print("🔍 Detecting available cameras...")
        
        available_cameras = []
        
        # Test camera indices 0-3
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    available_cameras.append({
                        'index': i,
                        'name': f'Camera {i}',
                        'resolution': f'{width}x{height}',
                        'fps': fps,
                        'working': True
                    })
                    
                    print(f"  ✅ Camera {i}: {width}x{height} @ {fps:.1f} FPS")
                cap.release()
            else:
                print(f"  ❌ Camera {i}: Not available")
        
        if not available_cameras:
            print("⚠️ No cameras detected!")
            print("💡 Try connecting a USB webcam or check camera permissions")
        
        return available_cameras
    
    def initialize_camera(self, camera_index=0):
        """Initialize specific camera"""
        print(f"📷 Initializing camera {camera_index}...")
        
        try:
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                print(f"❌ Cannot open camera {camera_index}")
                return False
            
            # Set optimal settings
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Test capture
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                print(f"❌ Cannot capture from camera {camera_index}")
                self.camera.release()
                return False
            
            # Get actual settings
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            print(f"✅ Camera initialized:")
            print(f"├─ Resolution: {width}x{height}")
            print(f"├─ Target FPS: {fps}")
            print(f"└─ Status: Ready")
            
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
            return False
    
    def start_camera_thread(self):
        """Start threaded camera capture"""
        if not self.camera or not self.camera.isOpened():
            print("❌ Camera not initialized")
            return False
        
        self.camera_active = True
        self.camera_thread = threading.Thread(target=self._camera_capture_loop, daemon=True)
        self.camera_thread.start()
        
        print("🎥 Camera thread started")
        return True
    
    def _camera_capture_loop(self):
        """Camera capture loop running in separate thread"""
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.camera_active and self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            
            if ret and frame is not None:
                self.latest_frame = frame.copy()
                self.camera_stats['frames_captured'] += 1
                
                # Update frame queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Remove oldest frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter >= 30:  # Update every 30 frames
                    current_time = time.time()
                    elapsed = current_time - fps_start_time
                    if elapsed > 0:
                        self.camera_stats['fps'] = fps_counter / elapsed
                    fps_counter = 0
                    fps_start_time = current_time
            
            time.sleep(0.001)  # Small delay to prevent busy waiting
    
    def get_latest_frame(self):
        """Get the most recent frame"""
        return self.latest_frame
    
    def stop_camera(self):
        """Stop camera capture"""
        self.camera_active = False
        
        if self.camera_thread:
            self.camera_thread.join(timeout=2.0)
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        print("📷 Camera stopped")
    
    def get_camera_stats(self):
        """Get camera performance statistics"""
        return self.camera_stats.copy()

# Initialize systems
try:
    ai_system = LocalAISystem()
    db = LocalDatabase()
    print("\n✅ Local systems initialized successfully!")
except Exception as e:
    print(f"\n❌ System initialization failed: {e}")
    print("Please check your installation and try again.")
    raise

print("\n📊 Current Database Statistics:")
stats = db.get_statistics()
print(f"├─ Employees: {stats['total_employees']}")
print(f"├─ Total Logs: {stats['total_logs']}")
print(f"└─ Today's Logs: {stats['today_logs']}")

# Initialize managers
employee_manager = LocalEmployeeManager(ai_system, db)
camera_manager = LocalCameraManager()

print("\n💡 EMPLOYEE FOLDER SETUP:")
print(f"1. Create folders in: {employees_dir}")
print(f"2. Folder structure: employees/John_Doe/photo1.jpg, photo2.jpg, ...")
print(f"3. Call employee_manager.scan_employee_folders() to auto-scan and register employees")

print("\n🎯 Available Actions:")
print("1. Scan employee folders: employee_manager.scan_employee_folders()")
print("2. Detect cameras: camera_manager.detect_cameras()")
print("3. Initialize camera: camera_manager.initialize_camera(0)")

print("\n💡 Quick Start:")
print("├─ Run employee_manager.scan_employee_folders() first")
print("└─ Then run camera_manager.detect_cameras()")
print("=" * 45)

if __name__ == "__main__":
    print("\n🚀 Starting demo...")
    
    # Example usage:
    print("\n1. Scanning for employees...")
    employee_manager.scan_employee_folders()
    
    print("\n2. Detecting cameras...")
    cameras = camera_manager.detect_cameras()
    
    if cameras:
        print(f"\n3. Initializing camera 0...")
        if camera_manager.initialize_camera(0):
            print("✅ Camera ready for real-time processing!")
            print("You can now start real-time attendance processing.")
        else:
            print("❌ Camera initialization failed")
    else:
        print("❌ No cameras found")
    
    print("\n�� Setup completed!") 