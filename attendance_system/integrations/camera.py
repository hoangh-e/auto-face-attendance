"""
Camera Management Module for Attendance System Pipeline V1

Enhanced camera input handling with error recovery, multiple camera support,
and automatic optimization for attendance processing.
"""

import cv2
import numpy as np
import time
import logging
import threading
from typing import List, Dict, Optional, Tuple, Iterator, Any
from datetime import datetime
import queue
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraManager:
    """Enhanced camera input handling with error recovery"""
    
    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (640, 480), 
                 fps: int = 30, buffer_size: int = 1):
        """
        Initialize camera with fallback options
        
        Args:
            camera_id: Camera device ID or video file path
            resolution: Target resolution (width, height)
            fps: Target frames per second
            buffer_size: Frame buffer size (1 for real-time)
        """
        self.camera_id = camera_id
        self.target_resolution = resolution
        self.target_fps = fps
        self.buffer_size = buffer_size
        self.cap = None
        self.is_opened = False
        
        # Camera information
        self.camera_info = {
            'id': camera_id,
            'type': 'unknown',
            'resolution': None,
            'fps': None,
            'backend': None,
            'fourcc': None
        }
        
        # Performance monitoring
        self.performance_stats = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'avg_fps': 0.0,
            'last_frame_time': 0.0,
            'connection_errors': 0,
            'reconnection_attempts': 0
        }
        
        # Threading for frame capture
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.capture_thread = None
        self.stop_capture = threading.Event()
        
        logger.info(f"📹 Initializing Camera Manager: {camera_id}")
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize camera with automatic detection and optimization"""
        try:
            # Determine camera type
            if isinstance(self.camera_id, str):
                if Path(self.camera_id).exists():
                    self.camera_info['type'] = 'video_file'
                    logger.info(f"📁 Video file source: {self.camera_id}")
                else:
                    # Might be RTSP stream or other URL
                    self.camera_info['type'] = 'stream'
                    logger.info(f"🌐 Stream source: {self.camera_id}")
            else:
                self.camera_info['type'] = 'webcam'
                logger.info(f"📷 Webcam source: {self.camera_id}")
            
            # Try different backends for better compatibility
            backends = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
            
            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(self.camera_id, backend)
                    if self.cap.isOpened():
                        self.camera_info['backend'] = backend
                        logger.info(f"✅ Camera opened with backend: {backend}")
                        break
                except Exception as e:
                    logger.warning(f"Backend {backend} failed: {e}")
                    continue
            
            if not self.cap or not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera: {self.camera_id}")
            
            # Configure camera settings
            self._configure_camera()
            
            # Get actual camera properties
            self._detect_camera_capabilities()
            
            self.is_opened = True
            logger.info("✅ Camera initialization successful")
            
        except Exception as e:
            logger.error(f"❌ Camera initialization failed: {e}")
            self.performance_stats['connection_errors'] += 1
            self.is_opened = False
            raise
    
    def _configure_camera(self):
        """Configure camera with optimal settings for attendance processing"""
        if not self.cap:
            return
        
        try:
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_resolution[1])
            
            # Set FPS
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Set buffer size for real-time processing
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            # Optimize for attendance processing
            if self.camera_info['type'] == 'webcam':
                # Auto-exposure and white balance for webcams
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Enable auto-exposure
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
            
            logger.info("🔧 Camera configured with optimal settings")
            
        except Exception as e:
            logger.warning(f"Camera configuration warning: {e}")
    
    def _detect_camera_capabilities(self):
        """Detect and log camera capabilities"""
        if not self.cap:
            return
        
        try:
            # Get actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            
            self.camera_info.update({
                'resolution': (actual_width, actual_height),
                'fps': actual_fps,
                'fourcc': fourcc
            })
            
            logger.info(f"📊 Camera capabilities detected:")
            logger.info(f"├─ Resolution: {actual_width}x{actual_height}")
            logger.info(f"├─ FPS: {actual_fps}")
            logger.info(f"├─ Backend: {self.camera_info['backend']}")
            logger.info(f"└─ Type: {self.camera_info['type']}")
            
        except Exception as e:
            logger.warning(f"Capability detection error: {e}")
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Get single frame with metadata
        
        Returns:
            Tuple of (frame, metadata) or None if failed
        """
        if not self.is_opened or not self.cap:
            return None
        
        try:
            start_time = time.time()
            ret, frame = self.cap.read()
            capture_time = time.time()
            
            if not ret or frame is None:
                logger.warning("Failed to capture frame")
                self.performance_stats['frames_dropped'] += 1
                return None
            
            # Update performance stats
            self.performance_stats['frames_captured'] += 1
            self.performance_stats['last_frame_time'] = capture_time
            
            # Calculate FPS
            if hasattr(self, '_last_frame_capture_time'):
                time_diff = capture_time - self._last_frame_capture_time
                if time_diff > 0:
                    current_fps = 1.0 / time_diff
                    # Exponential moving average for FPS
                    alpha = 0.1
                    self.performance_stats['avg_fps'] = (
                        alpha * current_fps + 
                        (1 - alpha) * self.performance_stats['avg_fps']
                    )
            
            self._last_frame_capture_time = capture_time
            
            # Create metadata
            metadata = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'frame_number': self.performance_stats['frames_captured'],
                'camera_id': self.camera_id,
                'resolution': frame.shape[:2][::-1],  # (width, height)
                'capture_time_ms': (capture_time - start_time) * 1000,
                'fps': self.performance_stats['avg_fps']
            }
            
            return frame, metadata
            
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            self.performance_stats['connection_errors'] += 1
            return None
    
    def stream_frames(self, max_fps: int = 30) -> Iterator[Tuple[np.ndarray, Dict]]:
        """
        Continuous frame streaming with FPS control
        
        Args:
            max_fps: Maximum frames per second
            
        Yields:
            Tuples of (frame, metadata)
        """
        logger.info(f"🎬 Starting frame streaming at max {max_fps} FPS")
        
        frame_interval = 1.0 / max_fps
        last_yield_time = 0
        
        try:
            while self.is_opened:
                current_time = time.time()
                
                # Control frame rate
                if current_time - last_yield_time >= frame_interval:
                    frame_data = self.get_frame()
                    
                    if frame_data:
                        frame, metadata = frame_data
                        metadata['stream_fps'] = max_fps
                        yield frame, metadata
                        last_yield_time = current_time
                    else:
                        # Try to reconnect if frame capture fails
                        if self._attempt_reconnection():
                            continue
                        else:
                            break
                
                # Small sleep to prevent excessive CPU usage
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            logger.info("⏹️ Frame streaming interrupted by user")
        except Exception as e:
            logger.error(f"❌ Frame streaming error: {e}")
        finally:
            logger.info("✅ Frame streaming completed")
    
    def start_threaded_capture(self):
        """Start threaded frame capture for better performance"""
        if self.capture_thread and self.capture_thread.is_alive():
            logger.warning("Threaded capture already running")
            return
        
        self.stop_capture.clear()
        self.capture_thread = threading.Thread(target=self._threaded_capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        logger.info("🔄 Started threaded frame capture")
    
    def stop_threaded_capture(self):
        """Stop threaded frame capture"""
        if self.capture_thread:
            self.stop_capture.set()
            self.capture_thread.join(timeout=5.0)
            
            # Clear frame queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            logger.info("⏹️ Stopped threaded frame capture")
    
    def _threaded_capture_loop(self):
        """Threaded capture loop"""
        while not self.stop_capture.is_set() and self.is_opened:
            try:
                frame_data = self.get_frame()
                
                if frame_data:
                    # Add to queue, remove old frame if queue is full
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                            self.performance_stats['frames_dropped'] += 1
                        except queue.Empty:
                            pass
                    
                    self.frame_queue.put(frame_data, timeout=0.01)
                
            except Exception as e:
                logger.error(f"Threaded capture error: {e}")
                time.sleep(0.1)  # Brief pause before retry
    
    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, Dict]]:
        """Get latest frame from threaded capture"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _attempt_reconnection(self, max_attempts: int = 3) -> bool:
        """Attempt to reconnect to camera"""
        logger.info(f"🔄 Attempting camera reconnection...")
        
        for attempt in range(max_attempts):
            try:
                self.performance_stats['reconnection_attempts'] += 1
                
                # Close existing connection
                if self.cap:
                    self.cap.release()
                
                # Wait before reconnection
                time.sleep(1.0 * (attempt + 1))
                
                # Reinitialize
                self._initialize_camera()
                
                if self.is_opened:
                    logger.info(f"✅ Reconnection successful (attempt {attempt + 1})")
                    return True
                    
            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")
        
        logger.error(f"❌ All reconnection attempts failed")
        self.is_opened = False
        return False
    
    def get_camera_info(self) -> Dict:
        """Get detailed camera information"""
        info = self.camera_info.copy()
        info['performance_stats'] = self.performance_stats.copy()
        info['is_opened'] = self.is_opened
        info['buffer_size'] = self.buffer_size
        info['target_resolution'] = self.target_resolution
        info['target_fps'] = self.target_fps
        
        return info
    
    def test_camera_performance(self, duration_seconds: int = 10) -> Dict:
        """
        Test camera performance over specified duration
        
        Args:
            duration_seconds: Test duration in seconds
            
        Returns:
            Performance test results
        """
        logger.info(f"🧪 Testing camera performance for {duration_seconds} seconds...")
        
        start_time = time.time()
        initial_frames = self.performance_stats['frames_captured']
        initial_drops = self.performance_stats['frames_dropped']
        
        frame_times = []
        
        while time.time() - start_time < duration_seconds:
            frame_start = time.time()
            frame_data = self.get_frame()
            frame_end = time.time()
            
            if frame_data:
                frame_times.append((frame_end - frame_start) * 1000)
            
            time.sleep(0.001)  # Small sleep to prevent excessive CPU usage
        
        end_time = time.time()
        test_duration = end_time - start_time
        
        # Calculate results
        frames_captured = self.performance_stats['frames_captured'] - initial_frames
        frames_dropped = self.performance_stats['frames_dropped'] - initial_drops
        
        results = {
            'test_duration_seconds': test_duration,
            'frames_captured': frames_captured,
            'frames_dropped': frames_dropped,
            'actual_fps': frames_captured / test_duration,
            'frame_drop_rate': frames_dropped / max(frames_captured + frames_dropped, 1),
            'avg_frame_time_ms': np.mean(frame_times) if frame_times else 0,
            'min_frame_time_ms': np.min(frame_times) if frame_times else 0,
            'max_frame_time_ms': np.max(frame_times) if frame_times else 0,
            'frame_time_std_ms': np.std(frame_times) if frame_times else 0
        }
        
        logger.info(f"📊 Performance test results:")
        logger.info(f"├─ Actual FPS: {results['actual_fps']:.1f}")
        logger.info(f"├─ Frame drop rate: {results['frame_drop_rate']*100:.1f}%")
        logger.info(f"├─ Avg frame time: {results['avg_frame_time_ms']:.1f}ms")
        logger.info(f"└─ Frames captured: {results['frames_captured']}")
        
        return results
    
    def save_frame(self, frame: np.ndarray, filename: str, 
                  metadata: Dict = None) -> bool:
        """
        Save frame to file with optional metadata
        
        Args:
            frame: Frame to save
            filename: Output filename
            metadata: Optional metadata to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save image
            success = cv2.imwrite(filename, frame)
            
            if success and metadata:
                # Save metadata as JSON
                metadata_file = Path(filename).with_suffix('.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            return success
            
        except Exception as e:
            logger.error(f"Frame save error: {e}")
            return False
    
    def release(self):
        """Cleanup with error handling"""
        try:
            logger.info("🧹 Releasing camera resources...")
            
            # Stop threaded capture if running
            self.stop_threaded_capture()
            
            # Release camera
            if self.cap:
                self.cap.release()
                self.cap = None
            
            self.is_opened = False
            
            logger.info("✅ Camera resources released successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Camera release warning: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
    
    def __del__(self):
        """Destructor"""
        self.release() 