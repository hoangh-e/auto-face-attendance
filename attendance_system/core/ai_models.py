"""
AI Models Module for Attendance System Pipeline V1

Unified SCRFD + ArcFace interface for face detection and recognition
with comprehensive GPU optimization and performance monitoring.
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Dict, Optional, Tuple
import torch
import insightface
from insightface.app import FaceAnalysis
import psutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttendanceAIModels:
    """Unified SCRFD + ArcFace interface for Pipeline V1"""
    
    def __init__(self, model_pack='buffalo_l', ctx_id=None, det_size=(640, 640), det_thresh=0.5):
        """
        Initialize InsightFace with SCRFD + ArcFace
        
        Args:
            model_pack: Model pack name (buffalo_l, buffalo_m, buffalo_s, buffalo_sc)
            ctx_id: Context ID for GPU (0) or CPU (-1). Auto-detect if None
            det_size: Detection input size (width, height)
            det_thresh: Detection confidence threshold
        """
        self.model_pack = model_pack
        self.det_size = det_size
        self.det_thresh = det_thresh
        
        # Performance monitoring
        self.performance_stats = {
            'total_inferences': 0,
            'total_processing_time': 0.0,
            'avg_latency_ms': 0.0,
            'gpu_memory_mb': 0.0
        }
        
        logger.info(f"🤖 Initializing AI Attendance Models with {model_pack}")
        self._setup_gpu_providers()
        self._init_unified_model(ctx_id)
        self._test_performance()
        
        logger.info("✅ AI Models ready for GPU-accelerated attendance processing!")
    
    def _setup_gpu_providers(self):
        """Setup optimal providers for GPU acceleration"""
        # Check GPU availability
        if torch.cuda.is_available():
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.device_info = f"GPU: {torch.cuda.get_device_name(0)}"
            self.use_gpu = True
            logger.info(f"🚀 GPU Mode: {self.device_info}")
        else:
            self.providers = ['CPUExecutionProvider']
            self.device_info = "CPU Only"
            self.use_gpu = False
            logger.warning("⚠️ Fallback to CPU mode. Enable GPU for better performance.")
    
    def _init_unified_model(self, ctx_id=None):
        """Initialize Buffalo unified model pack with GPU"""
        try:
            # Auto-detect context ID if not provided
            if ctx_id is None:
                ctx_id = 0 if self.use_gpu else -1
            
            logger.info(f"📦 Loading {self.model_pack} model pack...")
            logger.info(f"├─ Detection: SCRFD")
            logger.info(f"├─ Recognition: ArcFace (512-dim)")
            logger.info(f"├─ Device: {self.device_info}")
            logger.info(f"└─ Context ID: {ctx_id}")
            
            # Initialize FaceAnalysis app with GPU providers
            self.app = FaceAnalysis(
                name=self.model_pack,
                providers=self.providers
            )
            
            # Prepare with optimal settings for attendance processing
            self.app.prepare(
                ctx_id=ctx_id,
                det_size=self.det_size,
                det_thresh=self.det_thresh
            )
            
            logger.info(f"✅ {self.model_pack} loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {self.model_pack}: {e}")
            logger.info("🔄 Trying fallback options...")
            self._fallback_initialization()
    
    def _fallback_initialization(self):
        """Fallback to simpler models if buffalo fails"""
        fallback_models = ['buffalo_sc', 'buffalo_s', 'buffalo_m']
        
        for model in fallback_models:
            if model != self.model_pack:
                try:
                    logger.info(f"🔄 Trying {model}...")
                    
                    self.app = FaceAnalysis(name=model, providers=self.providers)
                    ctx_id = 0 if self.use_gpu else -1
                    self.app.prepare(ctx_id=ctx_id, det_size=self.det_size, det_thresh=self.det_thresh)
                    
                    self.model_pack = model
                    logger.info(f"✅ Fallback successful: {model}")
                    return
                    
                except Exception as e:
                    logger.warning(f"❌ {model} also failed: {e}")
                    continue
        
        raise RuntimeError("❌ All model options failed. Please check InsightFace installation and GPU setup.")
    
    def detect_and_recognize(self, image: np.ndarray) -> List[Dict]:
        """
        Single call for detection + recognition
        Return formatted results with embeddings
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detection/recognition results with embeddings
        """
        start_time = time.time()
        
        try:
            # Single unified call for complete face analysis
            faces = self.app.get(image)
            
            # Format results for compatibility
            results = []
            for face in faces:
                result = {
                    'bbox': face.bbox,
                    'det_score': float(face.det_score),
                    'landmarks': getattr(face, 'kps', None),
                    'embedding': face.embedding,  # 512-dim ArcFace embedding
                    'age': getattr(face, 'age', None),
                    'gender': getattr(face, 'gender', None),
                    'pose': getattr(face, 'pose', None),
                    'embedding_norm': float(np.linalg.norm(face.embedding))
                }
                results.append(result)
            
            # Update performance stats
            processing_time = (time.time() - start_time) * 1000
            self._update_performance_stats(processing_time)
            
            return results
            
        except Exception as e:
            logger.error(f"Detection/Recognition error: {e}")
            return []
    
    def extract_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 512-dim embedding for registration
        
        Args:
            image: Input image as numpy array
            
        Returns:
            512-dimensional embedding vector or None if no face detected
        """
        try:
            results = self.detect_and_recognize(image)
            
            if len(results) > 0:
                # Return the embedding from the face with highest confidence
                best_face = max(results, key=lambda x: x['det_score'])
                return best_face['embedding']
            
            return None
            
        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
            return None
    
    def batch_process(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Batch processing for efficiency
        
        Args:
            images: List of input images
            
        Returns:
            List of results for each image
        """
        batch_results = []
        start_time = time.time()
        
        logger.info(f"🔄 Processing batch of {len(images)} images...")
        
        for i, image in enumerate(images):
            try:
                result = self.detect_and_recognize(image)
                batch_results.append({
                    'image_index': i,
                    'faces': result,
                    'face_count': len(result)
                })
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                batch_results.append({
                    'image_index': i,
                    'faces': [],
                    'face_count': 0,
                    'error': str(e)
                })
        
        total_time = time.time() - start_time
        logger.info(f"✅ Batch processing completed in {total_time:.2f}s")
        
        return batch_results
    
    def _update_performance_stats(self, processing_time_ms: float):
        """Update performance statistics"""
        self.performance_stats['total_inferences'] += 1
        self.performance_stats['total_processing_time'] += processing_time_ms
        self.performance_stats['avg_latency_ms'] = (
            self.performance_stats['total_processing_time'] / 
            self.performance_stats['total_inferences']
        )
        
        # Update GPU memory usage if available
        if self.use_gpu and torch.cuda.is_available():
            self.performance_stats['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
    
    def _test_performance(self):
        """Test GPU performance and accuracy"""
        logger.info("\n🧪 PERFORMANCE TEST:")
        
        try:
            # Create test image
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Warm up GPU
            for _ in range(3):
                _ = self.app.get(test_image)
            
            # Performance test with multiple runs
            times = []
            for i in range(5):
                start_time = time.time()
                faces = self.app.get(test_image)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
            
            avg_latency = np.mean(times)
            min_latency = np.min(times)
            
            logger.info(f"├─ Average Latency: {avg_latency:.1f}ms")
            logger.info(f"├─ Best Latency: {min_latency:.1f}ms")
            logger.info(f"├─ Device: {self.device_info}")
            logger.info(f"├─ Model Pack: {self.model_pack}")
            
            # Performance rating
            if avg_latency < 50:
                logger.info("🌟 Performance: EXCELLENT (<50ms)")
            elif avg_latency < 100:
                logger.info("👍 Performance: VERY GOOD (<100ms)")
            elif avg_latency < 200:
                logger.info("✅ Performance: GOOD (<200ms)")
            else:
                logger.warning("⚠️ Performance: Needs optimization (>200ms)")
                
        except Exception as e:
            logger.warning(f"⚠️ Performance test error: {e}")
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        stats = self.performance_stats.copy()
        
        # Add system information
        if self.use_gpu and torch.cuda.is_available():
            stats['gpu_name'] = torch.cuda.get_device_name(0)
            stats['gpu_memory_total_mb'] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        
        stats['model_pack'] = self.model_pack
        stats['device_info'] = self.device_info
        stats['cpu_usage_percent'] = psutil.cpu_percent()
        stats['ram_usage_mb'] = psutil.virtual_memory().used / 1024 / 1024
        
        return stats
    
    def benchmark_real_time_capability(self, target_fps: int = 30) -> Dict:
        """
        Benchmark real-time processing capability
        
        Args:
            target_fps: Target frames per second for real-time processing
            
        Returns:
            Benchmark results
        """
        logger.info(f"🎯 Benchmarking real-time capability for {target_fps} FPS...")
        
        # Target frame time in milliseconds
        target_frame_time = 1000 / target_fps
        
        # Create test images of different sizes
        test_sizes = [(320, 240), (640, 480), (1280, 720), (1920, 1080)]
        benchmark_results = {}
        
        for width, height in test_sizes:
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Test processing time
            times = []
            for _ in range(10):
                start_time = time.time()
                _ = self.detect_and_recognize(test_image)
                times.append((time.time() - start_time) * 1000)
            
            avg_time = np.mean(times)
            can_real_time = avg_time < target_frame_time
            max_fps = 1000 / avg_time if avg_time > 0 else float('inf')
            
            benchmark_results[f"{width}x{height}"] = {
                'avg_processing_time_ms': avg_time,
                'can_achieve_target_fps': can_real_time,
                'max_achievable_fps': max_fps,
                'target_fps': target_fps
            }
            
            status = "✅" if can_real_time else "❌"
            logger.info(f"├─ {width}x{height}: {avg_time:.1f}ms {status}")
        
        logger.info("✅ Benchmark completed")
        return benchmark_results
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'app'):
                del self.app
            
            # Clear GPU cache if available
            if self.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("✅ AI Models cleanup completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}") 