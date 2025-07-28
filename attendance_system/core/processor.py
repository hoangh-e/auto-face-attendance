"""
Attendance Processing Module for Pipeline V1

Core business logic for attendance processing with comprehensive error handling,
performance monitoring, and sophisticated business rules.
"""

import cv2
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Iterator, Any
import json
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttendanceProcessor:
    """Core business logic for attendance processing with comprehensive error handling"""
    
    def __init__(self, ai_models, database, config: Dict = None):
        """
        Initialize processor with AI models and database instances
        
        Args:
            ai_models: AttendanceAIModels instance
            database: AttendanceDatabaseSQLite instance
            config: Configuration dictionary
        """
        self.ai_models = ai_models
        self.database = database
        
        # Default configuration
        self.config = {
            'cooldown_minutes': 30,
            'work_hours_start': '07:00',
            'work_hours_end': '19:00',
            'recognition_threshold': 0.65,
            'face_quality_threshold': 0.3,
            'max_faces_per_frame': 10,
            'enable_anti_spoofing': False,
            'enable_performance_monitoring': True,
            'snapshot_enabled': True,
            'snapshot_directory': 'snapshots'
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Performance monitoring
        self.performance_stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'faces_recognized': 0,
            'attendance_recorded': 0,
            'avg_processing_time_ms': 0.0,
            'total_processing_time': 0.0,
            'errors_count': 0
        }
        
        # Create snapshot directory
        if self.config['snapshot_enabled']:
            Path(self.config['snapshot_directory']).mkdir(exist_ok=True)
        
        logger.info("⚙️ Attendance Processor initialized")
        logger.info(f"├─ Cooldown: {self.config['cooldown_minutes']} minutes")
        logger.info(f"├─ Work hours: {self.config['work_hours_start']} - {self.config['work_hours_end']}")
        logger.info(f"├─ Recognition threshold: {self.config['recognition_threshold']}")
        logger.info(f"└─ Quality threshold: {self.config['face_quality_threshold']}")
    
    def process_frame(self, frame: np.ndarray, timestamp: str = None, 
                     camera_id: str = None) -> Dict:
        """
        Main pipeline with detailed processing info
        
        Args:
            frame: Input frame as numpy array
            timestamp: Optional timestamp string
            camera_id: Optional camera identifier
            
        Returns:
            Comprehensive result with metrics and attendance data
        """
        start_time = time.time()
        
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        result = {
            'timestamp': timestamp,
            'camera_id': camera_id,
            'frame_shape': frame.shape if frame is not None else None,
            'faces_detected': 0,
            'recognitions': [],
            'attendance_events': [],
            'processing_info': {
                'total_time_ms': 0.0,
                'ai_time_ms': 0.0,
                'db_time_ms': 0.0,
                'business_logic_time_ms': 0.0
            },
            'status': 'success',
            'message': '',
            'errors': []
        }
        
        try:
            # Validate input
            if frame is None or frame.size == 0:
                result['status'] = 'error'
                result['message'] = 'Invalid frame input'
                return result
            
            # 1. AI Processing (Detection + Recognition)
            ai_start = time.time()
            face_results = self.ai_models.detect_and_recognize(frame)
            ai_time = (time.time() - ai_start) * 1000
            result['processing_info']['ai_time_ms'] = ai_time
            
            result['faces_detected'] = len(face_results)
            
            if len(face_results) == 0:
                result['message'] = 'No faces detected'
                result['processing_info']['total_time_ms'] = (time.time() - start_time) * 1000
                self._update_performance_stats(result)
                return result
            
            # Limit number of faces processed
            if len(face_results) > self.config['max_faces_per_frame']:
                face_results = sorted(face_results, key=lambda x: x['det_score'], reverse=True)
                face_results = face_results[:self.config['max_faces_per_frame']]
                logger.warning(f"⚠️ Limited to {self.config['max_faces_per_frame']} faces per frame")
            
            # 2. Process each detected face
            business_logic_start = time.time()
            
            for i, face_data in enumerate(face_results):
                try:
                    face_result = self._process_single_face(
                        frame, face_data, timestamp, camera_id, i
                    )
                    result['recognitions'].append(face_result)
                    
                    # Check for attendance events
                    if face_result['should_record_attendance']:
                        attendance_event = self._record_attendance_event(
                            face_result, timestamp, camera_id
                        )
                        if attendance_event:
                            result['attendance_events'].append(attendance_event)
                
                except Exception as e:
                    error_msg = f"Error processing face {i}: {str(e)}"
                    logger.error(error_msg)
                    result['errors'].append(error_msg)
            
            business_logic_time = (time.time() - business_logic_start) * 1000
            result['processing_info']['business_logic_time_ms'] = business_logic_time
            
            # 3. Generate summary message
            result['message'] = self._generate_summary_message(result)
            
        except Exception as e:
            result['status'] = 'error'
            result['message'] = f'Processing error: {str(e)}'
            result['errors'].append(str(e))
            logger.error(f"❌ Frame processing error: {e}")
            self.performance_stats['errors_count'] += 1
        
        # Finalize timing
        result['processing_info']['total_time_ms'] = (time.time() - start_time) * 1000
        
        # Update performance statistics
        self._update_performance_stats(result)
        
        return result
    
    def process_video_stream(self, video_source, max_fps: int = 30) -> Iterator[Dict]:
        """
        Process video stream with frame-by-frame results
        
        Args:
            video_source: Video source (file path, camera index, etc.)
            max_fps: Maximum processing FPS
            
        Yields:
            Processing results for each frame
        """
        logger.info(f"🎬 Starting video stream processing: {video_source}")
        
        # Open video source
        if isinstance(video_source, (int, str)):
            cap = cv2.VideoCapture(video_source)
        else:
            cap = video_source
        
        if not cap.isOpened():
            logger.error(f"❌ Cannot open video source: {video_source}")
            return
        
        frame_interval = 1.0 / max_fps
        last_process_time = 0
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                
                # Control processing rate
                if current_time - last_process_time >= frame_interval:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    result = self.process_frame(frame, timestamp, f"stream_{video_source}")
                    result['frame_number'] = frame_count
                    
                    yield result
                    
                    last_process_time = current_time
                
                frame_count += 1
                
        except KeyboardInterrupt:
            logger.info("⏹️ Video processing interrupted by user")
        except Exception as e:
            logger.error(f"❌ Video processing error: {e}")
        finally:
            if isinstance(video_source, (int, str)):
                cap.release()
            logger.info(f"✅ Video processing completed. Frames processed: {frame_count}")
    
    def _process_single_face(self, frame: np.ndarray, face_data: Dict, 
                           timestamp: str, camera_id: str, face_index: int) -> Dict:
        """Process single detected face with comprehensive analysis"""
        
        face_result = {
            'face_index': face_index,
            'bbox': face_data['bbox'].tolist() if hasattr(face_data['bbox'], 'tolist') else face_data['bbox'],
            'detection_confidence': float(face_data['det_score']),
            'landmarks': face_data.get('landmarks'),
            'age': face_data.get('age'),
            'gender': face_data.get('gender'),
            
            # Recognition results
            'employee_found': False,
            'employee_id': None,
            'employee_name': None,
            'employee_code': None,
            'recognition_similarity': 0.0,
            'registration_quality': 0.0,
            
            # Quality assessment
            'face_quality_score': 0.0,
            'quality_issues': [],
            
            # Business logic
            'should_record_attendance': False,
            'attendance_decision_reason': '',
            'suggested_event_type': None,
            
            # Technical details
            'embedding_norm': 0.0,
            'processing_notes': []
        }
        
        try:
            # 1. Quality assessment
            quality_result = self._assess_face_quality(frame, face_data)
            face_result.update(quality_result)
            
            if face_result['face_quality_score'] < self.config['face_quality_threshold']:
                face_result['processing_notes'].append(
                    f"Low quality score: {face_result['face_quality_score']:.3f}"
                )
                return face_result
            
            # 2. Face recognition
            embedding = face_data['embedding']
            face_result['embedding_norm'] = float(np.linalg.norm(embedding))
            
            db_start = time.time()
            employee = self.database.find_employee_by_embedding(
                embedding, self.config['recognition_threshold']
            )
            db_time = (time.time() - db_start) * 1000
            
            if employee:
                face_result['employee_found'] = True
                face_result['employee_id'] = employee['id']
                face_result['employee_name'] = employee['name']
                face_result['employee_code'] = employee['employee_code']
                face_result['recognition_similarity'] = employee['similarity']
                face_result['registration_quality'] = employee.get('registration_quality', 0.0)
                
                # 3. Business logic evaluation
                business_logic_result = self._evaluate_business_logic(
                    employee['id'], timestamp
                )
                face_result.update(business_logic_result)
                
            else:
                face_result['attendance_decision_reason'] = 'Employee not recognized'
            
            # 4. Anti-spoofing check (if enabled)
            if self.config['enable_anti_spoofing']:
                spoofing_result = self._check_anti_spoofing(frame, face_data)
                face_result.update(spoofing_result)
            
        except Exception as e:
            face_result['processing_notes'].append(f"Processing error: {str(e)}")
            logger.error(f"Error processing face {face_index}: {e}")
        
        return face_result
    
    def _assess_face_quality(self, frame: np.ndarray, face_data: Dict) -> Dict:
        """Comprehensive face quality assessment"""
        
        quality_result = {
            'face_quality_score': 0.0,
            'quality_metrics': {},
            'quality_issues': []
        }
        
        try:
            bbox = face_data['bbox']
            if hasattr(bbox, 'tolist'):
                bbox = bbox.tolist()
            
            x1, y1, x2, y2 = [int(x) for x in bbox[:4]]
            
            # Ensure bbox is within frame bounds
            height, width = frame.shape[:2]
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(x1+1, min(x2, width))
            y2 = max(y1+1, min(y2, height))
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                quality_result['quality_issues'].append('Invalid face crop')
                return quality_result
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            
            # 1. Sharpness analysis (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 500, 1.0)
            quality_result['quality_metrics']['sharpness'] = sharpness_score
            
            if sharpness_score < 0.3:
                quality_result['quality_issues'].append('Blurry image')
            
            # 2. Brightness assessment
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            quality_result['quality_metrics']['brightness'] = brightness_score
            
            if brightness < 50:
                quality_result['quality_issues'].append('Too dark')
            elif brightness > 200:
                quality_result['quality_issues'].append('Too bright')
            
            # 3. Face size validation
            face_area = (x2 - x1) * (y2 - y1)
            frame_area = width * height
            face_ratio = face_area / frame_area
            size_score = min(face_ratio * 50, 1.0)  # Optimal around 2% of frame
            quality_result['quality_metrics']['size'] = size_score
            
            if face_area < 2500:  # Less than 50x50 pixels
                quality_result['quality_issues'].append('Face too small')
            
            # 4. Contrast assessment
            contrast = gray.std()
            contrast_score = min(contrast / 50, 1.0)
            quality_result['quality_metrics']['contrast'] = contrast_score
            
            if contrast < 20:
                quality_result['quality_issues'].append('Low contrast')
            
            # 5. Detection confidence
            det_confidence = face_data['det_score']
            quality_result['quality_metrics']['detection_confidence'] = det_confidence
            
            if det_confidence < 0.8:
                quality_result['quality_issues'].append('Low detection confidence')
            
            # 6. Pose estimation (if available)
            if 'pose' in face_data and face_data['pose'] is not None:
                pose = face_data['pose']
                # Simple pose quality (assuming pose is [yaw, pitch, roll])
                pose_quality = 1.0 - (abs(pose[0]) + abs(pose[1]) + abs(pose[2])) / 180
                quality_result['quality_metrics']['pose'] = max(pose_quality, 0.0)
                
                if abs(pose[0]) > 30 or abs(pose[1]) > 30:
                    quality_result['quality_issues'].append('Head pose too extreme')
            
            # Combined quality score with weights
            weights = {
                'sharpness': 0.25,
                'brightness': 0.15,
                'size': 0.15,
                'contrast': 0.15,
                'detection_confidence': 0.20,
                'pose': 0.10
            }
            
            total_score = 0.0
            total_weight = 0.0
            
            for metric, score in quality_result['quality_metrics'].items():
                if metric in weights:
                    total_score += score * weights[metric]
                    total_weight += weights[metric]
            
            if total_weight > 0:
                quality_result['face_quality_score'] = total_score / total_weight
            
        except Exception as e:
            quality_result['quality_issues'].append(f'Quality assessment error: {str(e)}')
            logger.error(f"Face quality assessment error: {e}")
        
        return quality_result
    
    def _evaluate_business_logic(self, employee_id: int, timestamp: str) -> Dict:
        """Enhanced business logic with detailed reasoning"""
        
        business_result = {
            'should_record_attendance': False,
            'attendance_decision_reason': '',
            'suggested_event_type': None,
            'business_checks': {
                'work_hours': False,
                'cooldown_period': False,
                'weekend_check': False,
                'last_event_type': None
            }
        }
        
        try:
            # Parse timestamp
            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            
            # 1. Work hours check
            work_start = datetime.strptime(self.config['work_hours_start'], '%H:%M').time()
            work_end = datetime.strptime(self.config['work_hours_end'], '%H:%M').time()
            current_time = dt.time()
            
            in_work_hours = work_start <= current_time <= work_end
            business_result['business_checks']['work_hours'] = in_work_hours
            
            if not in_work_hours:
                business_result['attendance_decision_reason'] = f'Outside work hours ({self.config["work_hours_start"]}-{self.config["work_hours_end"]})'
                return business_result
            
            # 2. Weekend check (optional)
            is_weekend = dt.weekday() >= 5  # Saturday = 5, Sunday = 6
            business_result['business_checks']['weekend_check'] = not is_weekend
            
            # You can enable/disable weekend attendance based on policy
            # For demo, we allow weekend attendance
            
            # 3. Get today's records for cooldown and event type determination
            today_records = self.database.get_today_records(employee_id)
            
            # 4. Cooldown period check
            cooldown_ok = True
            if today_records:
                last_record = today_records[-1]
                last_time = datetime.strptime(last_record['timestamp'], '%Y-%m-%d %H:%M:%S')
                time_diff = (dt - last_time).total_seconds() / 60
                
                cooldown_ok = time_diff >= self.config['cooldown_minutes']
                business_result['business_checks']['cooldown_period'] = cooldown_ok
                
                if not cooldown_ok:
                    remaining_minutes = self.config['cooldown_minutes'] - time_diff
                    business_result['attendance_decision_reason'] = f'Cooldown period: {remaining_minutes:.1f} minutes remaining'
                    return business_result
            
            # 5. Determine event type
            if not today_records:
                suggested_event = 'check_in'
            else:
                last_event = today_records[-1]['event_type']
                business_result['business_checks']['last_event_type'] = last_event
                suggested_event = 'check_out' if last_event == 'check_in' else 'check_in'
            
            business_result['suggested_event_type'] = suggested_event
            business_result['should_record_attendance'] = True
            business_result['attendance_decision_reason'] = f'Ready for {suggested_event}'
            
        except Exception as e:
            business_result['attendance_decision_reason'] = f'Business logic error: {str(e)}'
            logger.error(f"Business logic evaluation error: {e}")
        
        return business_result
    
    def _check_anti_spoofing(self, frame: np.ndarray, face_data: Dict) -> Dict:
        """Basic anti-spoofing checks (placeholder for advanced implementation)"""
        
        spoofing_result = {
            'anti_spoofing_score': 0.5,
            'spoofing_risk': 'medium',
            'spoofing_checks': {
                'liveness_detected': False,
                'texture_analysis': 0.5,
                'depth_estimation': 0.5
            }
        }
        
        # Placeholder implementation
        # In production, you would implement:
        # - Liveness detection
        # - Texture analysis
        # - Depth estimation
        # - Eye blink detection
        # - Head movement analysis
        
        logger.debug("🔒 Anti-spoofing check (placeholder implementation)")
        
        return spoofing_result
    
    def _record_attendance_event(self, face_result: Dict, timestamp: str, 
                               camera_id: str) -> Optional[Dict]:
        """Record attendance event in database"""
        
        if not face_result['should_record_attendance']:
            return None
        
        try:
            # Save snapshot if enabled
            snapshot_path = None
            if self.config['snapshot_enabled']:
                snapshot_filename = f"attendance_{face_result['employee_id']}_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
                snapshot_path = os.path.join(self.config['snapshot_directory'], snapshot_filename)
                # Note: In real implementation, you would save the actual face crop here
            
            # Record in database
            attendance_id = self.database.record_attendance(
                employee_id=face_result['employee_id'],
                event_type=face_result['suggested_event_type'],
                confidence=face_result['recognition_similarity'],
                timestamp=timestamp,
                face_quality=face_result['face_quality_score'],
                camera_id=camera_id,
                snapshot_path=snapshot_path,
                metadata={
                    'detection_confidence': face_result['detection_confidence'],
                    'embedding_norm': face_result['embedding_norm'],
                    'quality_metrics': face_result.get('quality_metrics', {}),
                    'business_checks': face_result.get('business_checks', {})
                }
            )
            
            if attendance_id:
                attendance_event = {
                    'attendance_id': attendance_id,
                    'employee_id': face_result['employee_id'],
                    'employee_name': face_result['employee_name'],
                    'employee_code': face_result['employee_code'],
                    'event_type': face_result['suggested_event_type'],
                    'timestamp': timestamp,
                    'confidence': face_result['recognition_similarity'],
                    'snapshot_path': snapshot_path
                }
                
                logger.info(f"✅ Attendance recorded: {face_result['employee_name']} - {face_result['suggested_event_type']}")
                self.performance_stats['attendance_recorded'] += 1
                
                return attendance_event
        
        except Exception as e:
            logger.error(f"❌ Failed to record attendance: {e}")
        
        return None
    
    def _generate_summary_message(self, result: Dict) -> str:
        """Generate human-readable summary message"""
        
        if result['status'] == 'error':
            return f"Error: {result.get('message', 'Unknown error')}"
        
        faces_count = result['faces_detected']
        recognized_count = len([r for r in result['recognitions'] if r['employee_found']])
        attendance_count = len(result['attendance_events'])
        
        if faces_count == 0:
            return "No faces detected"
        
        if attendance_count > 0:
            events = [f"{e['employee_name']} ({e['event_type']})" for e in result['attendance_events']]
            return f"Attendance recorded: {', '.join(events)}"
        
        if recognized_count > 0:
            return f"{recognized_count} face(s) recognized, no attendance recorded"
        
        return f"{faces_count} face(s) detected, none recognized"
    
    def _update_performance_stats(self, result: Dict):
        """Update performance statistics"""
        if not self.config['enable_performance_monitoring']:
            return
        
        try:
            self.performance_stats['frames_processed'] += 1
            self.performance_stats['faces_detected'] += result['faces_detected']
            self.performance_stats['faces_recognized'] += len([r for r in result['recognitions'] if r['employee_found']])
            
            processing_time = result['processing_info']['total_time_ms']
            self.performance_stats['total_processing_time'] += processing_time
            self.performance_stats['avg_processing_time_ms'] = (
                self.performance_stats['total_processing_time'] / 
                self.performance_stats['frames_processed']
            )
            
        except Exception as e:
            logger.warning(f"Performance stats update error: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        stats = self.performance_stats.copy()
        
        # Add derived metrics
        if stats['frames_processed'] > 0:
            stats['recognition_rate'] = stats['faces_recognized'] / max(stats['faces_detected'], 1)
            stats['attendance_rate'] = stats['attendance_recorded'] / max(stats['faces_recognized'], 1)
            stats['avg_faces_per_frame'] = stats['faces_detected'] / stats['frames_processed']
        
        # Add configuration info
        stats['config'] = self.config.copy()
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'faces_recognized': 0,
            'attendance_recorded': 0,
            'avg_processing_time_ms': 0.0,
            'total_processing_time': 0.0,
            'errors_count': 0
        }
        logger.info("📊 Performance statistics reset") 