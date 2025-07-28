"""
Database Module for Attendance System Pipeline V1

SQLite database with vector operations simulation and PostgreSQL-compatible design
for rapid demo deployment with migration path to production.
"""

import sqlite3
import json
import numpy as np
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
import os
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttendanceDatabaseSQLite:
    """SQLite database with vector operations simulation (PostgreSQL-compatible design)"""
    
    def __init__(self, db_path: str = "attendance_system.db", enable_wal: bool = True):
        """
        Initialize SQLite database with PostgreSQL-compatible schema
        
        Args:
            db_path: Path to SQLite database file
            enable_wal: Enable Write-Ahead Logging for better concurrency
        """
        self.db_path = db_path
        self.enable_wal = enable_wal
        
        logger.info(f"📊 Initializing Attendance Database: {db_path}")
        
        # Create database directory if needed
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        
        # Initialize connection
        self._init_connection()
        
        # Setup schema
        self._create_tables()
        self._create_indexes()
        self._setup_performance_optimization()
        self._initialize_system_config()
        
        logger.info("✅ Database initialization completed")
    
    def _init_connection(self):
        """Initialize database connection with optimizations"""
        self.conn = sqlite3.connect(
            self.db_path, 
            check_same_thread=False,
            timeout=30.0
        )
        
        # Enable row factory for dict-like access
        self.conn.row_factory = sqlite3.Row
        
        # Enable WAL mode for better concurrency
        if self.enable_wal:
            self.conn.execute("PRAGMA journal_mode=WAL")
        
        # Performance optimizations
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=10000")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA mmap_size=268435456")  # 256MB
        
        logger.info("🔧 Database connection optimized")
    
    def _create_tables(self):
        """Create PostgreSQL-compatible schema in SQLite"""
        cursor = self.conn.cursor()
        
        # Employees table with JSON embedding storage
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            department TEXT,
            position TEXT,
            face_embeddings TEXT,  -- JSON array of 512-dim vectors
            embedding_count INTEGER DEFAULT 0,
            registration_quality REAL DEFAULT 0.0,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Attendance logs with enhanced metadata
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER,
            event_type TEXT NOT NULL CHECK (event_type IN ('check_in', 'check_out')),
            timestamp TIMESTAMP NOT NULL,
            confidence REAL NOT NULL,
            face_quality REAL,
            processing_time_ms REAL,
            camera_id TEXT,
            snapshot_path TEXT,
            metadata TEXT,  -- JSON for additional data
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (employee_id) REFERENCES employees (id)
        )
        """)
        
        # Individual face registrations for quality tracking
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_registrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER,
            image_path TEXT NOT NULL,
            embedding TEXT NOT NULL,  -- JSON 512-dim vector
            quality_score REAL,
            image_metadata TEXT,  -- JSON for image info
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (employee_id) REFERENCES employees (id)
        )
        """)
        
        # System configuration for business rules
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_key TEXT UNIQUE NOT NULL,
            config_value TEXT NOT NULL,
            description TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Performance metrics tracking
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_type TEXT NOT NULL,
            metric_value REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT  -- JSON for additional context
        )
        """)
        
        self.conn.commit()
        logger.info("🗄️ Database tables created")
    
    def _create_indexes(self):
        """Create indexes for performance"""
        cursor = self.conn.cursor()
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_attendance_employee_date ON attendance_logs(employee_id, DATE(timestamp))",
            "CREATE INDEX IF NOT EXISTS idx_attendance_timestamp ON attendance_logs(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_employees_active ON employees(is_active)",
            "CREATE INDEX IF NOT EXISTS idx_face_registrations_employee ON face_registrations(employee_id)",
            "CREATE INDEX IF NOT EXISTS idx_system_config_key ON system_config(config_key)",
            "CREATE INDEX IF NOT EXISTS idx_performance_metrics_type ON performance_metrics(metric_type, timestamp)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        self.conn.commit()
        logger.info("📇 Database indexes created")
    
    def _setup_performance_optimization(self):
        """Setup additional performance optimizations"""
        cursor = self.conn.cursor()
        
        # Analyze tables for query optimization
        cursor.execute("ANALYZE")
        
        # Enable automatic index recommendations
        cursor.execute("PRAGMA optimize")
        
        self.conn.commit()
        logger.info("⚡ Performance optimization applied")
    
    def _initialize_system_config(self):
        """Initialize default system configuration"""
        default_configs = [
            ("recognition_threshold", "0.65", "Minimum similarity threshold for face recognition"),
            ("cooldown_minutes", "30", "Minimum minutes between attendance records"),
            ("work_hours_start", "07:00", "Work day start time"),
            ("work_hours_end", "19:00", "Work day end time"),
            ("max_face_registrations", "10", "Maximum face images per employee"),
            ("face_quality_threshold", "0.3", "Minimum face quality score"),
            ("backup_retention_days", "30", "Days to retain backup files"),
            ("performance_metrics_retention_days", "7", "Days to retain performance metrics")
        ]
        
        cursor = self.conn.cursor()
        for key, value, description in default_configs:
            cursor.execute("""
            INSERT OR IGNORE INTO system_config (config_key, config_value, description)
            VALUES (?, ?, ?)
            """, (key, value, description))
        
        self.conn.commit()
        logger.info("⚙️ System configuration initialized")
    
    def register_employee(self, employee_data: Dict, face_embeddings: List[np.ndarray]) -> int:
        """
        Employee registration with multiple face embeddings
        
        Args:
            employee_data: Employee information dict
            face_embeddings: List of 512-dim face embeddings
            
        Returns:
            Employee ID if successful, None otherwise
        """
        cursor = self.conn.cursor()
        
        try:
            logger.info(f"📝 Registering employee: {employee_data['name']}")
            
            # Begin transaction
            cursor.execute("BEGIN TRANSACTION")
            
            # Insert employee record
            cursor.execute("""
            INSERT INTO employees (employee_code, name, email, department, position, embedding_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                employee_data['employee_code'],
                employee_data['name'],
                employee_data['email'],
                employee_data.get('department', ''),
                employee_data.get('position', ''),
                len(face_embeddings)
            ))
            
            employee_id = cursor.lastrowid
            
            # Store individual face registrations
            registration_qualities = []
            
            for i, embedding in enumerate(face_embeddings):
                # Calculate quality score (mock implementation)
                quality_score = self._calculate_embedding_quality(embedding)
                registration_qualities.append(quality_score)
                
                cursor.execute("""
                INSERT INTO face_registrations (employee_id, image_path, embedding, quality_score)
                VALUES (?, ?, ?, ?)
                """, (
                    employee_id,
                    f"registration_{employee_id}_{i}.jpg",
                    json.dumps(embedding.tolist()),
                    quality_score
                ))
            
            # Calculate average embedding and overall quality
            if face_embeddings:
                avg_embedding = np.mean(face_embeddings, axis=0)
                avg_quality = np.mean(registration_qualities)
                
                # Update employee with averaged embedding
                cursor.execute("""
                UPDATE employees
                SET face_embeddings = ?, registration_quality = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """, (json.dumps(avg_embedding.tolist()), avg_quality, employee_id))
            
            # Commit transaction
            cursor.execute("COMMIT")
            
            logger.info(f"✅ Employee registered successfully: ID {employee_id}")
            
            # Record performance metric
            self._record_performance_metric("employee_registration", 1.0, {"employee_id": employee_id})
            
            return employee_id
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            logger.error(f"❌ Employee registration failed: {e}")
            return None
    
    def find_employee_by_embedding(self, embedding: np.ndarray, threshold: float = 0.65) -> Optional[Dict]:
        """
        Vector similarity search using cosine distance
        
        Args:
            embedding: Query embedding vector
            threshold: Similarity threshold
            
        Returns:
            Best matching employee or None
        """
        start_time = time.time()
        
        try:
            cursor = self.conn.cursor()
            
            # Get all active employees with embeddings
            cursor.execute("""
            SELECT id, employee_code, name, email, department, face_embeddings, registration_quality
            FROM employees
            WHERE is_active = 1 AND face_embeddings IS NOT NULL
            """)
            
            employees = cursor.fetchall()
            
            if not employees:
                return None
            
            best_match = None
            best_similarity = 0.0
            
            # Calculate similarities
            for emp in employees:
                try:
                    stored_embedding = np.array(json.loads(emp['face_embeddings']))
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity(
                        embedding.reshape(1, -1),
                        stored_embedding.reshape(1, -1)
                    )[0][0]
                    
                    if similarity > best_similarity and similarity > threshold:
                        best_similarity = similarity
                        best_match = {
                            'id': emp['id'],
                            'employee_code': emp['employee_code'],
                            'name': emp['name'],
                            'email': emp['email'],
                            'department': emp['department'],
                            'similarity': similarity,
                            'registration_quality': emp['registration_quality']
                        }
                
                except Exception as e:
                    logger.warning(f"Error processing employee {emp['id']}: {e}")
                    continue
            
            # Record performance metric
            search_time = (time.time() - start_time) * 1000
            self._record_performance_metric("embedding_search_ms", search_time, {
                "employees_searched": len(employees),
                "match_found": best_match is not None
            })
            
            return best_match
            
        except Exception as e:
            logger.error(f"❌ Employee search error: {e}")
            return None
    
    def record_attendance(self, employee_id: int, event_type: str, confidence: float, 
                         timestamp: str = None, **kwargs) -> int:
        """
        Attendance logging with business logic
        
        Args:
            employee_id: Employee ID
            event_type: 'check_in' or 'check_out'
            confidence: Recognition confidence score
            timestamp: Optional timestamp
            **kwargs: Additional metadata
            
        Returns:
            Attendance log ID if successful, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            cursor = self.conn.cursor()
            
            # Prepare metadata
            metadata = {
                'face_quality': kwargs.get('face_quality'),
                'processing_time_ms': kwargs.get('processing_time_ms'),
                'camera_info': kwargs.get('camera_info'),
                'additional_data': kwargs.get('metadata', {})
            }
            
            # Insert attendance record
            cursor.execute("""
            INSERT INTO attendance_logs 
            (employee_id, event_type, timestamp, confidence, face_quality, 
             processing_time_ms, camera_id, snapshot_path, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                employee_id,
                event_type,
                timestamp,
                confidence,
                kwargs.get('face_quality'),
                kwargs.get('processing_time_ms'),
                kwargs.get('camera_id'),
                kwargs.get('snapshot_path'),
                json.dumps(metadata)
            ))
            
            attendance_id = cursor.lastrowid
            self.conn.commit()
            
            logger.info(f"📋 Attendance recorded: Employee {employee_id}, {event_type}, ID {attendance_id}")
            
            # Record performance metric
            self._record_performance_metric("attendance_recorded", 1.0, {
                "employee_id": employee_id,
                "event_type": event_type
            })
            
            return attendance_id
            
        except Exception as e:
            logger.error(f"❌ Attendance recording error: {e}")
            return None
    
    def get_attendance_reports(self, filters: Dict) -> List[Dict]:
        """
        Analytics queries for reporting
        
        Args:
            filters: Report filters (date_from, date_to, employee_id, etc.)
            
        Returns:
            List of attendance records
        """
        cursor = self.conn.cursor()
        
        # Build dynamic query
        base_query = """
        SELECT al.*, e.name, e.employee_code, e.department
        FROM attendance_logs al
        JOIN employees e ON al.employee_id = e.id
        WHERE 1=1
        """
        
        params = []
        
        # Apply filters
        if filters.get('date_from'):
            base_query += " AND DATE(al.timestamp) >= ?"
            params.append(filters['date_from'])
        
        if filters.get('date_to'):
            base_query += " AND DATE(al.timestamp) <= ?"
            params.append(filters['date_to'])
        
        if filters.get('employee_id'):
            base_query += " AND al.employee_id = ?"
            params.append(filters['employee_id'])
        
        if filters.get('event_type'):
            base_query += " AND al.event_type = ?"
            params.append(filters['event_type'])
        
        if filters.get('department'):
            base_query += " AND e.department = ?"
            params.append(filters['department'])
        
        base_query += " ORDER BY al.timestamp DESC"
        
        # Apply limit
        if filters.get('limit'):
            base_query += " LIMIT ?"
            params.append(filters['limit'])
        
        cursor.execute(base_query, params)
        
        results = []
        for row in cursor.fetchall():
            record = dict(row)
            # Parse metadata if present
            if record['metadata']:
                try:
                    record['metadata'] = json.loads(record['metadata'])
                except:
                    record['metadata'] = {}
            results.append(record)
        
        return results
    
    def get_today_records(self, employee_id: int) -> List[Dict]:
        """Get today's attendance records for an employee"""
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT * FROM attendance_logs
        WHERE employee_id = ? AND DATE(timestamp) = DATE('now')
        ORDER BY timestamp ASC
        """, (employee_id,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_employee_statistics(self) -> Dict:
        """Get comprehensive employee statistics"""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Basic counts
        cursor.execute("SELECT COUNT(*) FROM employees WHERE is_active = 1")
        stats['total_active_employees'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM employees WHERE is_active = 1 AND face_embeddings IS NOT NULL")
        stats['employees_with_faces'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM attendance_logs")
        stats['total_attendance_logs'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM face_registrations")
        stats['total_face_registrations'] = cursor.fetchone()[0]
        
        # Today's statistics
        cursor.execute("SELECT COUNT(*) FROM attendance_logs WHERE DATE(timestamp) = DATE('now')")
        stats['todays_attendance_count'] = cursor.fetchone()[0]
        
        # Average quality scores
        cursor.execute("SELECT AVG(registration_quality) FROM employees WHERE registration_quality > 0")
        avg_quality = cursor.fetchone()[0]
        stats['avg_registration_quality'] = avg_quality if avg_quality else 0.0
        
        # Recent activity
        cursor.execute("""
        SELECT COUNT(*) FROM attendance_logs 
        WHERE timestamp >= datetime('now', '-24 hours')
        """)
        stats['activity_last_24h'] = cursor.fetchone()[0]
        
        return stats
    
    def _calculate_embedding_quality(self, embedding: np.ndarray) -> float:
        """Calculate quality score for face embedding"""
        try:
            # Simple quality metrics
            norm = np.linalg.norm(embedding)
            variance = np.var(embedding)
            
            # Normalize metrics
            norm_score = min(norm / 1.0, 1.0)  # Assuming good embeddings have norm around 1
            variance_score = min(variance * 10, 1.0)  # Higher variance usually better
            
            quality = (norm_score + variance_score) / 2
            return min(max(quality, 0.0), 1.0)
            
        except:
            return 0.5  # Default quality
    
    def _record_performance_metric(self, metric_type: str, metric_value: float, metadata: Dict = None):
        """Record performance metric"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
            INSERT INTO performance_metrics (metric_type, metric_value, metadata)
            VALUES (?, ?, ?)
            """, (metric_type, metric_value, json.dumps(metadata) if metadata else None))
            self.conn.commit()
        except Exception as e:
            logger.warning(f"Failed to record performance metric: {e}")
    
    def backup_database(self, backup_path: str = None) -> str:
        """Create database backup"""
        if backup_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{self.db_path}.backup_{timestamp}"
        
        try:
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"✅ Database backed up to: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            raise
    
    def optimize_database(self):
        """Perform database optimization"""
        cursor = self.conn.cursor()
        
        logger.info("🔧 Optimizing database...")
        
        # Vacuum to reclaim space
        cursor.execute("VACUUM")
        
        # Update statistics
        cursor.execute("ANALYZE")
        
        # Optimize queries
        cursor.execute("PRAGMA optimize")
        
        logger.info("✅ Database optimization completed")
    
    def migrate_to_postgresql(self, postgres_connection_string: str):
        """
        Future migration helper to PostgreSQL + pgvector
        
        Args:
            postgres_connection_string: PostgreSQL connection string
        """
        logger.info("🚀 PostgreSQL migration not yet implemented")
        logger.info("This method will export schema and data for PostgreSQL + pgvector")
        logger.info("JSON embeddings will be converted to pgvector format")
        
        # TODO: Implement actual migration
        # 1. Export schema with pgvector types
        # 2. Convert JSON embeddings to vector format
        # 3. Migrate data with proper foreign keys
        # 4. Create pgvector indexes
        # 5. Validate data integrity
    
    def close(self):
        """Close database connection"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
            logger.info("✅ Database connection closed")
        except Exception as e:
            logger.warning(f"⚠️ Database close warning: {e}") 