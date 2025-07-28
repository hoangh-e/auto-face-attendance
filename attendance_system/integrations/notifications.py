"""
Notification Management Module for Attendance System Pipeline V1

Handles various notification types including Slack, Teams, email,
and local notifications for attendance events.
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NotificationManager:
    """Unified notification management for attendance events"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize notification manager
        
        Args:
            config: Notification configuration
        """
        self.config = {
            'enabled': True,
            'channels': {
                'console': True,
                'slack': False,
                'teams': False,
                'email': False,
                'webhook': False
            },
            'slack': {
                'webhook_url': '',
                'channel': '#attendance',
                'username': 'AttendanceBot',
                'emoji': ':robot_face:'
            },
            'teams': {
                'webhook_url': '',
            },
            'email': {
                'smtp_server': '',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'from_email': '',
                'to_emails': []
            },
            'webhook': {
                'url': '',
                'headers': {},
                'timeout': 30
            },
            'templates': {
                'check_in': '✅ {name} checked in at {time}',
                'check_out': '🚪 {name} checked out at {time}',
                'unknown_face': '❓ Unknown person detected at {time}',
                'system_alert': '⚠️ System Alert: {message}'
            }
        }
        
        if config:
            self._update_config(config)
        
        self.notification_history = []
        
        logger.info("📬 Notification Manager initialized")
        self._log_enabled_channels()
    
    def _update_config(self, config: Dict):
        """Update configuration recursively"""
        def update_dict(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    update_dict(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        update_dict(self.config, config)
    
    def _log_enabled_channels(self):
        """Log enabled notification channels"""
        enabled_channels = [channel for channel, enabled in self.config['channels'].items() if enabled]
        if enabled_channels:
            logger.info(f"📢 Enabled notification channels: {', '.join(enabled_channels)}")
        else:
            logger.warning("⚠️ No notification channels enabled")
    
    def send_attendance_notification(self, event_type: str, employee_data: Dict, 
                                   timestamp: str = None, metadata: Dict = None) -> bool:
        """
        Send attendance notification
        
        Args:
            event_type: 'check_in' or 'check_out'
            employee_data: Employee information
            timestamp: Event timestamp
            metadata: Additional metadata
            
        Returns:
            True if at least one notification was sent successfully
        """
        if not self.config['enabled']:
            return False
        
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Prepare notification data
        notification_data = {
            'type': 'attendance',
            'event_type': event_type,
            'employee': employee_data,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        # Generate message
        template = self.config['templates'].get(event_type, '{name} - {event_type} at {time}')
        message = template.format(
            name=employee_data.get('name', 'Unknown'),
            time=timestamp,
            event_type=event_type
        )
        
        return self._send_notification(message, notification_data)
    
    def send_system_alert(self, message: str, severity: str = 'info', 
                         metadata: Dict = None) -> bool:
        """
        Send system alert notification
        
        Args:
            message: Alert message
            severity: Alert severity ('info', 'warning', 'error')
            metadata: Additional metadata
            
        Returns:
            True if at least one notification was sent successfully
        """
        if not self.config['enabled']:
            return False
        
        # Prepare notification data
        notification_data = {
            'type': 'system_alert',
            'severity': severity,
            'message': message,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metadata': metadata or {}
        }
        
        # Add severity emoji
        severity_emoji = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨'
        }
        
        formatted_message = f"{severity_emoji.get(severity, 'ℹ️')} {message}"
        
        return self._send_notification(formatted_message, notification_data)
    
    def send_unknown_face_alert(self, timestamp: str = None, 
                              metadata: Dict = None) -> bool:
        """
        Send unknown face detection alert
        
        Args:
            timestamp: Detection timestamp
            metadata: Additional metadata
            
        Returns:
            True if at least one notification was sent successfully
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        notification_data = {
            'type': 'unknown_face',
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        template = self.config['templates']['unknown_face']
        message = template.format(time=timestamp)
        
        return self._send_notification(message, notification_data)
    
    def _send_notification(self, message: str, data: Dict) -> bool:
        """Send notification through enabled channels"""
        success_count = 0
        
        # Console notification
        if self.config['channels']['console']:
            if self._send_console_notification(message, data):
                success_count += 1
        
        # Slack notification
        if self.config['channels']['slack']:
            if self._send_slack_notification(message, data):
                success_count += 1
        
        # Teams notification
        if self.config['channels']['teams']:
            if self._send_teams_notification(message, data):
                success_count += 1
        
        # Email notification
        if self.config['channels']['email']:
            if self._send_email_notification(message, data):
                success_count += 1
        
        # Webhook notification
        if self.config['channels']['webhook']:
            if self._send_webhook_notification(message, data):
                success_count += 1
        
        # Record in history
        self.notification_history.append({
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'data': data,
            'success_channels': success_count
        })
        
        return success_count > 0
    
    def _send_console_notification(self, message: str, data: Dict) -> bool:
        """Send console notification"""
        try:
            print(f"📬 NOTIFICATION: {message}")
            logger.info(f"Notification sent: {message}")
            return True
        except Exception as e:
            logger.error(f"Console notification error: {e}")
            return False
    
    def _send_slack_notification(self, message: str, data: Dict) -> bool:
        """Send Slack notification"""
        try:
            # This is a placeholder implementation
            # In production, you would use the Slack SDK or webhook
            slack_config = self.config['slack']
            
            if not slack_config.get('webhook_url'):
                logger.warning("Slack webhook URL not configured")
                return False
            
            payload = {
                'text': message,
                'channel': slack_config.get('channel', '#attendance'),
                'username': slack_config.get('username', 'AttendanceBot'),
                'icon_emoji': slack_config.get('emoji', ':robot_face:')
            }
            
            # TODO: Implement actual Slack API call
            logger.info(f"📱 Slack notification (simulated): {message}")
            return True
            
        except Exception as e:
            logger.error(f"Slack notification error: {e}")
            return False
    
    def _send_teams_notification(self, message: str, data: Dict) -> bool:
        """Send Microsoft Teams notification"""
        try:
            teams_config = self.config['teams']
            
            if not teams_config.get('webhook_url'):
                logger.warning("Teams webhook URL not configured")
                return False
            
            # TODO: Implement actual Teams webhook call
            logger.info(f"💼 Teams notification (simulated): {message}")
            return True
            
        except Exception as e:
            logger.error(f"Teams notification error: {e}")
            return False
    
    def _send_email_notification(self, message: str, data: Dict) -> bool:
        """Send email notification"""
        try:
            email_config = self.config['email']
            
            if not email_config.get('smtp_server') or not email_config.get('to_emails'):
                logger.warning("Email configuration incomplete")
                return False
            
            # TODO: Implement actual email sending
            logger.info(f"📧 Email notification (simulated): {message}")
            return True
            
        except Exception as e:
            logger.error(f"Email notification error: {e}")
            return False
    
    def _send_webhook_notification(self, message: str, data: Dict) -> bool:
        """Send webhook notification"""
        try:
            webhook_config = self.config['webhook']
            
            if not webhook_config.get('url'):
                logger.warning("Webhook URL not configured")
                return False
            
            payload = {
                'message': message,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            
            # TODO: Implement actual webhook call
            logger.info(f"🔗 Webhook notification (simulated): {message}")
            return True
            
        except Exception as e:
            logger.error(f"Webhook notification error: {e}")
            return False
    
    def get_notification_history(self, limit: int = 50) -> List[Dict]:
        """Get recent notification history"""
        return self.notification_history[-limit:]
    
    def clear_notification_history(self):
        """Clear notification history"""
        self.notification_history.clear()
        logger.info("📭 Notification history cleared")
    
    def test_notifications(self) -> Dict[str, bool]:
        """Test all enabled notification channels"""
        test_message = "🧪 Test notification from AI Attendance System"
        test_data = {
            'type': 'test',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        results = {}
        
        if self.config['channels']['console']:
            results['console'] = self._send_console_notification(test_message, test_data)
        
        if self.config['channels']['slack']:
            results['slack'] = self._send_slack_notification(test_message, test_data)
        
        if self.config['channels']['teams']:
            results['teams'] = self._send_teams_notification(test_message, test_data)
        
        if self.config['channels']['email']:
            results['email'] = self._send_email_notification(test_message, test_data)
        
        if self.config['channels']['webhook']:
            results['webhook'] = self._send_webhook_notification(test_message, test_data)
        
        logger.info(f"📋 Notification test results: {results}")
        return results 