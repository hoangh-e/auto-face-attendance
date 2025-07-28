"""
Integration modules for AI Attendance System

Contains camera management, Frigate NVR simulation, MQTT messaging,
and notification system integrations.
"""

from .camera import CameraManager
from .notifications import NotificationManager

__all__ = [
    'CameraManager',
    'NotificationManager'
] 