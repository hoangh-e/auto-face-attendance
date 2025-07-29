#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for AI Attendance System
"""

import sys
import os
import traceback
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    try:
        import numpy as np
        print("✅ numpy")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas")
    except ImportError as e:
        print(f"❌ pandas: {e}")
        return False
    
    try:
        import cv2
        print("✅ opencv-python")
    except ImportError as e:
        print(f"❌ opencv-python: {e}")
        return False
    
    try:
        import sqlite3
        print("✅ sqlite3")
    except ImportError as e:
        print(f"❌ sqlite3: {e}")
        return False
    
    try:
        import insightface
        print("✅ insightface")
    except ImportError as e:
        print(f"❌ insightface: {e}")
        return False
    
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        print("✅ scikit-learn")
    except ImportError as e:
        print(f"❌ scikit-learn: {e}")
        return False
    
    return True

def test_directories():
    """Test directory creation"""
    print("\n🗂️ Testing directories...")
    
    base_dir = Path.cwd()
    directories = [
        base_dir / 'employees',
        base_dir / 'data',
        base_dir / 'snapshots'
    ]
    
    for directory in directories:
        try:
            directory.mkdir(exist_ok=True)
            print(f"✅ {directory}")
        except Exception as e:
            print(f"❌ {directory}: {e}")
            return False
    
    return True

def test_camera():
    """Test camera detection"""
    print("\n📷 Testing camera...")
    
    try:
        import cv2
        
        # Test camera index 0
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print("✅ Camera 0 detected and working")
                cap.release()
                return True
            else:
                print("❌ Camera 0 detected but cannot capture frames")
                cap.release()
                return False
        else:
            print("❌ Camera 0 not available")
            return False
    
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_ai_system():
    """Test AI system initialization"""
    print("\n🤖 Testing AI system...")
    
    try:
        from insightface.app import FaceAnalysis
        
        print("📦 Loading AI model...")
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.5)
        print("✅ AI model loaded successfully")
        
        # Test with a simple image
        import numpy as np
        test_image = np.zeros((320, 320, 3), dtype=np.uint8)
        faces = app.get(test_image)
        print(f"✅ AI inference test completed (found {len(faces)} faces)")
        
        return True
    
    except Exception as e:
        print(f"❌ AI system test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 AI Attendance System - Test Suite")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("Directories", test_directories),
        ("Camera", test_camera),
        ("AI System", test_ai_system),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n📊 Test Results:")
    print("=" * 20)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Add employee photos in the 'employees' folder")
        print("2. Run the main demo script: python local_camera_demo_fixed.py")
    else:
        print("❌ Some tests failed. Please fix the issues before proceeding.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install opencv-python insightface scikit-learn")
        print("- Check camera permissions")
        print("- Ensure you have a working webcam connected")

if __name__ == "__main__":
    main() 