#!/usr/bin/env python3
"""
AI Attendance System - Setup Validation Script
Tests all components before running the main demo
"""

import sys
import subprocess
import platform
import pathlib
import time
from datetime import datetime

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*50}")
    print(f"🧪 {title}")
    print(f"{'='*50}")

def print_test(name, passed, details=""):
    """Print test result"""
    status = "✅" if passed else "❌"
    print(f"{status} {name}")
    if details:
        print(f"   {details}")

def test_python_version():
    """Test Python version compatibility"""
    version = sys.version_info
    required_major, required_minor = 3, 8
    
    passed = version.major >= required_major and version.minor >= required_minor
    details = f"Current: {version.major}.{version.minor}.{version.micro}, Required: {required_major}.{required_minor}+"
    
    print_test("Python Version", passed, details)
    return passed

def test_core_packages():
    """Test core package imports"""
    packages = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'matplotlib': 'matplotlib',
        'PIL': 'pillow',
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm',
        'psutil': 'psutil'
    }
    
    all_passed = True
    for import_name, package_name in packages.items():
        try:
            __import__(import_name)
            print_test(f"Package: {package_name}", True)
        except ImportError:
            print_test(f"Package: {package_name}", False, "Not installed")
            all_passed = False
    
    return all_passed

def test_opencv():
    """Test OpenCV installation and camera access"""
    try:
        import cv2
        version = cv2.__version__
        print_test("OpenCV Import", True, f"Version: {version}")
        
        # Test camera access
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                print_test("Camera Access", True, f"Resolution: {width}x{height}")
                cap.release()
                return True
            else:
                print_test("Camera Access", False, "Cannot read frames")
                cap.release()
                return False
        else:
            print_test("Camera Access", False, "Cannot open camera")
            return False
            
    except ImportError:
        print_test("OpenCV Import", False, "Not installed")
        return False
    except Exception as e:
        print_test("OpenCV Test", False, f"Error: {e}")
        return False

def test_pytorch():
    """Test PyTorch installation and CUDA"""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        
        print_test("PyTorch Import", True, f"Version: {version}")
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            print_test("CUDA Support", True, f"GPU: {gpu_name}")
        else:
            print_test("CUDA Support", False, "Using CPU mode")
        
        return True
        
    except ImportError:
        print_test("PyTorch Import", False, "Not installed")
        return False

def test_insightface():
    """Test InsightFace installation and model loading"""
    try:
        import insightface
        from insightface.app import FaceAnalysis
        
        version = getattr(insightface, '__version__', 'Unknown')
        print_test("InsightFace Import", True, f"Version: {version}")
        
        # Test model loading (lightweight test)
        try:
            app = FaceAnalysis(name='buffalo_s')
            print_test("Model Loading Test", True, "buffalo_s model available")
            return True
        except Exception as e:
            print_test("Model Loading Test", False, f"Error: {str(e)[:50]}...")
            return False
            
    except ImportError:
        print_test("InsightFace Import", False, "Not installed")
        return False

def test_directory_structure():
    """Test required directory structure"""
    base_dir = pathlib.Path.cwd()
    required_dirs = [
        base_dir / 'employees',
        base_dir / 'data', 
        base_dir / 'snapshots',
        base_dir / 'logs'
    ]
    
    all_passed = True
    for directory in required_dirs:
        if directory.exists():
            print_test(f"Directory: {directory.name}", True, str(directory))
        else:
            print_test(f"Directory: {directory.name}", False, "Will be created")
            # Create missing directories
            directory.mkdir(exist_ok=True)
            all_passed = False
    
    return all_passed

def test_employee_data():
    """Test employee data availability"""
    employees_dir = pathlib.Path.cwd() / 'employees'
    
    if not employees_dir.exists():
        print_test("Employee Data", False, "No employees directory")
        return False
    
    employee_folders = [d for d in employees_dir.iterdir() if d.is_dir()]
    
    if not employee_folders:
        print_test("Employee Data", False, "No employee folders found")
        return False
    
    valid_employees = 0
    for folder in employee_folders:
        image_files = list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg')) + list(folder.glob('*.png'))
        if len(image_files) >= 2:
            valid_employees += 1
    
    if valid_employees > 0:
        print_test("Employee Data", True, f"{valid_employees} employees with photos")
        return True
    else:
        print_test("Employee Data", False, "No employees with sufficient photos")
        return False

def test_memory_resources():
    """Test system memory and resources"""
    try:
        import psutil
        
        # Memory test
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        if memory_gb >= 4:
            print_test("System Memory", True, f"{memory_gb:.1f} GB available")
            memory_passed = True
        else:
            print_test("System Memory", False, f"{memory_gb:.1f} GB (need 4GB+)")
            memory_passed = False
        
        # Disk space test
        disk = psutil.disk_usage('.')
        disk_gb = disk.free / (1024**3)
        
        if disk_gb >= 2:
            print_test("Disk Space", True, f"{disk_gb:.1f} GB free")
            disk_passed = True
        else:
            print_test("Disk Space", False, f"{disk_gb:.1f} GB (need 2GB+)")
            disk_passed = False
        
        return memory_passed and disk_passed
        
    except ImportError:
        print_test("Resource Test", False, "psutil not available")
        return False

def run_comprehensive_test():
    """Run all validation tests"""
    print_header("SYSTEM VALIDATION STARTING")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Working Directory: {pathlib.Path.cwd()}")
    
    # Test categories
    tests = [
        ("Python Version", test_python_version),
        ("Core Packages", test_core_packages),
        ("OpenCV & Camera", test_opencv),
        ("PyTorch", test_pytorch),
        ("InsightFace", test_insightface),
        ("Directory Structure", test_directory_structure),
        ("Employee Data", test_employee_data),
        ("System Resources", test_memory_resources)
    ]
    
    results = {}
    passed_count = 0
    
    for test_name, test_func in tests:
        print_header(f"Testing {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed_count += 1
        except Exception as e:
            print_test(f"{test_name} Test", False, f"Exception: {e}")
            results[test_name] = False
    
    # Summary
    print_header("VALIDATION SUMMARY")
    total_tests = len(tests)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 RESULT: {passed_count}/{total_tests} tests passed")
    
    if passed_count == total_tests:
        print("🎉 ALL TESTS PASSED - System ready for demo!")
        print("\nNext steps:")
        print("1. Run: python improved_local_camera_demo.py")
        print("2. Follow on-screen instructions")
        return True
    else:
        print("❌ SOME TESTS FAILED - Fix issues before running demo")
        print("\nRecommended actions:")
        
        if not results.get("Core Packages", False):
            print("• Install missing packages: pip install -r requirements.txt")
        
        if not results.get("OpenCV & Camera", False):
            print("• Install OpenCV: pip install opencv-python")
            print("• Check camera permissions and connections")
        
        if not results.get("PyTorch", False):
            print("• Install PyTorch: pip install torch torchvision")
        
        if not results.get("InsightFace", False):
            print("• Install InsightFace: pip install insightface")
        
        if not results.get("Employee Data", False):
            print("• Add employee photos to employees/ directory")
            print("• Each employee needs 2+ photos in their folder")
        
        return False

def quick_test():
    """Quick validation for basic functionality"""
    print("🧪 QUICK SYSTEM TEST")
    print("=" * 30)
    
    try:
        # Basic imports
        import cv2, numpy, torch, insightface
        print("✅ Core imports: OK")
        
        # Camera test
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera: {frame.shape}")
            cap.release()
        else:
            print("⚠️ Camera: Not detected")
        
        # GPU test
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 GPU: Using CPU mode")
        
        print("\n🎉 Quick test completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        run_comprehensive_test() 