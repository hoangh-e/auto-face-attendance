"""
Setup script for AI Attendance System Pipeline V1
Repository: hoangh-e/auto-face-attendance/tree/pipeline-v1.0/
"""

from setuptools import setup, find_packages
import os
import sys

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "AI Attendance System with SCRFD detection and ArcFace recognition"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Handle conditional requirements
                    if ';' in line:
                        req, condition = line.split(';', 1)
                        # Evaluate simple conditions
                        if 'sys_platform' in condition:
                            if 'darwin' in condition and sys.platform == 'darwin':
                                requirements.append(req.strip())
                            elif 'darwin' not in condition and sys.platform != 'darwin':
                                requirements.append(req.strip())
                        else:
                            requirements.append(req.strip())
                    else:
                        requirements.append(line)
    
    return requirements

# Version information
VERSION = "1.0.0"

setup(
    name="ai-attendance-system",
    version=VERSION,
    author="AI Attendance System Team",
    author_email="contact@attendance-system.com",
    description="AI-powered attendance system with SCRFD face detection and ArcFace recognition",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/hoangh-e/auto-face-attendance",
    project_urls={
        "Repository": "https://github.com/hoangh-e/auto-face-attendance",
        "Pipeline V1": "https://github.com/hoangh-e/auto-face-attendance/tree/pipeline-v1.0",
        "Documentation": "https://github.com/hoangh-e/auto-face-attendance/blob/main/README.md",
        "Issues": "https://github.com/hoangh-e/auto-face-attendance/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Office/Business :: Human Resources",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "attendance", "face-recognition", "ai", "computer-vision", 
        "scrfd", "arcface", "insightface", "sqlite", "real-time",
        "biometrics", "opencv", "pytorch", "pipeline"
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "insightface>=0.7.3",
        "opencv-python>=4.7.0",
        "scikit-learn>=1.1.0",
        "numpy>=1.21.0",
        "pandas>=1.4.0",
        "pillow>=9.0.0",
        "tqdm>=4.64.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.10.0",
        "requests>=2.28.0",
        "psutil>=5.9.0",
        "pyyaml>=6.0",
        "python-dotenv>=0.20.0",
        "python-dateutil>=2.8.0",
    ],
    extras_require={
        "gpu": [
            "onnxruntime-gpu>=1.15.0",
        ],
        "postgresql": [
            "psycopg2-binary>=2.9.0",
            "pgvector>=0.2.0",
        ],
        "web": [
            "fastapi>=0.85.0",
            "uvicorn>=0.18.0",
            "streamlit>=1.12.0",
        ],
        "notifications": [
            "slack-sdk>=3.18.0",
            "pymsteams>=0.2.0",
        ],
        "performance": [
            "numba>=0.56.0",
            "redis>=4.3.0",
            "celery>=5.2.0",
        ],
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "black>=22.6.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
            "sphinx>=5.1.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "all": [
            # GPU support
            "onnxruntime-gpu>=1.15.0",
            # Database
            "psycopg2-binary>=2.9.0",
            # Web framework
            "fastapi>=0.85.0",
            "uvicorn>=0.18.0",
            "streamlit>=1.12.0",
            # Notifications
            "slack-sdk>=3.18.0",
            "pymsteams>=0.2.0",
            # Performance
            "numba>=0.56.0",
            "redis>=4.3.0",
            # Development
            "pytest>=7.1.0",
            "black>=22.6.0",
        ]
    },
    package_data={
        "attendance_system": [
            "config/*.yaml",
            "config/*.json",
        ],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "attendance-system=attendance_system.cli:main",
            "attendance-demo=attendance_system.demo:run_demo",
            "attendance-server=attendance_system.server:start_server",
        ],
    },
    zip_safe=False,
    test_suite="tests",
    tests_require=[
        "pytest>=7.1.0",
        "pytest-cov>=3.0.0",
        "pytest-mock>=3.8.0",
    ],
    license="MIT",
    platforms=["any"],
) 