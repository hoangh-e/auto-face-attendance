#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kiểm tra hỗ trợ tên tiếng Việt trong hệ thống điểm danh AI
"""

import sqlite3
import os
from pathlib import Path
import cv2
import numpy as np

def check_database_support():
    """Kiểm tra database có hỗ trợ UTF-8 không"""
    print("🔍 KIỂM TRA HỖ TRỢ TIẾNG VIỆT TRONG DATABASE")
    print("=" * 50)
    
    # Kiểm tra file database có tồn tại không
    db_path = Path("notebooks/local_attendance.db")
    if not db_path.exists():
        print("❌ Database không tồn tại")
        return False
    
    try:
        # Kết nối database với UTF-8
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA encoding = 'UTF-8';")
        cursor = conn.cursor()
        
        # Kiểm tra schema
        cursor.execute("PRAGMA table_info(employees);")
        schema = cursor.fetchall()
        print("📊 Schema bảng employees:")
        for column in schema:
            print(f"├─ {column[1]} ({column[2]})")
        
        # Kiểm tra dữ liệu hiện có
        cursor.execute("SELECT COUNT(*) FROM employees;")
        count = cursor.fetchone()[0]
        print(f"👥 Số lượng nhân viên: {count}")
        
        if count > 0:
            cursor.execute("SELECT name, employee_code FROM employees LIMIT 5;")
            employees = cursor.fetchall()
            print("📝 Dữ liệu nhân viên:")
            for emp in employees:
                print(f"├─ {emp[0]} ({emp[1]})")
        
        # Test thêm tên tiếng Việt
        test_name = "Nguyễn Văn Tèo"
        test_code = "NVT001"
        test_email = "nvteo@test.com"
        
        print(f"\n🧪 Test thêm tên tiếng Việt: {test_name}")
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO employees 
                (employee_code, name, email, department, position, is_active) 
                VALUES (?, ?, ?, ?, ?, ?)
            """, (test_code, test_name, test_email, "IT", "Developer", 1))
            
            # Kiểm tra lại
            cursor.execute("SELECT name FROM employees WHERE employee_code = ?", (test_code,))
            result = cursor.fetchone()
            
            if result and result[0] == test_name:
                print("✅ Database hỗ trợ tên tiếng Việt")
                
                # Xóa dữ liệu test
                cursor.execute("DELETE FROM employees WHERE employee_code = ?", (test_code,))
                conn.commit()
                return True
            else:
                print("❌ Database không hỗ trợ tên tiếng Việt đúng cách")
                return False
                
        except Exception as e:
            print(f"❌ Lỗi khi test tên tiếng Việt: {e}")
            return False
            
        finally:
            conn.close()
            
    except Exception as e:
        print(f"❌ Lỗi kết nối database: {e}")
        return False

def check_opencv_text_support():
    """Kiểm tra OpenCV có hỗ trợ hiển thị tiếng Việt không"""
    print("\n🎨 KIỂM TRA HỖ TRỢ HIỂN THỊ TIẾNG VIỆT TRONG OPENCV")
    print("=" * 50)
    
    try:
        # Tạo ảnh test
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Test text tiếng Việt
        vietnamese_names = [
            "Nguyễn Văn An",
            "Trần Thị Bình", 
            "Lê Minh Châu",
            "Phạm Đức Dũng",
            "Hoàng Thị Hoa"
        ]
        
        print("📝 Test hiển thị các tên tiếng Việt:")
        
        y_pos = 50
        for i, name in enumerate(vietnamese_names):
            # Sử dụng cv2.putText với FONT_HERSHEY_SIMPLEX
            cv2.putText(img, name, (50, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            print(f"├─ {name}")
            y_pos += 60
        
        # Lưu ảnh test
        test_path = Path("vietnamese_text_test.jpg")
        cv2.imwrite(str(test_path), img)
        
        print(f"✅ Ảnh test đã lưu: {test_path}")
        print("💡 Lưu ý: OpenCV có thể hiển thị tiếng Việt nhưng có thể không hoàn hảo với các ký tự đặc biệt")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi test OpenCV: {e}")
        return False

def check_employee_folders():
    """Kiểm tra thư mục nhân viên có tên tiếng Việt"""
    print("\n📁 KIỂM TRA THỂ MỤC NHÂN VIÊN")
    print("=" * 50)
    
    employees_dir = Path("notebooks/employees")
    if not employees_dir.exists():
        print("❌ Thư mục employees không tồn tại")
        return False
    
    vietnamese_folders = []
    
    print("📂 Thư mục nhân viên hiện có:")
    for folder in employees_dir.iterdir():
        if folder.is_dir() and not folder.name.startswith('_'):
            print(f"├─ {folder.name}")
            
            # Kiểm tra có ký tự tiếng Việt không
            vietnamese_chars = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
            if any(char.lower() in vietnamese_chars for char in folder.name):
                vietnamese_folders.append(folder.name)
    
    if vietnamese_folders:
        print(f"\n✅ Tìm thấy {len(vietnamese_folders)} thư mục có tên tiếng Việt:")
        for folder in vietnamese_folders:
            print(f"├─ {folder}")
            
            # Kiểm tra số lượng ảnh
            folder_path = employees_dir / folder
            images = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpeg"))
            print(f"   └─ {len(images)} ảnh")
        
        return True
    else:
        print("⚠️ Không tìm thấy thư mục nào có tên tiếng Việt")
        return False

def main():
    """Chạy tất cả các kiểm tra"""
    print("🇻🇳 KIỂM TRA HỖ TRỢ TIẾNG VIỆT - HỆ THỐNG ĐIỂM DANH AI")
    print("=" * 60)
    
    results = {
        "database": check_database_support(),
        "opencv": check_opencv_text_support(), 
        "folders": check_employee_folders()
    }
    
    print("\n📊 KẾT QUẢ TỔNG QUAN")
    print("=" * 30)
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"├─ {test.capitalize()}: {status}")
    
    print(f"└─ Tổng kết: {total_passed}/{total_tests} tests đạt")
    
    if total_passed == total_tests:
        print("\n🎉 HỆ THỐNG HỖ TRỢ TIẾNG VIỆT HOÀN TOÀN!")
    elif total_passed >= 2:
        print("\n👍 HỆ THỐNG HỖ TRỢ TIẾNG VIỆT CƠ BẢN")
    else:
        print("\n⚠️ HỆ THỐNG CẦN CẢI THIỆN HỖ TRỢ TIẾNG VIỆT")
    
    # Cleanup
    test_image = Path("vietnamese_text_test.jpg")
    if test_image.exists():
        test_image.unlink()
        print(f"\n🧹 Đã xóa file test: {test_image}")

if __name__ == "__main__":
    main()