import cv2
import torch
import os
import time
import numpy as np
from ultralytics import YOLO
from config.config import settings
from database.db import DatabaseManager
from config.enhanced_logger import EnhancedLogger

# استفاده از لاگر پیشرفته تعریف شده
elog = EnhancedLogger("ai.process")

class ProfileProcessor:
    def __init__(self):
        elog.info("=" * 80)
        elog.info("🚀 Initializing ProfileProcessor...")

        # راه‌اندازی دیتابیس
        self.db = DatabaseManager()

        # تشخیص دستگاه (GPU/CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # گزارش سخت‌افزار هنگام استارتاپ
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            elog.info(f"✓ GPU DETECTED: {gpu_name}")
            elog.info(f"✓ GPU Memory: {gpu_memory:.2f} GB")
        else:
            elog.warning("⚠ No GPU detected - Using CPU (performance will be slower)")

        # لود مدل YOLO
        model_path = settings.MODEL_PATH if os.path.exists(settings.MODEL_PATH) else "yolo11n.pt"
        elog.info(f"Loading YOLO model from: {model_path}")

        self.model = YOLO(model_path)
        self.model.to(self.device)
        elog.info(f"✓ Model loaded successfully on {self.device.upper()}")

        # تنظیم مرزها
        l1 = settings.LINE_LEFT_X
        l2 = settings.LINE_RIGHT_X
        self.left_boundary = min(l1, l2)
        self.right_boundary = max(l1, l2)
        elog.info(f"Detection boundaries: LEFT={self.left_boundary}px, RIGHT={self.right_boundary}px")

        # مقداردهی اولیه شمارنده‌ها
        self.reset_counters()
        elog.info("✓ ProfileProcessor initialized successfully")
        elog.info("=" * 80)

    def reset_counters(self):
        """ریست کردن تمام counter ها طبق منطق استاندارد"""
        self.candidate_ids = set()
        self.ids_counted_g1 = set()
        self.ids_counted_g2 = set()
        self.g1_count = 0
        self.g2_count = 0
        
        self.total_processed_frames = 0
        self.total_inference_time = 0 

    def get_gpu_memory(self):
        """دریافت مصرف لحظه‌ای حافظه گرافیک"""
        if self.device == 'cuda':
            return torch.cuda.memory_allocated(0) / 1024**2 # MB
        return 0

    def process_frame(self, frame, stream_id="default", latency=0.0):
        start_time = time.time()

        # تشخیص دستگاه فعلی
        try:
            current_device = str(next(self.model.parameters()).device)
            if "cuda" in current_device:
                current_device = f"CUDA:{torch.cuda.current_device()}"
        except:
            current_device = self.device.upper()

        # انجام اینفرنس
        results = self.model.track(frame, persist=True, verbose=False, device=self.device)
        r = results[0]

        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        self.total_processed_frames += 1

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # رسم خطوط فرضی
        cv2.line(annotated, (self.left_boundary, 0), (self.left_boundary, h), (0, 255, 0), 5)
        cv2.line(annotated, (self.right_boundary, 0), (self.right_boundary, h), (0, 0, 255), 5)

        # منطق شمارش (Logic Logic from process_old.py)
        if r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)

            for box, obj_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)

                # 1. کاندید (وسط)
                if self.left_boundary < center_x < self.right_boundary:
                    self.candidate_ids.add(obj_id)
                    color = (255, 0, 0) # Blue

                # 2. گرید 1 (خط چپ)
                elif center_x <= self.left_boundary and obj_id in self.candidate_ids:
                    if obj_id not in self.ids_counted_g1:
                        self.g1_count += 1
                        self.ids_counted_g1.add(obj_id)
                        
                        self.db.insert_record("Grade_1")
                        
                        # 🟢 لاگ سبز شمارش (بدون تکرار)
                        elog.log_profile_event("Grade_1", obj_id, self.g1_count, self.g2_count)
                        
                    color = (0, 255, 0) # Green

                # 3. گرید 2 (خط راست)
                elif center_x >= self.right_boundary and obj_id in self.candidate_ids:
                    if obj_id not in self.ids_counted_g2:
                        self.g2_count += 1
                        self.ids_counted_g2.add(obj_id)
                        
                        self.db.insert_record("Grade_2")
                        
                        # 🔴 لاگ قرمز شمارش (بدون تکرار)
                        elog.log_profile_event("Grade_2", obj_id, self.g1_count, self.g2_count)
                        
                    color = (0, 0, 255) # Red
                
                else:
                    color = (100, 100, 100)

                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                cv2.putText(annotated, f"ID:{obj_id}", (int(x1), int(y1)-10), 0, 0.7, color, 2)

        # آمار روی تصویر
        cv2.putText(annotated, f"Grade_1 Profile Counting : {self.g1_count}", (40, 70), 0, 1.3, (0, 255, 0), 4)
        cv2.putText(annotated, f"Grade_2 Profile Counting : {self.g2_count}", (w-620, 70), 0, 1.3, (0, 0, 255), 4)

        # لاگ وضعیت فریم (⚙️)
        vram_usage = self.get_gpu_memory()
        elog.log_frame_summary(
            stream_name=stream_id,
            device=current_device,
            redis_latency=latency if latency is not None else 0.0,
            inference_time=inference_time,
            memory_usage=vram_usage
        )

        return annotated, self.g1_count, self.g2_count