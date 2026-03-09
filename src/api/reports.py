import time
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import logging
from config.config import settings

logger = logging.getLogger(__name__)
TEHRAN_TZ = ZoneInfo("Asia/Tehran")

class StatsReporter:
    def __init__(self, processor_instance):
        self.processor = processor_instance
        self.start_time = datetime.now(TEHRAN_TZ)
        
        logger.info(f"{'='*60}")
        logger.info(f"SYSTEM STARTED AT: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"REPORT DURATION: {'INFINITE' if settings.REPORT_DURATION_HOURS is None else f'{settings.REPORT_DURATION_HOURS} HOURS'}")
        logger.info(f"{'='*60}")

    def run_periodic_reporting(self):
        """اجرای سیکل گزارش‌گیری بر اساس تنظیمات .env"""
        last_g1 = 0
        last_g2 = 0
        
        target_hours = settings.REPORT_DURATION_HOURS
        current_hour = 1
        
        mode_str = "INFINITE" if target_hours is None else f"{target_hours}-HOUR"
        logger.info(f"🚀 {mode_str} Reporting System activated.")

        while True:
            if target_hours is not None and current_hour > target_hours:
                break

            time.sleep(3600)
            
            current_g1 = self.processor.g1_count
            current_g2 = self.processor.g2_count
            
            hourly_g1 = current_g1 - last_g1
            hourly_g2 = current_g2 - last_g2
            
            last_g1, last_g2 = current_g1, current_g2
            
            now_time = datetime.now(TEHRAN_TZ).strftime("%H:%M:%S")
            log_line = f"Hour {current_hour:02d} | Time: {now_time} | Hourly: G1:{hourly_g1}, G2:{hourly_g2} | Total so far: {current_g1+current_g2}"
            
            logger.info(f"📊 {log_line}")
            
            current_hour += 1

        self.generate_final_report()

    def generate_final_report(self):
        total_frames = self.processor.total_processed_frames
        total_inf = self.processor.total_inference_time
        
        avg_ms = (total_inf / total_frames * 1000) if total_frames > 0 else 0
        final_g1 = self.processor.g1_count
        final_g2 = self.processor.g2_count
        
        end_time = datetime.now(TEHRAN_TZ).strftime('%Y-%m-%d %H:%M:%S')
        duration_hours = settings.REPORT_DURATION_HOURS

        logger.info(f"{'-'*30}")
        logger.info(f"FINAL SESSION SUMMARY ({duration_hours} HOURS)")
        logger.info(f"Total Grade 1: {final_g1}")
        logger.info(f"Total Grade 2: {final_g2}")
        logger.info(f"Combined Total: {final_g1 + final_g2}")
        logger.info(f"Average Inference Time: {avg_ms:.2f} ms per frame")
        logger.info(f"Total Frames Processed: {total_frames}")
        logger.info(f"Session Finished at: {end_time}")
        logger.info(f"{'='*60}")
        
        logger.info(f"🏁 {duration_hours}-Hour period finished. Closing system...")
        os._exit(0)