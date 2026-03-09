import redis
import pickle
import logging
import cv2
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from ai.process import ProfileProcessor
from config.config import settings
from config.enhanced_logger import EnhancedLogger

elog = EnhancedLogger("Consumer")
TEHRAN_TZ = ZoneInfo("Asia/Tehran")

def worker_process(processor: ProfileProcessor, stop_event):
    r_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0)
    stream_name = "camera_processing_tasks"
    
    elog.info("🚀 Consumer Ready. Waiting for Redis streams...")

    while not stop_event.is_set():
        messages = r_client.xread({stream_name: '$'}, count=1, block=1000)
        
        if not messages:
            continue
            
        for _, msg_list in messages:
            for msg_id, data in msg_list:
                try:
                    task = pickle.loads(data[b'data'])
                    frame_id = task['frame_id']
                    timestamp_str = task.get('timestamp_tehran')
                    
                    # محاسبه Latency
                    latency = 0.0
                    if timestamp_str:
                        sent_time = datetime.fromisoformat(timestamp_str)
                        current_time = datetime.now(TEHRAN_TZ)
                        latency = (current_time - sent_time).total_seconds()

                    frame_data = r_client.get(f"frame:{frame_id}")
                    if frame_data:
                        nparr = np.frombuffer(frame_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        # پردازش (با ارسال latency برای لاگ)
                        processor.process_frame(frame, stream_id="RTSP_Live", latency=latency)
                        
                        r_client.delete(f"frame:{frame_id}")
                    
                except Exception as e:
                    elog.error(f"Worker Error: {e}")

