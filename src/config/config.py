import os
from dotenv import load_dotenv
from pathlib import Path

# بارگذاری .env از مسیر اپلیکیشن در داکر
env_path = Path('/app/.env')
load_dotenv(dotenv_path=env_path)

class Settings:  
    # Camera & AI ROI
    RTSP_URL = os.getenv("RTSP_URL")
    
    LINE_LEFT_X = int(os.getenv("LINE_LEFT_X", 0))
    LINE_RIGHT_X = int(os.getenv("LINE_RIGHT_X", 0))
    LINE_HORIZONTAL = int(os.getenv("LINE_HORIZONTAL", 0))

    MODEL_PATH = os.getenv("MODEL_PATH", "/app/src/ai/weights/best.pt")

    # Video Processing
    FRAME_INTERVAL = int(os.getenv("FRAME_INTERVAL", 3))

    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

    # Reporting Configuration
    _duration_str = os.getenv("REPORT_DURATION_HOURS", "")
    if _duration_str.lower() in ["none", "", "null"]:
        REPORT_DURATION_HOURS = None
    else:
        try:
            REPORT_DURATION_HOURS = int(_duration_str)
        except ValueError:
            REPORT_DURATION_HOURS = None  # Fallback to infinite if invalid

settings = Settings()