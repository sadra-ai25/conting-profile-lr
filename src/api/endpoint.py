from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
import threading
import os
import shutil
import time
import logging
from capture.producer import camera_producer, video_producer
from ai.process import ProfileProcessor
from ai.consumer import worker_process
from api.reports import StatsReporter
from config.config import settings
from database.db import DatabaseManager
from pydantic import BaseModel
from typing import Optional, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

root_logger = logging.getLogger()
logger = logging.getLogger(__name__)

app = FastAPI()

logger.info("=" * 80)
logger.info("🚀 Profile Counting System Initializing...")
processor = ProfileProcessor()
rtsp_stop_event = threading.Event()
logger.info("=" * 80)

@app.on_event("startup")
async def startup_event():
    reporter = StatsReporter(processor)
    threading.Thread(target=reporter.run_periodic_reporting, name="Reporter_Thread", daemon=True).start()
    logger.info("📊 Stats Reporter Active (Console Only).")
    
    threading.Thread(target=auto_start_sequence, daemon=True).start()

def auto_start_sequence():
    time.sleep(5)
    logger.info("🛰️ Auto-starting RTSP Pipeline...")
    start_stream()

@app.post("/rtsp/start")
def start_stream():
    is_running = any(t.name == "RTSP_Producer" for t in threading.enumerate())
    if not is_running:
        rtsp_stop_event.clear()
        t_prod = threading.Thread(target=camera_producer, args=(rtsp_stop_event,), name="RTSP_Producer")
        t_prod.daemon = True
        t_prod.start()
        t_cons = threading.Thread(target=worker_process, args=(processor, rtsp_stop_event), name="RTSP_Consumer")
        t_cons.daemon = True
        t_cons.start()
        logger.info("✅ RTSP Pipeline started successfully.")
        return {"status": "Started"}
    return {"status": "Already running"}

@app.post("/rtsp/stop")
def stop_stream():
    rtsp_stop_event.set()
    logger.info("🛑 RTSP Pipeline stop signal sent.")
    return {"status": "Stopped"}

def cleanup_file(path):
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"🗑️ Cleaned up temporary file: {path}")
    except Exception as e:
        logger.error(f"❌ Failed to cleanup file {path}: {e}")

def run_analysis_and_clean(temp_path, proc):
    report_log_file = None 
    try:
        video_producer(temp_path, proc, report_log_file)
    finally:
        cleanup_file(temp_path)

@app.post("/video")
async def analyze_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    logger.info(f"📡 Request: Analyze Video -> {file.filename}")
    temp_filename = f"temp_{int(time.time())}_{file.filename}"
    temp_path = os.path.join("/app/outputs", temp_filename)
    with open(temp_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)
    logger.info("✅ Upload complete. Analysis queued in background.")
    background_tasks.add_task(run_analysis_and_clean, temp_path, processor)
    return {
        "status": "Analysis in progress",
        "file": file.filename,
        "logs": "Check container logs (stdout) for details"
    }

class TimeRangeRequest(BaseModel):
    start_time: Optional[str] = None  # YYYY-MM-DD HH:MM:SS
    end_time: Optional[str] = None    # YYYY-MM-DD HH:MM:SS

@app.post("/reports")
async def get_report_range(request: TimeRangeRequest):
    db = DatabaseManager()
    try:
        data = db.get_counts(
            start_time=request.start_time,
            end_time=request.end_time
        )
        return {"status": "success",
                "count": len(data),
                "data": data}

    except Exception as e:
        logger.error(f"❌ Error in /reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()