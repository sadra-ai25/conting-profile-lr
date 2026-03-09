import redis
import time
import cv2
import logging
import pickle
import uuid
import os
from zoneinfo import ZoneInfo
from datetime import datetime
from config.config import settings

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("capture.producer")

# ------------------------------------------------------------------------------
# Force OpenCV / FFmpeg to use TCP for RTSP
# ------------------------------------------------------------------------------
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ------------------------------------------------------------------------------
# Redis (timeouts added)
# ------------------------------------------------------------------------------
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=0,
    socket_connect_timeout=5,
    socket_timeout=5,
    retry_on_timeout=True,
)

# ------------------------------------------------------------------------------
# Timezone
# ------------------------------------------------------------------------------
TEHRAN_TZ = ZoneInfo("Asia/Tehran")

# ------------------------------------------------------------------------------
# Camera Producer with Robust Reconnect
# ------------------------------------------------------------------------------
def camera_producer(stop_event):
    logger.info("=" * 80)
    logger.info("🚀 Starting RTSP Camera Producer (LRcount / Stable Mode)")
    logger.info("=" * 80)

    rtsp_url = settings.RTSP_URL
    if not rtsp_url:
        logger.error("❌ RTSP URL is not defined in .env")
        return

    stream_name = "camera_processing_tasks"
    frame_count = 0

    backoff = 3
    max_backoff = 30

    while not stop_event.is_set():

        logger.info("🔌 Connecting to RTSP...")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            logger.warning(
                f"❌ Cannot open RTSP stream. Retrying in {backoff}s..."
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
            continue

        logger.info("✅ RTSP Connected")
        backoff = 3  # reset after success

        # ---------------- Frame Loop ----------------
        while not stop_event.is_set():

            ret, frame = cap.read()

            if not ret:
                logger.warning("⚠️ Stream lost. Reconnecting...")
                break

            frame_count += 1

            # Encode frame
            success, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )

            if not success:
                logger.warning("⚠️ JPEG encoding failed, skipping frame")
                continue

            frame_bytes = buffer.tobytes()
            frame_id = str(uuid.uuid4())

            # Save frame with TTL
            try:
                redis_client.setex(
                    f"frame:{frame_id}",
                    10,
                    frame_bytes,
                )
            except Exception as e:
                logger.error(f"❌ Redis SETEX failed: {e}")

            # Push task to Redis Stream
            task_message = {
                "frame_id": frame_id,
                "timestamp_tehran": datetime.now(TEHRAN_TZ).isoformat(),
                "frame_count": frame_count,
            }

            try:
                redis_client.xadd(
                    stream_name,
                    {"data": pickle.dumps(task_message)},
                    maxlen=100,
                )
            except Exception as e:
                logger.error(f"❌ Redis XADD failed: {e}")

            if frame_count % 100 == 0:
                logger.debug(f"📹 Produced frame #{frame_count}")

        # ---------------- Cleanup before reconnect ----------------
        try:
            cap.release()
        except Exception:
            pass

        logger.info("♻️ RTSP connection closed, retrying...")

        time.sleep(backoff)
        backoff = min(backoff * 2, max_backoff)

    logger.info("🛑 Camera Producer Stopped.")


# ------------------------------------------------------------------------------
# Offline Video Producer (kept mostly same, only small hardening)
# ------------------------------------------------------------------------------
def video_producer(video_path, processor_instance, report_file_path=None):

    logger.info("=" * 80)
    logger.info(f"🎞️ Starting Offline Video Analysis | Source: {video_path}")

    processor_instance.reset_counters()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"❌ Failed to open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = settings.FRAME_INTERVAL

    logger.info(f"✓ Video frames: {total_frames} | Interval: {frame_interval}")

    frame_count = 0
    processed_count = 0
    start_time = time.time()

    g1, g2 = 0, 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_interval == 0:

            try:
                _, g1, g2 = processor_instance.process_frame(
                    frame,
                    "Batch_File",
                    latency=None,
                )
                processed_count += 1
            except Exception as e:
                logger.error(f"❌ Processing error: {e}")
                continue

            if processed_count % 50 == 0:
                percent = (frame_count / total_frames) * 100
                logger.info(
                    f"📊 Progress: {percent:.1f}% | Counts -> G1: {g1}, G2: {g2}"
                )

    cap.release()

    elapsed = time.time() - start_time

    logger.info("=" * 80)
    logger.info(f"✅ Video Analysis Finished in {elapsed:.1f}s")
    logger.info(f"📊 Total in this file: Grade 1 = {g1}, Grade 2 = {g2}")
    logger.info("=" * 80)

    # Save report
    if report_file_path:
        try:
            with open(report_file_path, "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write("VIDEO ANALYSIS REPORT\n")
                f.write(f"File: {os.path.basename(video_path)}\n")
                f.write(
                    f"Date: {datetime.now(TEHRAN_TZ).strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"{'-'*30}\n")
                f.write(f"Total Grade 1: {g1}\n")
                f.write(f"Total Grade 2: {g2}\n")
                f.write(f"Total Profiles: {g1 + g2}\n")
                f.write(f"Processed Frames: {processed_count}\n")
                f.write(f"Processing Time: {elapsed:.2f} seconds\n")
                f.write(f"{'='*60}\n")

            logger.info(f"💾 Final Report appended to: {report_file_path}")

        except Exception as e:
            logger.error(f"❌ Failed to save report file: {e}")
