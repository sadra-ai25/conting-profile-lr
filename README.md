# Profile Counting — Left/Right Conveyor

![Python](https://img.shields.io/badge/Python-3.10-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-green) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red) ![Redis](https://img.shields.io/badge/Redis-Queue-red) ![Docker](https://img.shields.io/badge/Docker-Compose-blue)

Real-time steel profile counting system for left-right conveyor lines. Detects and counts steel profiles passing a configurable virtual counting line using YOLOv8, with Redis-based frame queuing and production reporting via REST API.

## Features

- **Virtual counting line** — configurable left (LINE_LEFT_X) and right (LINE_RIGHT_X) boundary triggers
- **YOLOv8 detection** — high-accuracy steel profile detection in real time
- **Redis frame queue** — decoupled producer/consumer for reliable stream processing
- **RTSP stream support** — connects to any IP camera via RTSP URL
- **Periodic reports** — configurable reporting window (hours) via `REPORT_DURATION_HOURS`
- **REST API** — start/stop stream processing and query shift counts
- **Structured logging** — enhanced logger with rotation and level control

## Tech Stack

| Component | Technology |
|---|---|
| AI Model | YOLOv8 (Ultralytics) |
| API Server | FastAPI + Uvicorn |
| Frame Queue | Redis |
| Containerization | Docker Compose |
| Camera Capture | OpenCV |

## Architecture

```
RTSP Camera
      │
      ▼
 Camera Producer (Thread)
   - Captures frames at FRAME_INTERVAL
   - Pushes raw frames to Redis queue
      │
      ▼
  Redis Queue
      │
      ▼
 Frame Consumer (Thread)
   - Pulls frames from Redis
   - Runs YOLOv8 detection
   - Checks centroid position vs. counting lines
   - Increments counter on crossing
      │
      ▼
 FastAPI REST API  (counts & reports)
```

## Prerequisites

- Docker & Docker Compose
- YOLOv8 model weights at `src/ai/weights/best.pt`
- RTSP-capable IP camera or MediaMTX relay

## Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/sadra-ai25/conting-profile-lr.git
cd conting-profile-lr

# 2. Configure environment
cp .env.example .env   # edit with your values

# 3. Place model weights
mkdir -p src/ai/weights
cp /path/to/best.pt src/ai/weights/

# 4. Start services
docker compose up -d --build
```

## Configuration

| Key | Description | Example |
|---|---|---|
| `RTSP_URL` | Camera RTSP stream URL | `rtsp://mediamtx:8554/mystream` |
| `LINE_LEFT_X` | Left boundary pixel X position | `953` |
| `LINE_RIGHT_X` | Right boundary pixel X position | `1181` |
| `LINE_HORIZONTAL` | Horizontal reference line Y | `600` |
| `MODEL_PATH` | YOLOv8 weights path | `/app/src/ai/weights/best.pt` |
| `FRAME_INTERVAL` | Process every Nth frame | `3` |
| `REDIS_HOST` | Redis hostname | `redis` |
| `REDIS_PORT` | Redis port | `6379` |
| `REPORT_DURATION_HOURS` | Reporting window in hours (`None` = unlimited) | `8` |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health and processing status |
| `POST` | `/start` | Start camera stream processing |
| `POST` | `/stop` | Stop camera stream processing |
| `GET` | `/count` | Get current profile count for active session |
| `GET` | `/report` | Get production report for configured time window |
| `POST` | `/reset` | Reset counter for new shift |

### Example: Get Current Count

```bash
curl http://localhost:8000/count
```

```json
{
  "count": 142,
  "session_start": "2024-01-15T08:00:00",
  "elapsed_minutes": 47
}
```

## Counting Logic

A steel profile is counted when:
1. YOLOv8 detects an object in the frame
2. The centroid X of the bounding box crosses `LINE_LEFT_X` → `LINE_RIGHT_X`
3. The object has not been counted in the current tracking window

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

## License

MIT
