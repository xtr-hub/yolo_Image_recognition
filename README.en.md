# YOLO Object Recognition Service

A Web service based on YOLOv8 for object detection in images, videos, and real-time cameras, supporting 80 classes from the COCO dataset.

## Features

- **Image Detection**: Upload an image to quickly detect objects in it
- **Video Detection**: Supports video file detection, annotates results and allows downloading of annotated videos
- **Video Tracking Mode**: Uses YOLO official tracking API for cross-frame object tracking and unique object counting
- **Batch Image Detection**: Supports batch processing of multiple images for improved efficiency
- **Real-time Communication**: WebSocket support for real-time camera detection
- **Web Interface**: Provides a user-friendly visual operation interface with PWA support
- **Health Check**: API endpoint for monitoring service status with performance stats
- **Multi-device Support**: Supports GPU acceleration and CPU inference
- **Class Filtering**: Support filtering detection results by class
- **Memory Management**: Intelligent memory management with dynamic batch size adjustment

## Technology Stack

- Python 3.8+
- FastAPI - High-performance web framework
- Ultralytics YOLOv8 - Object detection model
- OpenCV - Image and video processing
- WebSocket - Real-time communication
- imageio - Video encoding/decoding
- PyTorch - Deep learning framework
- psutil - System monitoring

## Quick Start

### Environment Preparation

Ensure Python 3.8 or higher is installed, and install the following dependencies:

```bash
pip install -r requirements.txt
```

### Start the Service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

After the service starts, access `http://localhost:8000` to use the web interface.

## API Documentation

### Health Check

```
GET /health
```

Check the service status, returns model info, device info, and performance statistics.

**Response Example:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "batch_processing_enabled": true,
  "supported_classes": ["person", "bicycle", "car", ...],
  "performance_stats": {...}
}
```

### Image Detection

```
POST /api/v1/detect
Content-Type: multipart/form-data

Parameters:
- file: Image file (JPG, PNG, BMP)
- classes: Classes to detect, comma-separated, e.g., 'person' or 'person,car'
- conf_threshold: Confidence threshold (0.1-0.9), default 0.5

Response: Detection results in JSON format
```

**Response Example:**
```json
{
  "success": true,
  "object_count": 2,
  "objects": [
    {
      "bbox": {
        "x1": 100,
        "y1": 50,
        "x2": 200,
        "y2": 300,
        "width": 100,
        "height": 250
      },
      "confidence": 0.95,
      "class_id": 0,
      "class_name": "person"
    }
  ],
  "inference_time_ms": 45.2,
  "image_shape": {
    "height": 480,
    "width": 640
  },
  "annotated_image": "data:image/jpeg;base64,..."
}
```

### Video Detection

```
POST /api/v1/video
Content-Type: multipart/form-data

Parameters:
- file: Video file (MP4, AVI, MOV)
- return_video: Whether to return the processed video ("true" or "false", default: "true")
- classes: Classes to detect, comma-separated, e.g., 'person' or 'person,car'
- use_batch_processing: Whether to use batch processing optimization (default: true)
- batch_size: Batch size (1-32, default: 8)
- frame_interval: Frame processing interval (1=every frame, 2=every other frame, default: 1)

Response: Detection result JSON or processed video file
```

**Response Example (JSON):**
```json
{
  "total_frames": 150,
  "processed_frames": 150,
  "fps": 30.0,
  "duration": 5.0,
  "resolution": {
    "width": 1920,
    "height": 1080
  },
  "frames_with_detection": 23,
  "frames": [
    {
      "frame": 10,
      "timestamp": 0.33,
      "object_count": 1,
      "objects": [...]
    }
  ]
}
```

### Video Detection (Tracking Mode)

Uses YOLO official tracking API for cross-frame object tracking:

```python
result = detector.process_video_file_track(
    video_path,
    output_path,
    classes=['person'],
    conf=0.5
)
```

**Response includes:**
- `class_counts`: Count of each class detected
- `frames`: Detection results for each frame

### Batch Image Detection

```
POST /api/v1/batch/detect
Content-Type: multipart/form-data

Parameters:
- image_files: List of image files
- classes: Classes to detect, comma-separated
- max_workers: Maximum worker threads (1-10)
- batch_size: Batch size (1-100)

Response: Batch detection results
```

**Response Example:**
```json
{
  "success": true,
  "total_processed": 10,
  "failed_count": 0,
  "results": [
    {...},
    {...}
  ]
}
```

### WebSocket Real-time Detection

```
WS /ws/detect
```

Supports real-time detection of image frames, suitable for live camera detection scenarios.

**Message Format:**
- Input: `{"image": "data:image/jpeg;base64,..."}`
- Output: Detection result JSON

## Web Interface

After the service starts, a visual interface is provided with support for:

- **Image Detection Mode**: Upload images and view detection results
- **Camera Mode**: Real-time detection of objects in camera feed
- **Video Mode**: Upload video files, process and download annotated videos
- **Real-time Statistics**: Display detection count, inference time and other information
- **Class Selection**: Option to detect specific classes of objects
- **Confidence Adjustment**: Adjust detection confidence threshold
- **PWA Support**: Can be added to home screen as standalone application

## Configuration Options

Configure service behavior through environment variables and configuration files:

### Environment Variables

- `YOLO_MODEL`: Specify YOLO model to use, defaults to `yolov8n`
  - Example: `export YOLO_MODEL=yolov8s` to use small model
  - Example: `export YOLO_MODEL=yolov8l` to use large model for higher accuracy

### Configuration File (app/core/config.py)

```python
BATCH_PROCESSING = {
    "default_max_workers": 4,
    "default_batch_size": 10,
    "max_batch_size": 100,
    "max_workers": 10,
    "memory_threshold": 80,  # Memory usage threshold percentage
}
```

## COCO Dataset Classes

Supports 80 common object classes:

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | person | 1 | bicycle |
| 2 | car | 3 | motorcycle |
| 4 | airplane | 5 | bus |
| 6 | train | 7 | truck |
| 8 | boat | 15 | cat |
| 16 | dog | 39 | bottle |
| ... | ... | 79 | toothbrush |

See `COCO_CLASSES` dictionary for complete class list.

## Model Information

The project defaults to using the `yolov8n.pt` pre-trained model. Other YOLOv8 models can be substituted as needed to balance speed and accuracy:

- `yolov8n.pt`: nano version, fastest but lowest accuracy
- `yolov8s.pt`: small version, balanced speed and accuracy
- `yolov8m.pt`: medium version, slower but higher accuracy
- `yolov8l.pt`: large version, slower but much higher accuracy
- `yolov8x.pt`: extra-large version, slowest but highest accuracy

On first startup, if no local model file exists, the system will automatically download the specified model.

## Project Structure

```
├── app/                   # Application code
│   ├── api/               # API routes
│   │   └── routes.py      # Route definitions
│   ├── core/              # Configuration module
│   │   └── config.py      # Configuration files
│   ├── models/            # Detection models
│   │   └── detector.py    # Object detector
│   ├── utils/             # Utility modules
│   │   └── batch_processor.py  # Batch processor
│   └── main.py            # Application entry point
├── static/                # Frontend static resources
│   ├── index.html         # Main page
│   └── manifest.json      # PWA configuration
├── test_api.py            # API tests
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
├── LICENSE                # License
├── cert.pem               # SSL certificate (optional)
└── key.pem                # SSL key (optional)
```

## Deployment Guide

### Production Deployment

Deploy production environment using Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Or use Nginx + Uvicorn combination deployment.

### HTTPS Deployment

To support camera functionality working properly in production environments (especially under HTTPS), SSL certificates can be used:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 443 --ssl-keyfile ./key.pem --ssl-certfile ./cert.pem
```

## Performance Optimization Suggestions

- **GPU Acceleration**: Run on machines supporting CUDA for 10-50x inference speedup
- **Model Selection**: Choose appropriate model for your scenario, recommend yolov8n/yolov8s for real-time applications
- **Batch Processing**: Use batch processing optimization for batch detection to improve throughput
- **Frame Interval**: Increase frame_interval to reduce processing time for videos
- **Memory Management**: System automatically monitors memory usage and adjusts batch size dynamically

## Troubleshooting

1. **Model Download Failure**: Check network connectivity, or manually download model files to project root directory
   - Download URL: https://github.com/ultralytics/assets/releases
2. **Camera Permission Issues**: Ensure browser has authorized camera access, camera only works under HTTPS
3. **Video Processing Failure**: Check video format support and disk space
4. **Out of Memory**: Reduce batch size or increase memory threshold configuration
5. **WebSocket Connection Failure**: Check firewall settings and proxy configuration

## License

This project is open-source; please refer to the [LICENSE](./LICENSE) file for specific licensing details.

## Contributing

Feel free to submit Issues and Pull Requests to improve the project.
