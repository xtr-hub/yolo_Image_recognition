# YOLO Image Recognition Service

A Web service based on YOLOv8 for person detection in images, videos, and real-time cameras.

## Features

- **Image Detection**: Upload an image to quickly detect persons in it
- **Video Detection**: Supports video file detection, annotates results and allows downloading of annotated videos
- **Real-time Communication**: WebSocket support for real-time camera detection
- **Web Interface**: Provides a user-friendly visual operation interface with PWA support
- **Health Check**: API endpoint for monitoring service status
- **Multi-device Support**: Supports GPU acceleration and CPU inference

## Technology Stack

- Python 3.8+
- FastAPI - High-performance web framework
- Ultralytics YOLOv8 - Object detection model
- OpenCV - Image and video processing
- WebSocket - Real-time communication
- imageio - Video encoding/decoding
- PyTorch - Deep learning framework

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

Check the service status.

**Response Example:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### Image Detection

```
POST /api/v1/detect
Content-Type: multipart/form-data

Parameters:
- file: Image file (JPG, PNG, BMP)

Response: Detection results in JSON format
```

**Response Example:**
```json
{
  "success": true,
  "person_count": 2,
  "persons": [
    {
      "bbox": {
        "x1": 100,
        "y1": 50,
        "x2": 200,
        "y2": 300,
        "width": 100,
        "height": 250
      },
      "confidence": 0.95
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
      "person_count": 1,
      "persons": [...]
    }
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
- **Camera Mode**: Real-time detection of people in camera feed
- **Video Mode**: Upload video files, process and download annotated videos
- **Real-time Statistics**: Display detection count, inference time and other information
- **PWA Support**: Can be added to home screen as standalone application

## Configuration Options

Configure service behavior through environment variables:

- `YOLO_MODEL`: Specify YOLO model to use, defaults to `yolov8s`
  - Example: `export YOLO_MODEL=yolov8n` to use lightweight model
  - Example: `export YOLO_MODEL=yolov8l` to use large model for higher accuracy

## Model Information

The project defaults to using the `yolov8s.pt` pre-trained model. Other YOLOv8 models can be substituted as needed to balance speed and accuracy:

- `yolov8n.pt`: nano version, fastest but lowest accuracy
- `yolov8s.pt`: small version, balanced speed and accuracy (recommended)
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
│   │   └── detector.py    # Person detector
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

- On machines supporting CUDA, use GPU acceleration for inference
- Select appropriate YOLO model versions based on requirements
- Adjust video processing frame rate to balance real-time performance and efficiency
- For large batch tasks, consider asynchronous processing mechanisms

## Troubleshooting

1. **Model Download Failure**: Check network connectivity, or manually download model files to project root directory
2. **Camera Permission Issues**: Ensure browser has authorized camera access
3. **Video Processing Failure**: Check video format support and disk space

## License

This project is open-source; please refer to the [LICENSE](./LICENSE) file for specific licensing details.

## Contributing

Feel free to submit Issues and Pull Requests to improve the project.