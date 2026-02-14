# YOLO Image Recognition Service

A Web service based on YOLOv8 for person detection in images and videos.

## Features

- **Image Detection**: Upload an image to quickly detect persons in it.
- **Video Detection**: Supports video file detection with annotated results.
- **Real-time Communication**: WebSocket support for real-time detection result streaming.
- **Web Interface**: Provides a user-friendly visual operation interface.
- **Health Check**: API endpoint for monitoring service status.

## Technology Stack

- Python 3.8+
- FastAPI - High-performance web framework
- Ultralytics YOLOv8 - Object detection model
- OpenCV - Image and video processing
- WebSocket - Real-time communication

## Quick Start

### Environment Preparation

Ensure Python 3.8 or higher is installed.

### Install Dependencies

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

### Image Detection

```
POST /detect
Content-Type: multipart/form-data

Parameters:
- file: Image file

Response: Detection results in JSON format
```

### Video Detection

```
POST /video
Content-Type: multipart/form-data

Parameters:
- file: Video file
- return_video: Whether to return the annotated video (true/false, default: true)

Response: Detection results in JSON format
```

### WebSocket Real-time Detection

```
WS /ws/detect
```

Supports real-time detection of image frames.

## Web Interface

After the service starts, a visual interface is provided with support for:

- Image upload and preview
- Video upload and progress display
- Visualization of detection results
- Summary of statistics

## Model Information

The project uses the pre-trained `yolov8s.pt` model. You can replace it with other YOLOv8 models as needed to balance speed and accuracy.

## Project Structure

```
├── app/
│   ├── api/           # API routes
│   ├── core/          # Configuration module
│   ├── models/        # Detection models
│   └── main.py        # Application entry point
├── static/            # Frontend static assets
├── test_api.py        # API tests
├── requirements.txt   # Dependency list
└── yolov8s.pt         # YOLOv8 model
```

## License

This project is open-source; please refer to the project repository for specific licensing details.