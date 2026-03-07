# YOLO 物体识别服务

基于 YOLOv8 的物体检测 Web 服务，提供图片、视频和实时摄像头中的物体检测功能，支持 COCO 数据集的 80 种常见物体类别。

## 功能特点

- **图片检测**：上传图片即可快速识别图中物体，支持 80 种类别
- **视频检测**：支持视频文件检测，标注检测结果并可下载标注视频
- **视频追踪模式**：使用 YOLO 官方追踪 API，支持跨帧物体追踪和唯一物体计数
- **批量图片检测**：支持多张图片批量处理，提升检测效率
- **实时通信**：WebSocket 支持实时摄像头检测
- **Web 界面**：提供友好的可视化操作界面，支持 PWA，可自定义检测类别和模型
- **健康检查**：服务状态监控接口
- **多设备支持**：支持 GPU 加速和 CPU 推理
- **类别过滤**：支持按类别筛选检测结果
- **内存管理**：智能内存管理，根据系统内存动态调整批处理大小

## 技术栈

- Python 3.8+
- FastAPI - 高性能 Web 框架
- Ultralytics YOLOv8 - 目标检测模型
- OpenCV - 图像/视频处理
- WebSocket - 实时通信
- imageio - 视频编解码
- PyTorch - 深度学习框架
- psutil - 系统监控

## 快速开始

### 环境准备

确保已安装 Python 3.8 或更高版本，并安装以下依赖：

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后，访问 `http://localhost:8000` 即可使用 Web 界面。

## API 文档

### 健康检查

```
GET /health
```

检查服务运行状态，返回模型信息、设备信息和性能统计。

**响应示例：**
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

### 图片检测

```
POST /api/v1/detect
Content-Type: multipart/form-data

参数：
- file: 图片文件 (JPG, PNG, BMP)
- classes: 要检测的类别，逗号分隔，例如 'person' 或 'person,car'
- conf_threshold: 置信度阈值 (0.1-0.9)，默认 0.5

返回：检测结果 JSON
```

**响应示例：**
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

### 视频检测

```
POST /api/v1/video
Content-Type: multipart/form-data

参数：
- file: 视频文件 (MP4, AVI, MOV)
- return_video: 是否返回处理后的视频文件 ("true" 或 "false", 默认 "true")
- classes: 要检测的类别，逗号分隔，例如 'person' 或 'person,car'
- use_batch_processing: 是否使用批处理优化 (默认 true)
- batch_size: 批处理大小 (1-32，默认 8)
- frame_interval: 帧处理间隔 (1=每帧处理，2=隔帧处理，默认 1)

返回：检测结果 JSON 或处理后的视频文件
```

**响应示例 (JSON)：**
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

### 视频检测（追踪模式）

使用 YOLO 官方追踪 API，支持跨帧物体追踪：

```python
result = detector.process_video_file_track(
    video_path,
    output_path,
    classes=['person'],
    conf=0.5
)
```

**返回值包含：**
- `class_counts`: 各类别的出现次数
- `frames`: 每帧的检测结果

### 批量图片检测

```
POST /api/v1/batch/detect
Content-Type: multipart/form-data

参数：
- image_files: 图片文件列表
- classes: 要检测的类别，逗号分隔
- max_workers: 最大工作线程数 (1-10)
- batch_size: 批处理大小 (1-100)

返回：批量检测结果
```

**响应示例：**
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

### WebSocket 实时检测

```
WS /ws/detect
```

支持实时图像帧检测，适用于摄像头实时检测场景。

**消息格式：**
- 输入：`{"image": "data:image/jpeg;base64,..."}`
- 输出：检测结果 JSON

## Web 界面

服务启动后提供可视化操作界面，支持：

- **图片检测模式**：上传图片并查看检测结果
- **摄像头模式**：实时检测摄像头画面中的物体
- **视频模式**：上传视频文件，处理后下载标注视频
- **实时统计**：显示检测物体数量、推理时间等信息
- **类别选择**：可选择检测特定类别的物体
- **置信度调节**：可调整检测置信度阈值
- **PWA 支持**：可添加到主屏幕作为独立应用使用

## 配置选项

通过环境变量和配置文件调整服务行为：

### 环境变量

- `YOLO_MODEL`：指定使用的 YOLO 模型，默认为 `yolov8n`
  - 示例：`export YOLO_MODEL=yolov8s` 使用 small 模型
  - 示例：`export YOLO_MODEL=yolov8l` 使用 large 模型

### 配置文件 (app/core/config.py)

```python
BATCH_PROCESSING = {
    "default_max_workers": 4,
    "default_batch_size": 10,
    "max_batch_size": 100,
    "max_workers": 10,
    "memory_threshold": 80,  # 内存使用阈值百分比
}
```

## COCO 数据集类别

支持 80 种常见物体类别：

| ID | 类别 | ID | 类别 |
|----|------|----|------|
| 0 | person | 1 | bicycle |
| 2 | car | 3 | motorcycle |
| 4 | airplane | 5 | bus |
| 6 | train | 7 | truck |
| 8 | boat | 15 | cat |
| 16 | dog | 39 | bottle |
| ... | ... | 79 | toothbrush |

完整类别列表见 `COCO_CLASSES` 字典。

## 模型说明

项目默认使用 `yolov8n.pt` 预训练模型，可根据需要替换为其他 YOLOv8 模型以平衡速度和精度：

- `yolov8n.pt`: nano 版本，最快但精度最低
- `yolov8s.pt`: small 版本，速度与精度平衡
- `yolov8m.pt`: medium 版本，较慢但精度更高
- `yolov8l.pt`: large 版本，较慢但精度很高
- `yolov8x.pt`: extra-large 版本，最慢但精度最高

首次启动时，如果本地没有模型文件，系统会自动下载指定的模型。

## 项目结构

```
├── app/                   # 应用代码
│   ├── api/               # API 路由
│   │   └── routes.py      # 路由定义
│   ├── core/              # 配置模块
│   │   └── config.py      # 配置文件
│   ├── models/            # 检测模型
│   │   └── detector.py    # 物体检测器
│   ├── utils/             # 工具模块
│   │   └── batch_processor.py  # 批处理器
│   └── main.py            # 应用入口
├── static/                # 前端静态资源
│   ├── index.html         # 主页面
│   └── manifest.json      # PWA 配置
├── test_api.py            # API 测试
├── requirements.txt       # 依赖列表
├── README.md              # 项目说明
├── LICENSE                # 许可证
├── cert.pem               # SSL 证书 (可选)
└── key.pem                # SSL 密钥 (可选)
```

## 部署指南

### 生产环境部署

使用 Uvicorn 部署生产环境：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

或者使用 Nginx + Uvicorn 组合部署。

### HTTPS 部署

若需要支持摄像头功能在生产环境中正常工作（特别是 HTTPS 环境下），可使用 SSL 证书：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 443 --ssl-keyfile ./key.pem --ssl-certfile ./cert.pem
```

## 性能优化建议

- **GPU 加速**：在支持 CUDA 的机器上运行，推理速度提升 10-50 倍
- **模型选择**：根据场景选择合适的模型，实时场景推荐 yolov8n/yolov8s
- **批处理**：批量检测时使用批处理优化，提升吞吐量
- **视频帧间隔**：适当增加 frame_interval 可减少处理时间
- **内存管理**：系统自动监控内存使用，动态调整批处理大小

## 故障排除

1. **模型下载失败**：检查网络连接，或手动下载模型文件至项目根目录
   - 下载地址：https://github.com/ultralytics/assets/releases
2. **摄像头权限问题**：确保浏览器已授权访问摄像头，HTTPS 环境下才能使用摄像头
3. **视频处理失败**：检查视频格式支持情况及磁盘空间
4. **内存不足**：降低批处理大小或增加内存阈值配置
5. **WebSocket 连接失败**：检查防火墙设置和代理配置

## 许可证

本项目遵循开源许可证协议，详情请见 [LICENSE](./LICENSE) 文件。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进项目。
