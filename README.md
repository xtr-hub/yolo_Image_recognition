

# YOLO 图像识别服务

基于 YOLOv8 的人物检测 Web 服务，提供图片和视频中的人物检测功能。

## 功能特点

- **图片检测**：上传图片即可快速识别图中人物
- **视频检测**：支持视频文件检测，标注检测结果
- **实时通信**：WebSocket 支持实时检测结果推送
- **Web 界面**：提供友好的可视化操作界面
- **健康检查**：服务状态监控接口

## 技术栈

- Python 3.8+
- FastAPI - 高性能 Web 框架
- Ultralytics YOLOv8 - 目标检测模型
- OpenCV - 图像/视频处理
- WebSocket - 实时通信

## 快速开始

### 环境准备

确保已安装 Python 3.8 或更高版本。

### 安装依赖

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

检查服务运行状态。

### 图片检测

```
POST /detect
Content-Type: multipart/form-data

参数：
- file: 图片文件

返回：检测结果 JSON
```

### 视频检测

```
POST /video
Content-Type: multipart/form-data

参数：
- file: 视频文件
- return_video: 是否返回标注视频 (true/false, 默认 true)

返回：检测结果 JSON
```

### WebSocket 实时检测

```
WS /ws/detect
```

支持实时图像帧检测。

## Web 界面

服务启动后提供可视化操作界面，支持：

- 图片上传与预览
- 视频上传与进度显示
- 检测结果可视化展示
- 统计数据汇总

## 模型说明

项目使用 `yolov8s.pt` 预训练模型，可根据需要替换为其他 YOLOv8 模型以平衡速度和精度。

## 项目结构

```
├── app/
│   ├── api/           # API 路由
│   ├── core/          # 配置模块
│   ├── models/        # 检测模型
│   └── main.py        # 应用入口
├── static/            # 前端静态资源
├── test_api.py        # API 测试
├── requirements.txt   # 依赖列表
└── yolov8s.pt         # YOLOv8 模型
```

## 许可证

本项目遵循开源协议，具体许可信息请参阅项目仓库。