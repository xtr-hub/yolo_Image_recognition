from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.routes import router
from app.models.detector import detector
import os
import cv2
import numpy as np

# 创建应用
app = FastAPI()

# 启动时加载模型
@app.on_event("startup")
async def startup_event():
    detector.load_model()

# 注册路由
app.include_router(router, prefix="/api/v1")

# WebSocket 实时视频检测
@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket 连接已建立")

    try:
        while True:
            # 接收前端发送的二进制图片
            data = await websocket.receive_bytes()

            try:
                # 直接解码二进制图片
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

                if frame is not None:
                    # 检测
                    result = detector.detect_video_frame(frame)
                    if result.get("annotated_image") is not None:
                        _, buffer = cv2.imencode('.jpg', result["annotated_image"])
                        await websocket.send_bytes(bytes(buffer))

                else:
                    await websocket.send_json({
                        "success": False,
                        "error": "Image decode failed"
                    })

            except Exception as e:
                await websocket.send_json({
                    "success": False,
                    "error": str(e)
                })

    except WebSocketDisconnect:
        print("WebSocket 连接已断开")

# 静态文件服务
static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

# 根路径返回前端页面
@app.get("/")
async def root():
    return FileResponse(os.path.join(static_path, "index.html"))
