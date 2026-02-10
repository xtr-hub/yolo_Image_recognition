from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.routes import router
from app.models.detector import detector
import os
import base64
import cv2
import numpy as np
import json

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
            # 接收前端发送的 base64 图片
            data = await websocket.receive_text()
            
            try:
                # 解析 JSON
                message = json.loads(data)
                image_data = message.get("image", "")
                
                # 解码 base64 图片
                if "," in image_data:
                    image_data = image_data.split(",")[1]
                
                img_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # 检测
                    result = detector.detect_video_frame(frame)
                    
                    # 编码标注后的图片为 base64
                    if result.get("annotated_image") is not None:
                        _, buffer = cv2.imencode('.jpg', result["annotated_image"])
                        annotated_b64 = base64.b64encode(buffer).decode('utf-8')
                        result["annotated_image"] = f"data:image/jpeg;base64,{annotated_b64}"
                    
                    # 发送结果
                    await websocket.send_json({
                        "success": True,
                        "person_count": result["person_count"],
                        "persons": result["persons"],
                        "annotated_image": result.get("annotated_image", ""),
                        "inference_time_ms": result["inference_time_ms"]
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
