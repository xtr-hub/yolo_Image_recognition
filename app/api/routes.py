from fastapi import APIRouter, File, UploadFile, HTTPException
import cv2
import numpy as np

from app.models.detector import detector

router = APIRouter()

@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    # 读取上传的图片文件
    contents = await file.read()

    # 转 OpenCV 格式
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None: 
        raise HTTPException(status_code=400, detail="Invalid image file")
    # 检测
    result = detector.detect_persons(image)

    return result

@router.post("/video")
async def detect_video(
    file: UploadFile = File(...), 
    return_video: str = "true"  # 默认改为 true，确保默认生成视频
):
    """
    上传视频文件进行检测
    - return_video: 是否返回处理后的视频文件 ("true" 或 "false")
    """
    import tempfile
    import os
    import asyncio
    
    # 将字符串转为布尔值
    should_return_video = return_video.lower() == "true"
    print(f"接收到的 return_video 参数: {return_video}")
    print(f"转换后的布尔值: {should_return_video}")
    
    input_path = None
    output_path = None
    
    try:
        # 保存上传的视频到临时文件
        suffix = os.path.splitext(file.filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            input_path = tmp.name
        
        # 处理视频，输出为 MP4 格式（imageio 最兼容）
        print(f"should_return_video={should_return_video}, 准备设置 output_path")
        if should_return_video:
            # 强制使用 .mp4 扩展名
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(os.path.dirname(input_path), f"{base_name}_output.mp4")
            print(f"output_path 已设置: {output_path}")
        else:
            print(f"不返回视频，output_path 保持为 None")
        
        result = detector.process_video_file(input_path, output_path)
        
        if should_return_video and output_path and os.path.exists(output_path):
            # 验证视频文件
            file_size = os.path.getsize(output_path)
            print(f"准备返回视频: {output_path}, 大小: {file_size / (1024*1024):.2f} MB")
            
            if file_size < 1024:  # 小于 1KB 可能是空文件
                print(f"视频文件太小，可能生成失败")
                raise HTTPException(status_code=500, detail="视频文件生成失败，文件大小异常")
            
            # 返回视频文件
            from fastapi.responses import FileResponse
            
            # 延迟删除文件
            async def cleanup_files():
                await asyncio.sleep(2)
                try:
                    if input_path and os.path.exists(input_path):
                        os.unlink(input_path)
                    if output_path and os.path.exists(output_path):
                        os.unlink(output_path)
                except:
                    pass
            
            asyncio.create_task(cleanup_files())
            
            return FileResponse(
                output_path,
                media_type="video/mp4",
                filename=f"detected_{os.path.splitext(file.filename)[0]}.mp4"
            )
        
        # 清理输入文件
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)
        
        # 返回 JSON 结果
        return {
            "success": True,
            "filename": file.filename,
            **result
        }
        
    except Exception as e:
        # 出错时清理文件
        if input_path and os.path.exists(input_path):
            try:
                os.unlink(input_path)
            except:
                pass
        if output_path and os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except:
                pass
        
        import traceback
        error_detail = traceback.format_exc()
        print(f"视频处理错误:\n{error_detail}")
        raise HTTPException(status_code=500, detail=f"视频处理失败: {str(e)}")


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",  # 服务状态
        "model_loaded": detector.model_loaded,  # 模型是否加载
        "device": detector.device if detector.model_loaded else None  # cuda 或 cpu
    }
