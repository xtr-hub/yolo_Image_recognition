from fastapi import APIRouter, File, UploadFile, HTTPException, Query
import cv2
import numpy as np
import base64
from typing import List, Optional

from app.models.detector import detector, COCO_CLASSES
from app.core.config import BATCH_PROCESSING

router = APIRouter()


@router.post("/detect")
async def detect(
    file: UploadFile = File(...),
    classes: Optional[str] = Query(None, description="要检测的类别，逗号分隔，例如 'person' 或 'person,car'"),
    conf_threshold: float = Query(0.5, ge=0.1, le=0.9, description="置信度阈值")
):
    """
    单张图片物体检测
    - file: 图片文件
    - classes: 要检测的类别，逗号分隔，例如 'person' 或 'person,car'。不传则检测所有 80 个类别
    - conf_threshold: 置信度阈值，默认 0.5
    """
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 解析类别参数
    class_list = None
    if classes:
        class_list = [c.strip() for c in classes.split(',')]

    result = detector.detect_objects(image, return_annotated=True, classes=class_list, conf_threshold=conf_threshold)

    if result.get("annotated_image") is not None:
        _, buffer = cv2.imencode('.jpg', result["annotated_image"])
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')
        result["annotated_image"] = f"data:image/jpeg;base64,{annotated_b64}"

    return result


@router.post("/video")
async def detect_video(
    file: UploadFile = File(...),
    return_video: str = "true",
    classes: Optional[str] = Query(None, description="要检测的类别，逗号分隔"),
    use_batch_processing: bool = Query(True, description="是否使用批处理优化"),
    batch_size: int = Query(8, ge=1, le=32, description="批处理大小"),
    frame_interval: int = Query(1, ge=1, le=5, description="帧处理间隔")
):
    """
    上传视频文件进行检测
    - file: 视频文件
    - return_video: 是否返回处理后的视频文件 ("true" 或 "false")
    - classes: 要检测的类别，逗号分隔，例如 'person' 或 'person,car'
    - use_batch_processing: 是否使用批处理优化
    - batch_size: 批处理大小 (1-32)
    - frame_interval: 帧处理间隔 (1=每帧处理，2=隔帧处理)
    """
    import tempfile
    import os
    import asyncio

    should_return_video = return_video.lower() == "true"
    print(f"接收到的 return_video 参数：{return_video}")
    print(f"转换后的布尔值：{should_return_video}")
    print(f"批处理参数：use_batch_processing={use_batch_processing}, batch_size={batch_size}, frame_interval={frame_interval}")

    input_path = None
    output_path = None

    # 解析类别参数
    class_list = None
    if classes:
        class_list = [c.strip() for c in classes.split(',')]

    try:
        # 保存上传的视频到临时文件
        suffix = os.path.splitext(file.filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            input_path = tmp.name

        # 处理视频
        print(f"should_return_video={should_return_video}, 准备设置 output_path")
        if should_return_video:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(os.path.dirname(input_path), f"{base_name}_output.mp4")
            print(f"output_path 已设置：{output_path}")
        else:
            print(f"不返回视频，output_path 保持为 None")

        result = detector.process_video_file_track(
            input_path,
            output_path,
            classes=class_list,
            use_batch_processing=use_batch_processing,
            batch_size=batch_size,
            frame_interval=frame_interval
        )

        if should_return_video and output_path and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"准备返回视频：{output_path}, 大小：{file_size / (1024*1024):.2f} MB")

            if file_size < 1024:
                print(f"视频文件太小，可能生成失败")
                raise HTTPException(status_code=500, detail="视频文件生成失败，文件大小异常")

            from fastapi.responses import FileResponse

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

        return {
            "success": True,
            "filename": file.filename,
            **result
        }

    except Exception as e:
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
        raise HTTPException(status_code=500, detail=f"视频处理失败：{str(e)}")


@router.get("/health")
async def health_check():
    from app.utils.batch_processor import get_batch_processor, BatchConfig
    from app.core.config import BATCH_PROCESSING

    # 获取批处理器性能统计
    try:
        batch_processor = get_batch_processor(detector, BatchConfig(
            batch_size=BATCH_PROCESSING["default_batch_size"],
            max_workers=BATCH_PROCESSING["default_max_workers"],
            memory_threshold=BATCH_PROCESSING["memory_threshold"]
        ))
        perf_stats = batch_processor.get_performance_stats()
    except Exception as e:
        perf_stats = {"error": str(e)}

    return {
        "status": "healthy",
        "model_loaded": detector.model_loaded,
        "device": detector.device if detector.model_loaded else None,
        "batch_processing_enabled": True,
        "batch_processing_config": BATCH_PROCESSING,
        "supported_classes": list(COCO_CLASSES.values()),
        "performance_stats": perf_stats
    }


@router.post("/batch/detect")
async def batch_detect(
    image_files: List[UploadFile] = File(...),
    classes: Optional[str] = Query(None, description="要检测的类别，逗号分隔"),
    max_workers: int = Query(BATCH_PROCESSING["default_max_workers"], ge=1, le=BATCH_PROCESSING["max_workers"]),
    batch_size: int = Query(BATCH_PROCESSING["default_batch_size"], ge=1, le=BATCH_PROCESSING["max_batch_size"])
):
    """
    批量图片检测
    - image_files: 图片文件列表
    - classes: 要检测的类别，逗号分隔，例如 'person' 或 'person,car'
    - max_workers: 最大工作线程数
    - batch_size: 批处理大小
    """
    if len(image_files) == 0:
        raise HTTPException(status_code=400, detail="At least one image file is required")

    import tempfile

    # 解析类别参数
    class_list = None
    if classes:
        class_list = [c.strip() for c in classes.split(',')]

    with tempfile.TemporaryDirectory() as temp_dir:
        saved_paths = []

        for file in image_files:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_paths.append(file_path)

        # 读取所有图像
        images = []
        for path in saved_paths:
            image = cv2.imread(path)
            if image is not None:
                images.append(image)

        if not images:
            raise HTTPException(status_code=400, detail="No valid images found")

        try:
            results = detector.batch_predict_optimized(images, return_annotated=True, classes=class_list)

            successful_results = []
            failed_count = len(saved_paths) - len(images)

            for result, original_path in zip(results, saved_paths[:len(results)]):
                result["input_path"] = original_path
                successful_results.append(result)

            return {
                "success": True,
                "total_processed": len(successful_results),
                "failed_count": failed_count,
                "results": successful_results
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@router.post("/batch/detect-with-progress")
async def batch_detect_with_progress(
    image_files: List[UploadFile] = File(...),
    classes: Optional[str] = Query(None, description="要检测的类别，逗号分隔"),
    max_workers: int = Query(BATCH_PROCESSING["default_max_workers"], ge=1, le=BATCH_PROCESSING["max_workers"])
):
    """
    带进度反馈的批量检测
    - image_files: 图片文件列表
    - classes: 要检测的类别，逗号分隔
    - max_workers: 最大工作线程数
    """
    import tempfile
    import uuid

    task_id = str(uuid.uuid4())

    # 解析类别参数
    class_list = None
    if classes:
        class_list = [c.strip() for c in classes.split(',')]

    with tempfile.TemporaryDirectory() as temp_dir:
        saved_paths = []

        for file in image_files:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_paths.append(file_path)

        images = []
        for path in saved_paths:
            image = cv2.imread(path)
            if image is not None:
                images.append(image)

        if not images:
            raise HTTPException(status_code=400, detail="No valid images found")

        try:
            results = detector.batch_predict_optimized(images, return_annotated=True, classes=class_list)

            successful_results = []
            failed_count = len(saved_paths) - len(images)

            for result, original_path in zip(results, saved_paths[:len(results)]):
                result["input_path"] = original_path
                successful_results.append(result)

            return {
                "task_id": task_id,
                "status": "completed",
                "total_processed": len(successful_results),
                "failed_count": failed_count,
                "results": successful_results
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
