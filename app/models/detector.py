import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Optional, Union
import time
from pathlib import Path
import os
import imageio
import psutil
import gc

# 默认模型，可通过环境变量覆盖
DEFAULT_MODEL = os.getenv('YOLO_MODEL', 'yolov8n')

# COCO 数据集的 80 个类别
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
    38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork',
    43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
    48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
    53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch',
    58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv',
    63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
    68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush'
}

# 类别名称反转映射，用于从名称查找类别 ID
COCO_CLASSES_REVERSE = {v: k for k, v in COCO_CLASSES.items()}


class MemoryManager:
    """内存管理器，监控和控制内存使用"""
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent

    def check_memory_usage(self) -> bool:
        """检查当前内存使用率"""
        memory_percent = psutil.virtual_memory().percent
        return memory_percent <= self.max_memory_percent

    def estimate_image_memory(self, image_shape) -> int:
        """估算单张图像内存需求 (bytes)"""
        if len(image_shape) == 3:
            height, width, channels = image_shape
        else:
            height, width = image_shape
            channels = 1
        return height * width * channels * 4

    def cleanup_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def adjust_batch_size_based_on_memory(self, base_batch_size: int,
                                        estimated_per_item_memory: int) -> int:
        """根据可用内存调整批处理大小"""
        available_memory = psutil.virtual_memory().available
        max_items_by_memory = available_memory // estimated_per_item_memory
        return min(base_batch_size, max(1, max_items_by_memory // 2))


class ObjectsDetector:
    """通用物体检测器，支持 COCO 数据集的 80 个类别"""

    def __init__(self):
        self.model = None
        self.device = None
        self.model_loaded = False
        self.memory_manager = MemoryManager()

    def _fourcc_to_str(self, fourcc):
        """将 fourcc 编码转换为字符串"""
        try:
            return "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
        except:
            return str(fourcc)

    def load_model(self):
        """加载 YOLO 模型"""
        model_name = DEFAULT_MODEL
        model_file = f"{model_name}.pt"

        print(f"使用模型：{model_name}")

        model_path = Path(model_file)

        if model_path.exists():
            print(f"加载本地模型：{model_path.absolute()}")
            self.model = YOLO(str(model_path))
        else:
            current_dir = Path(__file__).parent.parent.parent
            model_path = current_dir / model_file

            if model_path.exists():
                print(f"加载本地模型：{model_path.absolute()}")
                self.model = YOLO(str(model_path))
            else:
                print(f"本地模型不存在，尝试下载...")
                print(f"正在下载 {model_file}，请稍候...")
                try:
                    self.model = YOLO(model_file)
                except Exception as e:
                    print(f"模型下载失败：{e}")
                    print(f"请手动下载模型文件放到项目根目录:")
                    print(f"   https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_file}")
                    raise

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备：{self.device}")
        self.model.to(self.device)
        self.model_loaded = True
        print("模型加载完成")

    def _parse_classes(self, classes: Optional[Union[List[int], List[str]]]) -> Optional[List[int]]:
        """
        解析类别参数，将类别名称转换为类别 ID
        :param classes: 类别列表，可以是 ID 或名称
        :return: 类别 ID 列表
        """
        if classes is None:
            return None

        class_ids = []
        for c in classes:
            if isinstance(c, str):
                if c.lower() in COCO_CLASSES_REVERSE:
                    class_ids.append(COCO_CLASSES_REVERSE[c.lower()])
                else:
                    print(f"警告：未知类别名称 '{c}'，已忽略")
            elif isinstance(c, int) and c in COCO_CLASSES:
                class_ids.append(c)
            else:
                print(f"警告：未知类别 ID {c}，已忽略")

        return class_ids if class_ids else None

    def detect_objects(
        self,
        image: np.ndarray,
        return_annotated: bool = False,
        classes: Optional[Union[List[int], List[str]]] = None,
        conf_threshold: float = 0.5
    ) -> Dict:
        """
        检测图像中的物体
        :param image: 输入图像 (BGR 格式)
        :param return_annotated: 是否返回标注后的图像
        :param classes: 要检测的类别列表，可以是类别 ID 或类别名称
                       如果为 None，则检测所有 80 个类别
                       例如：[0] 或 ['person'] 只检测人，['car', 'person'] 检测车和人
        :param conf_threshold: 置信度阈值，默认 0.5
        :return: 检测结果字典
        """
        if not self.model_loaded:
            raise Exception("Model not loaded. Please load the model first.")

        start_time = time.time()

        class_ids = self._parse_classes(classes)

        predict_kwargs = {
            'conf': conf_threshold,
            'device': self.device,
            'verbose': False
        }
        if class_ids is not None:
            predict_kwargs['classes'] = class_ids

        results = self.model.predict(image, **predict_kwargs)
        inference_time = time.time() - start_time

        objects = []
        result = results[0]

        annotated_image = None
        if return_annotated:
            annotated_image = result.plot() if len(result.boxes) > 0 else image.copy()

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = COCO_CLASSES.get(class_id, f'unknown_{class_id}')

                objects.append(
                    {
                        "bbox": {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "width": int(x2 - x1),
                            "height": int(y2 - y1)
                        },
                        "confidence": round(confidence, 3),
                        "class_id": class_id,
                        "class_name": class_name
                    }
                )

        return {
            "success": True,
            "object_count": len(objects),
            "objects": objects,
            "inference_time_ms": round(inference_time * 1000, 2),
            "image_shape": {
                "height": image.shape[0],
                "width": image.shape[1]
            },
            "annotated_image": annotated_image
        }

    def batch_detect_objects(
        self,
        images: List[np.ndarray],
        return_annotated: bool = False,
        classes: Optional[Union[List[int], List[str]]] = None,
        conf_threshold: float = 0.5
    ) -> List[Dict]:
        """
        批量检测多张图像中的物体
        :param images: 图像列表
        :param return_annotated: 是否返回标注后的图像
        :param classes: 要检测的类别列表
        :param conf_threshold: 置信度阈值
        :return: 检测结果列表
        """
        if not self.model_loaded:
            raise Exception("Model not loaded. Please load the model first.")

        if not images:
            return []

        results = []
        for image in images:
            result = self.detect_objects(image, return_annotated, classes, conf_threshold)
            results.append(result)

        return results

    def batch_predict_optimized(
        self,
        images: List[np.ndarray],
        return_annotated: bool = False,
        classes: Optional[Union[List[int], List[str]]] = None,
        conf_threshold: float = 0.5
    ) -> List[Dict]:
        """
        使用优化的批量预测方法检测多张图像
        :param images: 图像列表
        :param return_annotated: 是否返回标注后的图像
        :param classes: 要检测的类别列表
        :param conf_threshold: 置信度阈值
        :return: 检测结果列表
        """
        if not self.model_loaded:
            raise Exception("Model not loaded. Please load the model first.")

        if not images:
            return []

        start_time = time.time()

        class_ids = self._parse_classes(classes)

        predict_kwargs = {
            'conf': conf_threshold,
            'device': self.device,
            'verbose': False
        }
        if class_ids is not None:
            predict_kwargs['classes'] = class_ids

        batch_results = self.model.predict(images, **predict_kwargs)

        results = []
        for i, result in enumerate(batch_results):
            objects = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = COCO_CLASSES.get(class_id, f'unknown_{class_id}')

                    objects.append(
                        {
                            "bbox": {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                                "width": int(x2 - x1),
                                "height": int(y2 - y1)
                            },
                            "confidence": round(confidence, 3),
                            "class_id": class_id,
                            "class_name": class_name
                        }
                    )

            annotated_image = None
            if return_annotated:
                annotated_image = result.plot() if len(result.boxes) > 0 else images[i].copy()

            inference_time_ms = round((time.time() - start_time) * 1000 / len(images), 2)

            results.append({
                "success": True,
                "object_count": len(objects),
                "objects": objects,
                "inference_time_ms": inference_time_ms,
                "image_shape": {
                    "height": images[i].shape[0],
                    "width": images[i].shape[1]
                },
                "annotated_image": annotated_image
            })

        return results

    def detect_video_frame(self, frame: np.ndarray, **kwargs) -> Dict:
        """
        检测视频单帧，返回带标注的图片和结果
        """
        result = self.detect_objects(frame, return_annotated=True, **kwargs)
        return result

    def process_video_frames_batch(
        self,
        video_path: str,
        frame_interval: int = 1,
        output_dir: Optional[str] = None,
        classes: Optional[Union[List[int], List[str]]] = None
    ) -> Dict:
        """批量处理视频帧"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"success": False, "error": f"Cannot open video: {video_path}"}

        frame_count = 0
        frames_to_process = []
        frame_timestamps = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    frames_to_process.append(frame)
                    frame_timestamps.append(frame_count / cap.get(cv2.CAP_PROP_FPS))

                frame_count += 1

            cap.release()

            if not frames_to_process:
                return {"success": False, "error": "No frames extracted from video"}

            results = self.batch_predict_optimized(frames_to_process, return_annotated=True, classes=classes)

            for i, (result, timestamp) in enumerate(zip(results, frame_timestamps)):
                result["frame_number"] = i * frame_interval
                result["timestamp"] = timestamp

                if output_dir and result.get("annotated_image") is not None:
                    from datetime import datetime
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = Path(output_dir) / f"frame_{i:04d}_{timestamp_str}.jpg"
                    cv2.imwrite(str(output_path), result["annotated_image"])
                    result["output_path"] = str(output_path)

            return {
                "success": True,
                "total_frames_processed": len(results),
                "results": results
            }

        except Exception as e:
            cap.release()
            return {"success": False, "error": str(e)}

    def process_video_file(
        self,
        video_path: str,
        output_path: str = None,
        classes: Optional[Union[List[int], List[str]]] = None,
        use_batch_processing: bool = True,
        batch_size: int = 8,
        frame_interval: int = 1
    ) -> Dict:
        """
        处理视频文件，逐帧检测

        Args:
            video_path: 视频文件路径
            output_path: 输出文件路径（可选）
            classes: 要检测的类别
            use_batch_processing: 是否使用批处理优化
            batch_size: 批处理大小
            frame_interval: 帧处理间隔（1=每帧处理，2=隔帧处理）
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件：{video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        original_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

        print(f"处理视频：{width}x{height} @ {fps}fps, 总帧数：{total_frames}")
        print(f"原视频编码：{original_fourcc} ({self._fourcc_to_str(original_fourcc)})")
        print(f"output_path: {output_path}")
        print(f"批处理：{use_batch_processing}, batch_size: {batch_size}, frame_interval: {frame_interval}")

        writer = None
        if output_path:
            print(f"使用 imageio 创建视频写入器...")
            try:
                writer = imageio.get_writer(
                    output_path,
                    fps=fps,
                    codec='libx264'
                )
                print(f"imageio 写入器创建成功")
            except Exception as e:
                print(f"无法创建视频写入器：{e}")
                import traceback
                traceback.print_exc()
                writer = None

        frame_results = []
        frame_count = 0
        written_frames = 0

        # 批处理相关变量
        frame_buffer = []
        frame_indices = []

        try:
            if use_batch_processing:
                # 批处理模式
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % frame_interval == 0:
                        frame_buffer.append(frame)
                        frame_indices.append(frame_count)

                        # 当缓冲区满或到达视频末尾时进行处理
                        if len(frame_buffer) >= batch_size:
                            batch_results = self._process_video_frame_batch(
                                frame_buffer,
                                classes=classes,
                                return_annotated=(writer is not None)
                            )

                            for idx, (result, original_frame) in enumerate(zip(batch_results, frame_buffer)):
                                frame_num = frame_indices[idx]

                                if writer and result.get("annotated_image") is not None:
                                    annotated_frame = result["annotated_image"]
                                    if annotated_frame.shape[1] != width or annotated_frame.shape[0] != height:
                                        annotated_frame = cv2.resize(annotated_frame, (width, height))
                                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                                    writer.append_data(rgb_frame)
                                    written_frames += 1

                                if result["object_count"] > 0:
                                    frame_results.append({
                                        "frame": frame_num,
                                        "timestamp": round(frame_num / fps, 2),
                                        "object_count": result["object_count"],
                                        "objects": result["objects"]
                                    })

                            # 清空缓冲区
                            frame_buffer.clear()
                            frame_indices.clear()

                            if written_frames % 30 == 0:
                                print(f"已写入 {written_frames} 帧")

                    frame_count += 1

                    if frame_count % 30 == 0:
                        progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                        print(f"处理进度：{frame_count}/{total_frames} ({progress:.1f}%)")

                # 处理剩余帧
                if frame_buffer:
                    batch_results = self._process_video_frame_batch(
                        frame_buffer,
                        classes=classes,
                        return_annotated=(writer is not None)
                    )

                    for idx, (result, original_frame) in enumerate(zip(batch_results, frame_buffer)):
                        frame_num = frame_indices[idx]

                        if writer and result.get("annotated_image") is not None:
                            annotated_frame = result["annotated_image"]
                            if annotated_frame.shape[1] != width or annotated_frame.shape[0] != height:
                                annotated_frame = cv2.resize(annotated_frame, (width, height))
                            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            writer.append_data(rgb_frame)
                            written_frames += 1

                        if result["object_count"] > 0:
                            frame_results.append({
                                "frame": frame_num,
                                "timestamp": round(frame_num / fps, 2),
                                "object_count": result["object_count"],
                                "objects": result["objects"]
                            })

                    frame_buffer.clear()
                    frame_indices.clear()

            else:
                # 单帧处理模式（向后兼容）
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    result = self.detect_objects(frame, return_annotated=True, classes=classes)

                    if writer:
                        try:
                            if result.get("annotated_image") is not None:
                                annotated_frame = result["annotated_image"]
                            else:
                                annotated_frame = frame

                            if annotated_frame.shape[1] != width or annotated_frame.shape[0] != height:
                                annotated_frame = cv2.resize(annotated_frame, (width, height))

                            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            writer.append_data(rgb_frame)
                            written_frames += 1

                            if written_frames % 30 == 0:
                                print(f"已写入 {written_frames} 帧，当前帧尺寸：{rgb_frame.shape}")

                        except Exception as e:
                            print(f"写入帧 {frame_count} 失败：{e}")
                            import traceback
                            traceback.print_exc()

                    if result["object_count"] > 0:
                        frame_results.append({
                            "frame": frame_count,
                            "timestamp": round(frame_count / fps, 2),
                            "object_count": result["object_count"],
                            "objects": result["objects"]
                        })

                    frame_count += 1

                    if frame_count % 30 == 0:
                        progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                        print(f"处理进度：{frame_count}/{total_frames} ({progress:.1f}%)")

        finally:
            cap.release()
            if writer:
                writer.close()
                print("视频写入器已关闭")

        if output_path:
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                print(f"视频处理完成：共处理 {frame_count} 帧，成功写入 {written_frames} 帧")
                print(f"输出文件：{output_path}")
                print(f"文件大小：{file_size:.2f} MB")

                try:
                    test_reader = imageio.get_reader(output_path)
                    test_frame = test_reader.get_data(0)
                    test_reader.close()
                    print(f"视频文件可正常读取，第一帧尺寸：{test_frame.shape}")
                except Exception as e:
                    print(f"视频文件无法读取：{e}")
            else:
                print(f"错误：输出文件不存在!")
        else:
            print(f"视频分析完成：共处理 {frame_count} 帧")

        return {
            "total_frames": total_frames,
            "processed_frames": frame_count,
            "fps": fps,
            "duration": round(frame_count / fps, 2) if fps > 0 else 0,
            "resolution": {"width": width, "height": height},
            "frames_with_detection": len(frame_results),
            "frames": frame_results,
            "batch_processing_used": use_batch_processing
        }
    def process_video_file_track(
            self,
            video_path: str,
            output_path: str = None,
            classes: Optional[Union[List[int], List[str]]] = None,
            use_batch_processing: bool = True,
            batch_size: int = 8,
            frame_interval: int = 1,
            conf: Optional[float] = 0.5,
        ) -> Dict:
        """
        处理视频文件（跟踪模式）

        Args:
            video_path: 视频文件路径
            output_path: 输出文件路径
            classes: 要检测的类别
            use_batch_processing: 是否使用批处理优化
            batch_size: 批处理大小
            frame_interval: 帧处理间隔

        Returns:
            处理结果
        """

        try:
            # 使用opencv获取视频信息
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

            print(f"处理视频：{width}x{height} @ {fps}fps, 总帧数：{total_frames}")
            print(f"原视频编码：{original_fourcc} ({self._fourcc_to_str(original_fourcc)})")
            print(f"output_path: {output_path}")
            print(f"批处理：{use_batch_processing}, batch_size: {batch_size}, frame_interval: {frame_interval}")

            # 创建视频写入器
            writer = None
            if output_path:
                print(f"使用 imageio 创建视频写入器...")
                try:
                    writer = imageio.get_writer(
                        output_path,
                        fps=fps,
                        codec='libx264'
                    )
                    print(f"imageio 写入器创建成功")
                except Exception as e:
                    print(f"无法创建视频写入器：{e}")
                    import traceback
                    traceback.print_exc()
                    writer = None

            # classes转为id
            class_ids = self._parse_classes(classes)
            
            # 处理视频文件(流处理)
            results = self.model.track(
                source=video_path,  # 视频文件路径
                persist=True,       # 是否保存跟踪结果
                conf=conf,          # 置信度
                classes=class_ids,  # 要检测的类别
                stream=True,         # 是否流处理
            )
    
            # 结果帧列表
            frame_results = []

            # 帧计数器
            frame_count = 0
            # dict字典存出现过的类别和数量
            class_counts = {}

            # set集合存出现的id
            class_ids = set()
            
            for result in results:

                objects = []
                
                # 根据id判断当前的物品是不是重复的
                for box in result.boxes:

                    # 获取框信息
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = COCO_CLASSES.get(class_id, f'unknown_{class_id}')

                    # 添加id
                    if class_id not in class_ids:
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        class_ids.add(class_id)

                    objects.append(
                        {
                            "bbox": {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                                "width": int(x2 - x1),
                                "height": int(y2 - y1)
                            },
                            "confidence": round(confidence, 3),
                            "class_id": class_id,
                            "class_name": class_name
                        }
                    )
            
                # 获取图像
                written_frame = 0
                if writer:
                    frame_result = result.plot(
                        conf=True,      # 显示置信度
                        labels=True,    # 显示标签
                        boxes=True,     # 显示框
                        probs=True,     # 显示概率
                        line_width=2,   # 线宽
                        font_size=12,   # 字体大小
                    )
                    try:
                        if frame_result is not None:
                            if frame_result.shape[1] != width or frame_result.shape[0] != height:
                                frame_result = cv2.resize(frame_result, (width, height))

                            rgb_frame = cv2.cvtColor(frame_result, cv2.COLOR_BGR2RGB)
                            writer.append_data(rgb_frame)
                            written_frame += 1

                            if written_frame % 30 == 0:
                                print(f"已写入 {written_frame} 帧, 当前帧尺寸: {rgb_frame.shape}")
                        
                    except Exception as e:
                        print(f"写入帧 {frame_count} 失败：{e}")
                        import traceback
                        traceback.print_exc()
                frame_results.append({
                    "frame": frame_count,
                    "timestamp": round(frame_count / fps, 2),
                    "object_count": len(objects),
                    "objects": objects
                })
                
                # 帧计数器更新
                frame_count += 1

        finally:
            cap.release()
            if writer: 
                writer.close()
                print("视频写入器已关闭")
            
        if output_path:
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                print(f"视频处理完成：共处理 {frame_count} 帧，成功写入 {written_frame} 帧")
                print(f"输出文件：{output_path}")
                print(f"文件大小：{file_size:.2f} MB")

                try:
                    test_reader = imageio.get_reader(output_path)
                    test_frame = test_reader.get_data(0)
                    test_reader.close()
                    print(f"视频文件可正常读取，第一帧尺寸：{test_frame.shape}")
                except Exception as e:
                    print(f"视频文件无法读取：{e}")
            else:
                print(f"错误：输出文件不存在!")
        else:
            print(f"视频分析完成：共处理 {frame_count} 帧")

        return {
            "total_frames": total_frames,
            "processed_frames": frame_count,
            "fps": fps,
            "duration": round(frame_count / fps, 2) if fps > 0 else 0,
            "resolution": {"width": width, "height": height},
            "frames_with_detection": len(frame_results),
            "frames": frame_results,
            "batch_processing_used": use_batch_processing,
            "class_counts": class_counts
        }

    def _process_video_frame_batch(
        self,
        frames: List[np.ndarray],
        classes: Optional[Union[List[int], List[str]]] = None,
        return_annotated: bool = True
    ) -> List[Dict]:
        """
        批量处理视频帧

        Args:
            frames: 视频帧列表
            classes: 要检测的类别
            return_annotated: 是否返回标注图像

        Returns:
            检测结果列表
        """
        if not frames:
            return []

        if not self.model_loaded:
            raise Exception("Model not loaded. Please load the model first.")

        class_ids = self._parse_classes(classes)

        predict_kwargs = {
            'conf': 0.5,
            'device': self.device,
            'verbose': False
        }
        if class_ids is not None:
            predict_kwargs['classes'] = class_ids

        # 使用 YOLO 的批量预测功能
        batch_results = self.model.predict(frames, **predict_kwargs)

        results = []
        for i, result in enumerate(batch_results):
            objects = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = COCO_CLASSES.get(class_id, f'unknown_{class_id}')

                    objects.append(
                        {
                            "bbox": {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                                "width": int(x2 - x1),
                                "height": int(y2 - y1)
                            },
                            "confidence": round(confidence, 3),
                            "class_id": class_id,
                            "class_name": class_name
                        }
                    )

            annotated_image = None
            if return_annotated:
                annotated_image = result.plot() if len(result.boxes) > 0 else frames[i].copy()

            results.append({
                "success": True,
                "object_count": len(objects),
                "objects": objects,
                "image_shape": {
                    "height": frames[i].shape[0],
                    "width": frames[i].shape[1]
                },
                "annotated_image": annotated_image
            })

        return results


# 全局实例
detector = ObjectsDetector()
