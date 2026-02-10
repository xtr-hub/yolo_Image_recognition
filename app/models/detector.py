import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict
import time
from pathlib import Path
import os
import imageio

# 默认模型，可通过环境变量覆盖
DEFAULT_MODEL = os.getenv('YOLO_MODEL', 'yolov8s')

class PersonDetector:
    # 初始化
    def __init__(self):
        self.model = None # 模型
        self.device = None # 设备
        self.model_loaded = False # 模型是否加载
    
    def _fourcc_to_str(self, fourcc):
        """将 fourcc 编码转换为字符串"""
        try:
            return "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
        except:
            return str(fourcc)
    
    def load_model(self):
        # 模型加载逻辑
        model_name = DEFAULT_MODEL
        model_file = f"{model_name}.pt"
        
        print(f"使用模型: {model_name}")
        
        # 优先使用本地模型
        model_path = Path(model_file)
        
        if model_path.exists():
            print(f"加载本地模型: {model_path.absolute()}")
            self.model = YOLO(str(model_path))
        else:
            # 尝试从当前目录查找
            current_dir = Path(__file__).parent.parent.parent
            model_path = current_dir / model_file
            
            if model_path.exists():
                print(f"加载本地模型: {model_path.absolute()}")
                self.model = YOLO(str(model_path))
            else:
                print(f"本地模型不存在，尝试下载...")
                print(f"正在下载 {model_file}，请稍候...")
                try:
                    self.model = YOLO(model_file)
                except Exception as e:
                    print(f"模型下载失败: {e}")
                    print(f"请手动下载模型文件放到项目根目录:")
                    print(f"   https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_file}")
                    raise
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        self.model.to(self.device) # 模型加载到设备
        self.model_loaded = True # 模型是否加载完成
        print("模型加载完成")
    # 检测逻辑
    def detect_persons(self, image, return_annotated=False):
        # 检测模型是否已加载
        if not self.model_loaded:
            raise Exception("Model not loaded. Please load the model first.")
        
        start_time = time.time()
        results = self.model.predict(
            image,
            classes=[0],
            conf=0.5,
            device=self.device,
            verbose=False # 不打印日志
        )
        inference_time = time.time() - start_time
        
        # 解析结果
        persons = []
        result = results[0]
        
        # 生成标注图片
        annotated_image = None
        if return_annotated:
            # 总是生成标注图片，即使没有检测到目标
            annotated_image = result.plot() if len(result.boxes) > 0 else image.copy()
        
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                persons.append(
                    {
                        "bbox":{
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "width": int(x2 - x1),
                            "height": int(y2 - y1)
                        },
                        "confidence": round(confidence, 3)
                    }
                )
        
        return {
            "success": True,
            "person_count": len(persons),
            "persons": persons,
            "inference_time_ms": round(inference_time * 1000, 2),
            "image_shape": {
                "height": image.shape[0],
                "width": image.shape[1]
            },
            "annotated_image": annotated_image
        }
    
    def detect_video_frame(self, frame: np.ndarray) -> Dict:
        """
        检测视频单帧，返回带标注的图片和结果
        """
        result = self.detect_persons(frame, return_annotated=True)
        return result
    
    def process_video_file(self, video_path: str, output_path: str = None) -> Dict:
        """
        处理视频文件，逐帧检测
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # 默认 30 FPS
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 获取原视频的编码格式
        original_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        print(f"处理视频: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")
        print(f"原视频编码: {original_fourcc} ({self._fourcc_to_str(original_fourcc)})")
        print(f"output_path: {output_path}")  # 调试日志
        
        # 创建输出视频
        writer = None
        if output_path:
            # 使用 imageio 创建视频写入器（更稳定可靠）
            print(f"使用 imageio 创建视频写入器...")
            try:
                writer = imageio.get_writer(
                    output_path,
                    fps=fps,
                    codec='libx264'
                )
                print(f"imageio 写入器创建成功")
            except Exception as e:
                print(f"无法创建视频写入器: {e}")
                import traceback
                traceback.print_exc()
                writer = None
        
        frame_results = []
        frame_count = 0
        written_frames = 0  # 记录成功写入的帧数
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检测当前帧
                result = self.detect_persons(frame, return_annotated=True)
                
                # 每一帧都写入输出视频（重要！）
                if writer:
                    try:
                        if result.get("annotated_image") is not None:
                            annotated_frame = result["annotated_image"]
                        else:
                            # 如果没有检测结果，使用原始帧
                            annotated_frame = frame
                        
                        # 确保帧尺寸匹配
                        if annotated_frame.shape[1] != width or annotated_frame.shape[0] != height:
                            annotated_frame = cv2.resize(annotated_frame, (width, height))
                        
                        # 转换 BGR 为 RGB（imageio 需要 RGB）
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        writer.append_data(rgb_frame)
                        written_frames += 1
                        
                        # 每 30 帧确认一次写入成功
                        if written_frames % 30 == 0:
                            print(f"已写入 {written_frames} 帧，当前帧尺寸: {rgb_frame.shape}")
                            
                    except Exception as e:
                        print(f"写入帧 {frame_count} 失败: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 只保存有检测结果的帧到 JSON（减少数据量）
                if result["person_count"] > 0:
                    frame_results.append({
                        "frame": frame_count,
                        "timestamp": round(frame_count / fps, 2),
                        "person_count": result["person_count"],
                        "persons": result["persons"]
                    })
                
                frame_count += 1
                
                # 每 30 帧打印一次进度
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    print(f"⏳ 处理进度: {frame_count}/{total_frames} ({progress:.1f}%)")
        
        finally:
            cap.release()
            if writer:
                writer.close()
                print("视频写入器已关闭")
        
        # 检查输出文件
        if output_path:
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"视频处理完成: 共处理 {frame_count} 帧, 成功写入 {written_frames} 帧")
                print(f"输出文件: {output_path}")
                print(f"文件大小: {file_size:.2f} MB")
                
                # 检查文件是否可以被读取
                try:
                    test_reader = imageio.get_reader(output_path)
                    test_frame = test_reader.get_data(0)
                    test_reader.close()
                    print(f"视频文件可正常读取，第一帧尺寸: {test_frame.shape}")
                except Exception as e:
                    print(f"视频文件无法读取: {e}")
            else:
                print(f"错误: 输出文件不存在!")
        else:
            print(f"视频分析完成: 共处理 {frame_count} 帧")
        
        return {
            "total_frames": total_frames,
            "processed_frames": frame_count,
            "fps": fps,
            "duration": round(frame_count / fps, 2) if fps > 0 else 0,
            "resolution": {"width": width, "height": height},
            "frames_with_detection": len(frame_results),
            "frames": frame_results
        }

# 全局实例
detector = PersonDetector()