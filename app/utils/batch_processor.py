"""
批处理器模块

提供高效的批量图像检测功能，支持：
- 内存管理和监控
- 并发控制
- 分块处理策略
- 自适应系统资源管理
- 错误处理和恢复
"""

import cv2
import numpy as np
import torch
import psutil
import gc
import time
from typing import List, Dict, Optional, Union, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import traceback


@dataclass
class BatchConfig:
    """批处理配置"""
    batch_size: int = 10
    max_workers: int = 4
    memory_threshold: float = 80.0
    chunk_size: int = 5
    enable_gpu_batch: bool = True
    gpu_batch_size: int = 8


@dataclass
class ProcessingProgress:
    """处理进度信息"""
    total: int
    processed: int
    failed: int
    progress_percent: float
    elapsed_time: float
    estimated_remaining: float


class MemoryManager:
    """内存管理器，监控和控制内存使用"""

    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self._baseline_memory = psutil.Process().memory_info().rss

    def get_current_memory_percent(self) -> float:
        """获取当前内存使用百分比"""
        return psutil.virtual_memory().percent

    def get_process_memory_mb(self) -> float:
        """获取当前进程内存使用 (MB)"""
        return psutil.Process().memory_info().rss / (1024 * 1024)

    def is_memory_available(self) -> bool:
        """检查是否有可用内存"""
        return self.get_current_memory_percent() <= self.max_memory_percent

    def estimate_image_memory(self, image_shape: tuple) -> int:
        """估算单张图像内存需求 (bytes)"""
        if len(image_shape) == 3:
            height, width, channels = image_shape
        else:
            height, width = image_shape
            channels = 1
        return height * width * channels * 4

    def calculate_safe_batch_size(
        self,
        image_shapes: List[tuple],
        base_batch_size: int
    ) -> int:
        """
        根据可用内存计算安全的批处理大小

        Args:
            image_shapes: 图像形状列表
            base_batch_size: 基础批处理大小

        Returns:
            调整后的批处理大小
        """
        if not image_shapes:
            return base_batch_size

        # 估算平均图像内存
        avg_memory = sum(
            self.estimate_image_memory(shape) for shape in image_shapes
        ) / len(image_shapes)

        # 计算可用内存
        available_memory = psutil.virtual_memory().available
        max_items_by_memory = int(available_memory / avg_memory)

        # 留出 50% 的安全余量
        safe_batch_size = max(1, max_items_by_memory // 2)

        return min(base_batch_size, safe_batch_size)

    def cleanup_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class BatchProcessor:
    """
    批处理器

    提供高效的批量图像检测功能，支持内存管理、并发控制和错误处理
    """

    def __init__(self, detector, config: Optional[BatchConfig] = None):
        """
        初始化批处理器

        Args:
            detector: 检测器实例
            config: 批处理配置
        """
        self.detector = detector
        self.config = config or BatchConfig()
        self.memory_manager = MemoryManager(self.config.memory_threshold)
        self._executor = None

    def _get_executor(self) -> ThreadPoolExecutor:
        """获取或创建线程池执行器"""
        if self._executor is None or self._executor._shutdown:
            self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        return self._executor

    def shutdown(self):
        """关闭批处理器"""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _process_chunk(
        self,
        images: List[np.ndarray],
        classes: Optional[List[int]] = None,
        conf_threshold: float = 0.5,
        return_annotated: bool = False
    ) -> List[Dict]:
        """
        处理图像块

        Args:
            images: 图像列表
            classes: 要检测的类别
            conf_threshold: 置信度阈值
            return_annotated: 是否返回标注图像

        Returns:
            检测结果列表
        """
        if self.detector.model_loaded and self.config.enable_gpu_batch:
            # 使用 YOLO 的批量预测功能
            return self.detector.batch_predict_optimized(
                images,
                return_annotated=return_annotated,
                classes=classes,
                conf_threshold=conf_threshold
            )
        else:
            # 逐张处理
            results = []
            for image in images:
                result = self.detector.detect_objects(
                    image,
                    return_annotated=return_annotated,
                    classes=classes,
                    conf_threshold=conf_threshold
                )
                results.append(result)
            return results

    def process_batch(
        self,
        images: List[np.ndarray],
        classes: Optional[Union[List[int], List[str]]] = None,
        conf_threshold: float = 0.5,
        return_annotated: bool = False,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ) -> List[Dict]:
        """
        批量处理图像

        Args:
            images: 图像列表
            classes: 要检测的类别
            conf_threshold: 置信度阈值
            return_annotated: 是否返回标注图像
            progress_callback: 进度回调函数

        Returns:
            检测结果列表
        """
        if not images:
            return []

        start_time = time.time()
        total_images = len(images)

        # 解析类别
        class_ids = self.detector._parse_classes(classes)

        # 计算安全的批处理大小
        image_shapes = [img.shape for img in images]
        safe_batch_size = self.memory_manager.calculate_safe_batch_size(
            image_shapes,
            self.config.batch_size
        )

        print(f"批处理：共 {total_images} 张图像，批处理大小：{safe_batch_size}")

        all_results = []
        processed_count = 0
        failed_count = 0

        # 分块处理
        for i in range(0, total_images, safe_batch_size):
            chunk = images[i:i + safe_batch_size]

            try:
                # 检查内存状态
                if not self.memory_manager.is_memory_available():
                    print("内存使用过高，等待清理...")
                    self.memory_manager.cleanup_memory()
                    time.sleep(0.5)

                # 处理当前块
                chunk_results = self._process_chunk(
                    chunk,
                    classes=class_ids,
                    conf_threshold=conf_threshold,
                    return_annotated=return_annotated
                )

                all_results.extend(chunk_results)
                processed_count += len(chunk)

            except Exception as e:
                print(f"处理块 {i // safe_batch_size} 失败：{e}")
                traceback.print_exc()
                failed_count += len(chunk)

                # 为失败的块逐张处理
                for image in chunk:
                    try:
                        result = self.detector.detect_objects(
                            image,
                            return_annotated=return_annotated,
                            classes=classes,
                            conf_threshold=conf_threshold
                        )
                        all_results.append(result)
                        processed_count += 1
                        failed_count -= 1
                    except Exception as inner_e:
                        print(f"逐张处理也失败：{inner_e}")
                        # 返回空结果
                        all_results.append({
                            "success": False,
                            "error": str(inner_e),
                            "object_count": 0,
                            "objects": [],
                            "image_shape": {
                                "height": image.shape[0],
                                "width": image.shape[1]
                            }
                        })

            # 更新进度
            if progress_callback:
                elapsed = time.time() - start_time
                avg_time_per_image = elapsed / processed_count if processed_count > 0 else 0
                remaining = total_images - processed_count
                estimated_remaining = avg_time_per_image * remaining

                progress = ProcessingProgress(
                    total=total_images,
                    processed=processed_count,
                    failed=failed_count,
                    progress_percent=(processed_count / total_images) * 100,
                    elapsed_time=elapsed,
                    estimated_remaining=estimated_remaining
                )
                progress_callback(progress)

        # 清理内存
        self.memory_manager.cleanup_memory()

        return all_results

    def process_batch_with_chunks(
        self,
        images: List[np.ndarray],
        classes: Optional[Union[List[int], List[str]]] = None,
        conf_threshold: float = 0.5,
        return_annotated: bool = False,
        use_parallel: bool = False
    ) -> List[Dict]:
        """
        使用分块策略批量处理图像

        Args:
            images: 图像列表
            classes: 要检测的类别
            conf_threshold: 置信度阈值
            return_annotated: 是否返回标注图像
            use_parallel: 是否使用并行处理

        Returns:
            检测结果列表
        """
        if not images:
            return []

        total_images = len(images)
        chunk_size = self.config.chunk_size

        # 计算安全的批处理大小
        image_shapes = [img.shape for img in images]
        safe_batch_size = self.memory_manager.calculate_safe_batch_size(
            image_shapes,
            self.config.batch_size
        )

        # 使用较小的块大小
        effective_chunk_size = min(chunk_size, safe_batch_size)

        print(f"分块批处理：共 {total_images} 张图像，块大小：{effective_chunk_size}")

        if use_parallel and total_images > effective_chunk_size * 2:
            # 并行处理多个块
            return self._process_parallel_chunks(
                images,
                classes=classes,
                conf_threshold=conf_threshold,
                return_annotated=return_annotated,
                chunk_size=effective_chunk_size
            )
        else:
            # 串行处理
            return self.process_batch(
                images,
                classes=classes,
                conf_threshold=conf_threshold,
                return_annotated=return_annotated
            )

    def _process_parallel_chunks(
        self,
        images: List[np.ndarray],
        classes: Optional[List[int]] = None,
        conf_threshold: float = 0.5,
        return_annotated: bool = False,
        chunk_size: int = 5
    ) -> List[Dict]:
        """并行处理多个图像块"""

        # 将图像分块
        chunks = [
            images[i:i + chunk_size]
            for i in range(0, len(images), chunk_size)
        ]

        executor = self._get_executor()
        futures = {
            executor.submit(
                self._process_chunk,
                chunk,
                classes=classes,
                conf_threshold=conf_threshold,
                return_annotated=return_annotated
            ): idx
            for idx, chunk in enumerate(chunks)
        }

        # 按原始顺序收集结果
        results_by_index = {}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                chunk_results = future.result()
                results_by_index[idx] = chunk_results
            except Exception as e:
                print(f"处理块 {idx} 失败：{e}")
                # 返回空结果
                results_by_index[idx] = [
                    {"success": False, "error": str(e), "object_count": 0, "objects": []}
                    for _ in range(len(chunks[idx]))
                ]

        # 按顺序合并结果
        all_results = []
        for idx in sorted(results_by_index.keys()):
            all_results.extend(results_by_index[idx])

        return all_results

    def process_video_frames(
        self,
        frames: List[np.ndarray],
        classes: Optional[Union[List[int], List[str]]] = None,
        conf_threshold: float = 0.5,
        return_annotated: bool = True,
        frame_interval: int = 1
    ) -> List[Dict]:
        """
        批量处理视频帧

        Args:
            frames: 视频帧列表
            classes: 要检测的类别
            conf_threshold: 置信度阈值
            return_annotated: 是否返回标注图像
            frame_interval: 帧处理间隔

        Returns:
            检测结果列表，包含帧信息
        """
        # 根据间隔采样帧
        sampled_frames = frames[::frame_interval]

        if not sampled_frames:
            return []

        # 批量处理
        results = self.process_batch(
            sampled_frames,
            classes=classes,
            conf_threshold=conf_threshold,
            return_annotated=return_annotated
        )

        # 添加帧信息
        for i, result in enumerate(results):
            result["frame_index"] = i * frame_interval
            result["timestamp"] = i * frame_interval / 30.0  # 假设 30fps

        return results

    def get_performance_stats(self) -> Dict:
        """获取性能统计信息"""
        return {
            "memory_usage_mb": self.memory_manager.get_process_memory_mb(),
            "memory_percent": self.memory_manager.get_current_memory_percent(),
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_allocated_gb": (
                torch.cuda.memory_allocated() / (1024 ** 3)
                if torch.cuda.is_available() else 0
            ),
            "config": {
                "batch_size": self.config.batch_size,
                "max_workers": self.config.max_workers,
                "memory_threshold": self.config.memory_threshold
            }
        }


# 全局批处理器实例（延迟初始化）
_batch_processor: Optional[BatchProcessor] = None


def get_batch_processor(detector, config: Optional[BatchConfig] = None) -> BatchProcessor:
    """获取或创建全局批处理器实例"""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor(detector, config)
    return _batch_processor


def shutdown_batch_processor():
    """关闭全局批处理器"""
    global _batch_processor
    if _batch_processor:
        _batch_processor.shutdown()
        _batch_processor = None
