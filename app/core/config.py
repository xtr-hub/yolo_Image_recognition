"""
应用配置文件
包含批处理和其他功能的相关配置参数
"""

# 批处理相关配置
BATCH_PROCESSING = {
    "default_max_workers": 4,
    "default_batch_size": 10,
    "max_batch_size": 100,
    "max_workers": 10,
    "chunk_size": 5,
    "memory_threshold": 80,  # 内存使用阈值百分比
    "frame_interval_default": 1  # 视频帧处理间隔
}

# 模型相关配置
MODEL_CONFIG = {
    "default_model": "yolov8s",
    "confidence_threshold": 0.5,
    "classes": [0]  # 仅检测人
}

# 服务器相关配置
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True
}