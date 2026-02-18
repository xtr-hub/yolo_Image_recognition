"""
批处理性能测试脚本
用于测试和验证批处理优化的效果
"""
import time
import asyncio
import cv2
import numpy as np
from PIL import Image
import requests
import os
from concurrent.futures import ThreadPoolExecutor
import tempfile

# 生成测试图像
def generate_test_image(width=640, height=480):
    """生成用于测试的图像"""
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    # 添加一些模拟的"人"形状
    cv2.rectangle(image, (100, 100), (200, 300), (0, 255, 0), -1)
    cv2.rectangle(image, (300, 150), (400, 350), (255, 0, 0), -1)
    return image

def test_single_image_detection():
    """测试单张图像检测性能"""
    from app.models.detector import detector

    if not detector.model_loaded:
        detector.load_model()

    test_image = generate_test_image()

    start_time = time.time()
    result = detector.detect_persons(test_image, return_annotated=True)
    single_time = time.time() - start_time

    print(f"单张图像检测时间: {single_time:.3f}秒")
    print(f"检测到 {result['person_count']} 个人")
    return single_time

def test_batch_detection():
    """测试批量图像检测性能"""
    from app.models.detector import detector

    if not detector.model_loaded:
        detector.load_model()

    # 生成测试图像批次
    batch_size = 10
    test_images = [generate_test_image() for _ in range(batch_size)]

    start_time = time.time()
    results = detector.batch_predict_optimized(test_images, return_annotated=True)
    batch_time = time.time() - start_time

    total_people = sum(r['person_count'] for r in results)
    avg_time_per_image = batch_time / batch_size

    print(f"批量检测({batch_size}张图像)总时间: {batch_time:.3f}秒")
    print(f"平均每张图像时间: {avg_time_per_image:.3f}秒")
    print(f"总共检测到 {total_people} 个人")
    return batch_time, avg_time_per_image

def test_comparison():
    """比较单张检测和批量检测的性能"""
    print("="*50)
    print("批处理性能对比测试")
    print("="*50)

    # 测试单张检测
    print("\n1. 单张图像检测:")
    single_times = []
    for i in range(10):
        single_time = test_single_image_detection()
        single_times.append(single_time)
        if i < 9:  # 不在最后一次打印
            print(f"  运行 {i+1}/10")

    avg_single_time = sum(single_times) / len(single_times)
    total_single_time = sum(single_times)

    print(f"\n单张检测平均时间: {avg_single_time:.3f}秒")
    print(f"10张图像总检测时间: {total_single_time:.3f}秒")

    # 测试批量检测
    print("\n2. 批量图像检测:")
    batch_time, avg_batch_time = test_batch_detection()

    print(f"\n批量检测10张图像总时间: {batch_time:.3f}秒")

    # 计算性能提升
    speedup = total_single_time / batch_time if batch_time > 0 else 0
    print(f"\n性能提升: {speedup:.2f}x")
    print(f"时间节省: {(total_single_time - batch_time):.3f}秒 ({((total_single_time - batch_time) / total_single_time) * 100:.1f}%)")

    return speedup

def test_api_endpoints():
    """测试API端点功能"""
    print("\n" + "="*50)
    print("API端点功能测试")
    print("="*50)

    # 生成测试图像
    test_images = [generate_test_image() for _ in range(5)]

    # 保存到临时文件
    with tempfile.TemporaryDirectory() as temp_dir:
        image_paths = []
        for i, img in enumerate(test_images):
            path = os.path.join(temp_dir, f"test_{i}.jpg")
            cv2.imwrite(path, img)
            image_paths.append(path)

        print(f"生成了 {len(image_paths)} 张测试图像")

        # 这里可以添加对API的实际调用测试
        # 由于需要运行服务器，暂时跳过

if __name__ == "__main__":
    try:
        speedup = test_comparison()
        test_api_endpoints()

        print("\n" + "="*50)
        print("测试完成!")
        print(f"批处理优化性能提升: {speedup:.2f}x")
        print("="*50)

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()