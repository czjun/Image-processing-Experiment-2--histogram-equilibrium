#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# 使用标准字体而非中文字体，避免字体问题
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

def histogram_equalization(img):
    """经典直方图均衡化（HE）"""
    # 获取图像尺寸
    height, width = img.shape
    total_pixels = height * width
    
    # 计算图像直方图
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    
    # 计算概率密度函数 (PDF)
    pdf = hist / total_pixels
    
    # 计算累积分布函数 (CDF)
    cdf = np.cumsum(pdf)
    
    # 创建灰度映射表
    # 映射函数: s_k = r_0 + (r_{L-1} - r_0) * CDF(r_k)
    mapping = np.round(255 * cdf).astype(np.uint8)
    
    # 应用映射表到图像
    equalized_img = mapping[img]
    
    return equalized_img

def brightness_preserving_bi_histogram_equalization(img):
    """亮度保持双子直方图均衡化（BBHE）"""
    # 计算图像平均灰度值 (阈值)
    threshold = np.uint8(np.mean(img))
    
    # 将图像分割为两个子图像
    lower_img = img[img <= threshold]  # 低于或等于阈值的部分
    upper_img = img[img > threshold]   # 高于阈值的部分
    
    # 如果某个子图像为空，返回原始图像
    if len(lower_img) == 0 or len(upper_img) == 0:
        return img
    
    # 分别计算两个子图像的直方图
    lower_hist = np.bincount(lower_img.flatten(), minlength=threshold+1)
    upper_hist = np.bincount(upper_img.flatten() - threshold - 1, minlength=256-threshold-1)
    
    # 计算概率密度函数 (PDF)
    lower_pdf = lower_hist / len(lower_img)
    upper_pdf = upper_hist / len(upper_img)
    
    # 计算累积分布函数 (CDF)
    lower_cdf = np.cumsum(lower_pdf)
    upper_cdf = np.cumsum(upper_pdf)
    
    # 创建灰度映射表
    # 对于下半部分: [0, threshold]
    lower_mapping = np.round(threshold * lower_cdf).astype(np.uint8)
    
    # 对于上半部分: [threshold+1, 255]
    upper_mapping = np.round((255 - threshold - 1) * upper_cdf).astype(np.uint8) + threshold + 1
    
    # 创建输出图像
    bbhe_img = np.zeros_like(img)
    
    # 应用映射
    bbhe_img[img <= threshold] = lower_mapping[img[img <= threshold]]
    bbhe_img[img > threshold] = upper_mapping[img[img > threshold] - threshold - 1]
    
    return bbhe_img

def plot_histogram(hist, title, ax=None):
    """绘制直方图"""
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(hist)
    ax.set_title(title)
    ax.set_xlim([0, 256])
    ax.grid(True)
    return ax

def create_side_by_side_image(image, hist, title1, title2, filename):
    """创建并保存图像和直方图并排显示的图片"""
    # 创建图像对象和画布
    fig = Figure(figsize=(12, 5))
    canvas = FigureCanvas(fig)
    
    # 添加子图
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    # 在左侧显示图像
    ax1.imshow(image, cmap='gray')
    ax1.set_title(title1)
    ax1.axis('off')
    
    # 在右侧显示直方图
    ax2.plot(hist)
    ax2.set_title(title2)
    ax2.set_xlim([0, 256])
    ax2.grid(True)
    
    # 调整布局
    fig.tight_layout()
    
    # 保存图像
    canvas.draw()
    img_array = np.array(canvas.renderer.buffer_rgba())
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(filename, img_array)

def main():
    # 创建输出目录（确保目录存在）
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取测试图像并转为灰度图
    img = cv2.imread('couple.jpg', 0)
    if img is None:
        print("Error: Could not read image. Make sure 'couple.jpg' exists in the current directory.")
        return
    
    # 应用经典直方图均衡化 (HE)
    he_img = histogram_equalization(img)
    
    # 应用亮度保持双子直方图均衡化 (BBHE)
    bbhe_img = brightness_preserving_bi_histogram_equalization(img)
    
    # 计算原图、HE和BBHE的平均亮度
    original_mean = np.mean(img)
    he_mean = np.mean(he_img)
    bbhe_mean = np.mean(bbhe_img)
    
    # 打印亮度信息
    print("Brightness Analysis:")
    print(f"Original Image Mean: {original_mean:.2f}")
    print(f"HE Mean: {he_mean:.2f}")
    print(f"BBHE Mean: {bbhe_mean:.2f}")
    print(f"HE Shift: {abs(he_mean - original_mean):.2f}")
    print(f"BBHE Shift: {abs(bbhe_mean - original_mean):.2f}")
    
    # 使用OpenCV保存单独的图像
    cv2.imwrite(os.path.join(output_dir, "original.png"), img)
    cv2.imwrite(os.path.join(output_dir, "he_result.png"), he_img)
    cv2.imwrite(os.path.join(output_dir, "bbhe_result.png"), bbhe_img)
    
    # 计算直方图
    hist_orig = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    hist_he = cv2.calcHist([he_img], [0], None, [256], [0, 256]).flatten()
    hist_bbhe = cv2.calcHist([bbhe_img], [0], None, [256], [0, 256]).flatten()
    
    # 1. HE及其直方图
    create_side_by_side_image(
        he_img, 
        hist_he, 
        f'HE Result (Mean: {he_mean:.2f})', 
        'HE Histogram', 
        os.path.join(output_dir, "1_he_with_hist.png")
    )
    
    # 2. BBHE及其直方图
    create_side_by_side_image(
        bbhe_img, 
        hist_bbhe, 
        f'BBHE Result (Mean: {bbhe_mean:.2f})', 
        'BBHE Histogram', 
        os.path.join(output_dir, "2_bbhe_with_hist.png")
    )
    
    # 3. 原图及其直方图
    create_side_by_side_image(
        img, 
        hist_orig, 
        f'Original (Mean: {original_mean:.2f})', 
        'Original Histogram', 
        os.path.join(output_dir, "3_original_with_hist.png")
    )
    
    # 4. 图像对比图 - 创建一个三列的图像
    fig = Figure(figsize=(15, 5))
    canvas = FigureCanvas(fig)
    
    ax1 = fig.add_subplot(131)
    ax1.imshow(img, cmap='gray')
    ax1.set_title(f'Original (Mean: {original_mean:.2f})')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(132)
    ax2.imshow(he_img, cmap='gray')
    ax2.set_title(f'HE (Mean: {he_mean:.2f})')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(133)
    ax3.imshow(bbhe_img, cmap='gray')
    ax3.set_title(f'BBHE (Mean: {bbhe_mean:.2f})')
    ax3.axis('off')
    
    fig.tight_layout()
    canvas.draw()
    comparison_img = np.array(canvas.renderer.buffer_rgba())
    comparison_img = cv2.cvtColor(comparison_img, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(os.path.join(output_dir, "4_image_comparison.png"), comparison_img)
    
    # 5. 直方图对比图
    fig = Figure(figsize=(10, 6))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    
    ax.plot(hist_orig, 'k', label='Original')
    ax.plot(hist_he, 'b', label='HE')
    ax.plot(hist_bbhe, 'r', label='BBHE')
    ax.set_title('Histogram Comparison')
    ax.set_xlim([0, 256])
    ax.grid(True)
    ax.legend()
    
    fig.tight_layout()
    canvas.draw()
    hist_comparison_img = np.array(canvas.renderer.buffer_rgba())
    hist_comparison_img = cv2.cvtColor(hist_comparison_img, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(os.path.join(output_dir, "5_histogram_comparison.png"), hist_comparison_img)
    
    print(f"\n所有结果已保存到 '{output_dir}' 目录")
    print("输出图片：")
    print("1. HE及其直方图: 1_he_with_hist.png")
    print("2. BBHE及其直方图: 2_bbhe_with_hist.png")
    print("3. 原始图像及其直方图: 3_original_with_hist.png")
    print("4. 图像对比图: 4_image_comparison.png")
    print("5. 图像对比直方图: 5_histogram_comparison.png")
    
if __name__ == "__main__":
    main() 