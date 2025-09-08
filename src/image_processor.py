"""
图像数据分析模块
将3D点云投影为2D图像，并进行边缘检测和轮廓提取
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List
import logging
from config import IMAGE_CONFIG

class ImageProcessor:
    """图像处理器"""
    
    def __init__(self):
        self.output_size = IMAGE_CONFIG['output_size']
        self.gaussian_kernel = IMAGE_CONFIG['gaussian_blur_kernel']
        self.canny_low = IMAGE_CONFIG['canny_low_threshold']
        self.canny_high = IMAGE_CONFIG['canny_high_threshold']
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def project_to_2d(self, chamber_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        将3D燃烧室点云投影到XY平面并生成深度图像
        
        Args:
            chamber_points: 燃烧室点云 (N, 3)
            
        Returns:
            (深度图像, 灰度图像, 投影信息)
        """
        if len(chamber_points) == 0:
            empty_image = np.zeros(self.output_size, dtype=np.uint8)
            return empty_image, empty_image, {'error': '无点云数据'}
        
        self.logger.info(f"开始3D到2D投影，点云数量: {len(chamber_points)}")
        
        # 提取XY坐标和Z深度
        x_coords = chamber_points[:, 0]
        y_coords = chamber_points[:, 1]
        z_coords = chamber_points[:, 2]
        
        # 计算坐标范围
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        z_min, z_max = z_coords.min(), z_coords.max()
        
        # 防止除零错误
        if x_max == x_min:
            x_max = x_min + 1
        if y_max == y_min:
            y_max = y_min + 1
        if z_max == z_min:
            z_max = z_min + 1
        
        # 归一化坐标到图像尺寸
        x_normalized = ((x_coords - x_min) / (x_max - x_min) * (self.output_size[1] - 1)).astype(int)
        y_normalized = ((y_coords - y_min) / (y_max - y_min) * (self.output_size[0] - 1)).astype(int)
        
        # 创建深度图像
        depth_image = np.full(self.output_size, z_max, dtype=np.float64)  # 用最大深度初始化
        
        # 将点云映射到图像上，保留最小Z值（最深的点）
        for i in range(len(chamber_points)):
            row, col = y_normalized[i], x_normalized[i]
            if 0 <= row < self.output_size[0] and 0 <= col < self.output_size[1]:
                depth_image[row, col] = min(depth_image[row, col], z_coords[i])
        
        # 将深度图转换为8位灰度图像
        # 深度越大（Z值越大）越亮，深度越小（Z值越小）越暗
        depth_normalized = (depth_image - z_min) / (z_max - z_min)
        grayscale_image = (depth_normalized * 255).astype(np.uint8)
        
        # 对于没有点云数据的区域，设为中间灰度值
        no_data_mask = (depth_image == z_max)
        grayscale_image[no_data_mask] = 128  # 中间灰度
        
        projection_info = {
            'point_count': len(chamber_points),
            'x_range': [float(x_min), float(x_max)],
            'y_range': [float(y_min), float(y_max)],
            'z_range': [float(z_min), float(z_max)],
            'image_size': self.output_size,
            'pixel_density': len(chamber_points) / (self.output_size[0] * self.output_size[1])
        }
        
        self.logger.info(f"投影完成: {projection_info['image_size']}, 像素密度: {projection_info['pixel_density']:.4f}")
        
        return depth_image, grayscale_image, projection_info
    
    def edge_detection(self, grayscale_image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        对灰度图像进行边缘检测
        
        Args:
            grayscale_image: 输入灰度图像
            
        Returns:
            (边缘图像, 检测信息)
        """
        self.logger.info("开始边缘检测")
        
        # 高斯模糊降噪
        blurred = cv2.GaussianBlur(grayscale_image, self.gaussian_kernel, 0)
        
        # Canny边缘检测
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # 统计边缘像素
        edge_pixel_count = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        
        detection_info = {
            'edge_pixels': int(edge_pixel_count),
            'total_pixels': total_pixels,
            'edge_ratio': float(edge_pixel_count / total_pixels),
            'gaussian_kernel': self.gaussian_kernel,
            'canny_thresholds': [self.canny_low, self.canny_high]
        }
        
        self.logger.info(f"边缘检测完成: {detection_info['edge_pixels']}个边缘像素 ({detection_info['edge_ratio']*100:.2f}%)")
        
        return edges, detection_info
    
    def contour_extraction(self, edges: np.ndarray) -> Tuple[List, Dict]:
        """
        从边缘图像中提取轮廓
        
        Args:
            edges: 边缘检测结果
            
        Returns:
            (轮廓列表, 轮廓信息)
        """
        self.logger.info("开始轮廓提取")
        
        # 查找轮廓
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return [], {'error': '未检测到轮廓'}
        
        # 按面积排序轮廓
        contours_with_area = [(cv2.contourArea(contour), contour) for contour in contours]
        contours_with_area.sort(key=lambda x: x[0], reverse=True)
        
        # 分析主要轮廓
        largest_contour = contours_with_area[0][1] if contours_with_area else None
        
        contour_info = {
            'contour_count': len(contours),
            'largest_area': float(contours_with_area[0][0]) if contours_with_area else 0,
            'total_area': sum(area for area, _ in contours_with_area),
            'contour_areas': [float(area) for area, _ in contours_with_area[:5]]  # 前5大轮廓面积
        }
        
        # 分析最大轮廓的几何特性
        if largest_contour is not None and len(largest_contour) >= 5:
            try:
                # 拟合椭圆
                ellipse = cv2.fitEllipse(largest_contour)
                contour_info['ellipse'] = {
                    'center': [float(ellipse[0][0]), float(ellipse[0][1])],
                    'axes': [float(ellipse[1][0]), float(ellipse[1][1])],
                    'angle': float(ellipse[2])
                }
                
                # 最小外接圆
                (center_x, center_y), radius = cv2.minEnclosingCircle(largest_contour)
                contour_info['min_circle'] = {
                    'center': [float(center_x), float(center_y)],
                    'radius': float(radius)
                }
                
                # 轮廓周长
                perimeter = cv2.arcLength(largest_contour, True)
                contour_info['largest_perimeter'] = float(perimeter)
                
            except Exception as e:
                self.logger.warning(f"轮廓几何分析失败: {e}")
        
        self.logger.info(f"轮廓提取完成: {contour_info['contour_count']}个轮廓")
        if contour_info.get('largest_area', 0) > 0:
            self.logger.info(f"最大轮廓面积: {contour_info['largest_area']:.1f}")
        
        return contours, contour_info
    
    def create_result_visualization(self, original_image: np.ndarray, 
                                  edges: np.ndarray, 
                                  contours: List,
                                  save_path: Optional[str] = None) -> np.ndarray:
        """
        创建结果可视化图像
        
        Args:
            original_image: 原始灰度图像
            edges: 边缘检测结果
            contours: 轮廓列表
            save_path: 保存路径（可选）
            
        Returns:
            可视化结果图像
        """
        # 创建彩色图像用于可视化
        result_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        
        # 绘制轮廓
        if contours:
            # 绘制最大轮廓（红色）
            cv2.drawContours(result_image, [contours[0]], -1, (0, 0, 255), 2)
            
            # 绘制其他轮廓（绿色）
            if len(contours) > 1:
                cv2.drawContours(result_image, contours[1:], -1, (0, 255, 0), 1)
        
        # 如果提供了保存路径，保存图像
        if save_path:
            cv2.imwrite(save_path, result_image)
            self.logger.info(f"结果图像已保存: {save_path}")
        
        return result_image
    
    def complete_image_analysis(self, chamber_points: np.ndarray, 
                              output_dir: str = "output/") -> Dict:
        """
        完整的图像分析流程
        
        Args:
            chamber_points: 燃烧室点云数据
            output_dir: 输出目录
            
        Returns:
            完整分析结果字典
        """
        self.logger.info("开始完整图像分析流程")
        
        # 步骤1: 3D投影到2D
        depth_image, grayscale_image, projection_info = self.project_to_2d(chamber_points)
        
        # 步骤2: 边缘检测
        edges, edge_info = self.edge_detection(grayscale_image)
        
        # 步骤3: 轮廓提取
        contours, contour_info = self.contour_extraction(edges)
        
        # 步骤4: 创建可视化结果
        result_image = self.create_result_visualization(
            grayscale_image, edges, contours,
            save_path=f"{output_dir}/result_visualization.png"
        )
        
        # 保存中间结果
        cv2.imwrite(f"{output_dir}/grayscale_image.png", grayscale_image)
        cv2.imwrite(f"{output_dir}/edges.png", edges)
        
        # 整合所有结果
        complete_results = {
            'projection': projection_info,
            'edge_detection': edge_info,
            'contour_extraction': contour_info,
            'images': {
                'grayscale': grayscale_image,
                'depth': depth_image,
                'edges': edges,
                'result': result_image
            },
            'contours': contours
        }
        
        self.logger.info("图像分析流程完成")
        return complete_results
    
    def generate_analysis_report(self, results: Dict) -> str:
        """
        生成图像分析报告
        
        Args:
            results: 完整分析结果
            
        Returns:
            报告文本
        """
        report = []
        report.append("=== 图像数据分析报告 ===")
        report.append("")
        
        # 投影信息
        if 'projection' in results:
            proj = results['projection']
            report.append("1. 3D到2D投影:")
            report.append(f"   - 点云数量: {proj.get('point_count', 0)}")
            report.append(f"   - 图像尺寸: {proj.get('image_size', 'N/A')}")
            report.append(f"   - X范围: {proj.get('x_range', 'N/A')}")
            report.append(f"   - Y范围: {proj.get('y_range', 'N/A')}")
            report.append(f"   - Z范围: {proj.get('z_range', 'N/A')}")
            report.append(f"   - 像素密度: {proj.get('pixel_density', 0):.4f}")
            report.append("")
        
        # 边缘检测信息
        if 'edge_detection' in results:
            edge = results['edge_detection']
            report.append("2. 边缘检测:")
            report.append(f"   - 边缘像素数量: {edge.get('edge_pixels', 0)}")
            report.append(f"   - 边缘像素比例: {edge.get('edge_ratio', 0)*100:.2f}%")
            report.append(f"   - Canny阈值: {edge.get('canny_thresholds', 'N/A')}")
            report.append("")
        
        # 轮廓提取信息
        if 'contour_extraction' in results:
            contour = results['contour_extraction']
            report.append("3. 轮廓提取:")
            report.append(f"   - 轮廓数量: {contour.get('contour_count', 0)}")
            report.append(f"   - 最大轮廓面积: {contour.get('largest_area', 0):.1f}")
            
            if 'min_circle' in contour:
                circle = contour['min_circle']
                report.append(f"   - 最小外接圆半径: {circle['radius']:.2f}")
                report.append(f"   - 圆心坐标: ({circle['center'][0]:.1f}, {circle['center'][1]:.1f})")
            
            if 'ellipse' in contour:
                ellipse = contour['ellipse']
                report.append(f"   - 拟合椭圆长轴: {max(ellipse['axes']):.2f}")
                report.append(f"   - 拟合椭圆短轴: {min(ellipse['axes']):.2f}")
            
            report.append("")
        
        report.append("=== 分析完成 ===")
        
        return "\n".join(report)


if __name__ == "__main__":
    # 测试图像处理器
    processor = ImageProcessor()
    
    # 创建测试燃烧室点云数据
    np.random.seed(42)
    
    # 模拟圆形燃烧室
    test_points = []
    center_x, center_y = 25.0, -0.3
    radius = 10.0
    
    for i in range(1000):
        # 在圆内随机生成点
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0, radius)
        
        x = center_x + r * np.cos(angle)
        y = center_y + r * np.sin(angle)
        z = -0.05 * (1 - r/radius)**2  # 深度随半径变化
        
        test_points.append([x, y, z])
    
    test_points = np.array(test_points)
    
    # 运行完整分析
    results = processor.complete_image_analysis(test_points)
    report = processor.generate_analysis_report(results)
    
    print(report)
    print(f"\n生成了 {len(results['contours'])} 个轮廓")
