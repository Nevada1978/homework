import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import least_squares
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CylinderHeadAnalyzer:
    def __init__(self, data_file):
        self.data_file = data_file
        self.fixed_point = (-0.017861, 0.228109, 18.9038)
        self.rows = 1280
        self.cols = 640
        
    def load_and_preprocess_data(self):
        """数据预处理：去除固定标记点，reshape成有序点云"""
        print("正在加载和预处理数据...")
        start_time = time.time()
        
        # 读取所有数据
        data = np.loadtxt(self.data_file)
        print(f"总数据点: {len(data)}")
        
        # 去除固定标记点
        fixed_mask = ~((np.abs(data[:, 0] - self.fixed_point[0]) < 1e-6) & 
                      (np.abs(data[:, 1] - self.fixed_point[1]) < 1e-6) & 
                      (np.abs(data[:, 2] - self.fixed_point[2]) < 1e-6))
        
        valid_data = data[fixed_mask]
        print(f"有效数据点: {len(valid_data)}")
        
        # 存储有效点云
        self.point_cloud = valid_data
        
        print(f"数据预处理完成，耗时: {time.time() - start_time:.2f}秒")
        return valid_data
        
    def extract_combustion_chamber(self, z_threshold=0.1):
        """燃烧室点云分离：基于Z坐标阈值"""
        print("正在分离燃烧室点云...")
        
        # 分析Z坐标分布
        z_coords = self.point_cloud[:, 2]
        print(f"Z坐标范围: [{z_coords.min():.4f}, {z_coords.max():.4f}]")
        
        # 燃烧室点云：Z坐标接近0且大于某个阈值的点
        chamber_mask = (z_coords > -z_threshold) & (z_coords < z_threshold)
        chamber_points = self.point_cloud[chamber_mask]
        
        print(f"燃烧室点云数量: {len(chamber_points)}")
        
        if len(chamber_points) == 0:
            # 如果没找到，尝试更大的阈值
            print("未找到燃烧室点云，尝试更大阈值...")
            chamber_mask = (z_coords > z_coords.min() + 0.1) & (z_coords < z_coords.min() + 1.0)
            chamber_points = self.point_cloud[chamber_mask]
            print(f"调整后燃烧室点云数量: {len(chamber_points)}")
        
        self.chamber_points = chamber_points
        return chamber_points
        
    def extract_boundary_points(self):
        """边界点识别：基于XY平面投影的边缘检测"""
        print("正在提取边界点...")
        
        if len(self.chamber_points) == 0:
            print("没有燃烧室点云数据")
            return np.array([])
        
        # XY坐标
        xy_points = self.chamber_points[:, :2]
        
        # 创建2D网格进行边缘检测
        x_min, x_max = xy_points[:, 0].min(), xy_points[:, 0].max()
        y_min, y_max = xy_points[:, 1].min(), xy_points[:, 1].max()
        
        # 创建密度图
        img_size = 200
        x_bins = np.linspace(x_min, x_max, img_size)
        y_bins = np.linspace(y_min, y_max, img_size)
        
        density, _, _ = np.histogram2d(xy_points[:, 0], xy_points[:, 1], 
                                     bins=[x_bins, y_bins])
        
        # 二值化
        binary_img = (density > 0).astype(np.uint8) * 255
        
        # 边缘检测
        edges = cv2.Canny(binary_img, 50, 150)
        
        # 提取边缘点坐标
        edge_indices = np.where(edges > 0)
        
        if len(edge_indices[0]) == 0:
            print("未检测到边缘")
            return np.array([])
        
        # 转换回实际坐标
        edge_x = x_bins[edge_indices[1]]
        edge_y = y_bins[edge_indices[0]]
        boundary_points = np.column_stack([edge_x, edge_y])
        
        print(f"边界点数量: {len(boundary_points)}")
        self.boundary_points = boundary_points
        return boundary_points
        
    def fit_circle(self, points):
        """最小二乘法圆拟合"""
        if len(points) < 3:
            print("边界点不足，无法拟合圆")
            return None
            
        print("正在进行圆拟合...")
        
        # 初始估计
        x_mean = np.mean(points[:, 0])
        y_mean = np.mean(points[:, 1])
        r_init = np.std(points)
        
        def circle_residuals(params, points):
            cx, cy, r = params
            return np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2) - r
        
        # 最小二乘拟合
        result = least_squares(circle_residuals, [x_mean, y_mean, r_init], args=(points,))
        
        if result.success:
            cx, cy, r = result.x
            print(f"拟合圆心: ({cx:.4f}, {cy:.4f}), 半径: {r:.4f}")
            self.circle_params = (cx, cy, r)
            return (cx, cy, r)
        else:
            print("圆拟合失败")
            return None
            
    def create_projection_image(self):
        """创建XY投影图像并进行边缘检测"""
        print("正在创建投影图像...")
        
        # 创建高分辨率投影图像
        xy_points = self.chamber_points[:, :2]
        x_min, x_max = xy_points[:, 0].min(), xy_points[:, 0].max()
        y_min, y_max = xy_points[:, 1].min(), xy_points[:, 1].max()
        
        # 创建图像
        img_size = 400
        x_bins = np.linspace(x_min, x_max, img_size)
        y_bins = np.linspace(y_min, y_max, img_size)
        
        density, _, _ = np.histogram2d(xy_points[:, 0], xy_points[:, 1], 
                                     bins=[x_bins, y_bins])
        
        # 归一化为灰度图像
        img = (density / density.max() * 255).astype(np.uint8)
        
        # 边缘检测
        edges = cv2.Canny(img, 30, 100)
        
        self.projection_image = img
        self.edges_image = edges
        
        return img, edges
        
    def visualize_results(self):
        """可视化所有结果"""
        _, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 原始点云XY投影
        axes[0, 0].scatter(self.point_cloud[:, 0], self.point_cloud[:, 1], 
                          c=self.point_cloud[:, 2], cmap='viridis', s=0.1)
        axes[0, 0].set_title('原始点云XY投影')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        
        # 2. 燃烧室点云
        if hasattr(self, 'chamber_points') and len(self.chamber_points) > 0:
            axes[0, 1].scatter(self.chamber_points[:, 0], self.chamber_points[:, 1], 
                              c='red', s=1)
            axes[0, 1].set_title('燃烧室点云')
            axes[0, 1].set_xlabel('X')
            axes[0, 1].set_ylabel('Y')
        
        # 3. 边界点和拟合圆
        if hasattr(self, 'boundary_points') and len(self.boundary_points) > 0:
            axes[0, 2].scatter(self.boundary_points[:, 0], self.boundary_points[:, 1], 
                              c='blue', s=2)
            if hasattr(self, 'circle_params'):
                cx, cy, r = self.circle_params
                circle = plt.Circle((cx, cy), r, fill=False, color='red', linewidth=2)
                axes[0, 2].add_patch(circle)
                axes[0, 2].set_xlim(cx-r*1.2, cx+r*1.2)
                axes[0, 2].set_ylim(cy-r*1.2, cy+r*1.2)
            axes[0, 2].set_title('边界点和拟合圆')
            axes[0, 2].set_xlabel('X')
            axes[0, 2].set_ylabel('Y')
            axes[0, 2].set_aspect('equal')
        
        # 4. 投影图像
        if hasattr(self, 'projection_image'):
            axes[1, 0].imshow(self.projection_image, cmap='gray', origin='lower')
            axes[1, 0].set_title('XY投影灰度图')
        
        # 5. 边缘检测结果
        if hasattr(self, 'edges_image'):
            axes[1, 1].imshow(self.edges_image, cmap='gray', origin='lower')
            axes[1, 1].set_title('边缘检测结果')
        
        # 6. Z坐标分布
        axes[1, 2].hist(self.point_cloud[:, 2], bins=50, alpha=0.7)
        axes[1, 2].set_title('Z坐标分布')
        axes[1, 2].set_xlabel('Z')
        axes[1, 2].set_ylabel('频次')
        
        plt.tight_layout()
        plt.savefig('cylinder_head_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_analysis(self):
        """运行完整分析流程"""
        print("开始缸盖燃烧室点云分析...")
        total_start = time.time()
        
        # 1. 数据预处理
        self.load_and_preprocess_data()
        
        # 2. 燃烧室分离
        self.extract_combustion_chamber()
        
        # 3. 边界点提取
        self.extract_boundary_points()
        
        # 4. 圆拟合
        if hasattr(self, 'boundary_points') and len(self.boundary_points) > 0:
            self.fit_circle(self.boundary_points)
        
        # 5. 图像处理
        self.create_projection_image()
        
        # 6. 可视化
        self.visualize_results()
        
        total_time = time.time() - total_start
        print(f"分析完成！总耗时: {total_time:.2f}秒")
        
        return {
            'total_points': len(self.point_cloud),
            'chamber_points': len(self.chamber_points) if hasattr(self, 'chamber_points') else 0,
            'boundary_points': len(self.boundary_points) if hasattr(self, 'boundary_points') else 0,
            'circle_params': self.circle_params if hasattr(self, 'circle_params') else None,
            'processing_time': total_time
        }

if __name__ == "__main__":
    # 主程序执行
    data_file = "大作业实验数据/8.txt"
    analyzer = CylinderHeadAnalyzer(data_file)
    results = analyzer.run_analysis()
    
    print("\n=== 分析结果摘要 ===")
    for key, value in results.items():
        print(f"{key}: {value}")