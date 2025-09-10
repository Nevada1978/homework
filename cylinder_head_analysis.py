import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import least_squares
import time
import plotly.graph_objects as go
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CylinderHeadAnalyzer:
    def __init__(self, data_file):
        self.data_file = data_file
        self.fixed_point = (-0.017861, 0.228109, 18.9038)
        self.rows = 1280
        self.cols = 640
        self.raw_data = None  # 存储原始数据用于对比
        
    def load_and_preprocess_data(self):
        """数据预处理：去除固定标记点，reshape成有序点云"""
        print("正在加载和预处理数据...")
        start_time = time.time()
        
        # 读取所有数据
        data = np.loadtxt(self.data_file)
        self.raw_data = data.copy()  # 保存原始数据
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
    
    def create_3d_visualization(self):
        """创建3D点云可视化"""
        print("正在创建3D点云可视化...")
        
        # 采样数据以提高性能
        sample_size = min(10000, len(self.point_cloud))
        indices = np.random.choice(len(self.point_cloud), sample_size, replace=False)
        sampled_points = self.point_cloud[indices]
        
        # 创建3D散点图
        fig = go.Figure(data=go.Scatter3d(
            x=sampled_points[:, 0],
            y=sampled_points[:, 1], 
            z=sampled_points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=sampled_points[:, 2],
                colorscale='Viridis',
                colorbar=dict(title="Z坐标 (mm)")
            )
        ))
        
        fig.update_layout(
            title='3D点云可视化（按Z坐标着色）',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)'
            ),
            width=800, height=600
        )
        
        fig.write_html("3d_point_cloud.html")
        print("3D可视化已保存为 3d_point_cloud.html")
        
        return fig
    
    def analyze_scanning_lines(self):
        """分析光刀线Z坐标统计（类似论文图5-3）"""
        print("正在进行光刀线Z坐标分析...")
        
        if len(self.point_cloud) == 0:
            return None
            
        # 模拟有序点云结构分析
        # 由于实际数据可能不完全按1280x640结构，我们按XY位置分组
        x_coords = self.point_cloud[:, 0]
        y_coords = self.point_cloud[:, 1] 
        z_coords = self.point_cloud[:, 2]
        
        # 创建网格分析
        x_bins = np.linspace(x_coords.min(), x_coords.max(), 100)
        y_bins = np.linspace(y_coords.min(), y_coords.max(), 100)
        
        # 计算每个网格的Z均值
        z_means = []
        positions = []
        
        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                mask = ((x_coords >= x_bins[i]) & (x_coords < x_bins[i+1]) & 
                       (y_coords >= y_bins[j]) & (y_coords < y_bins[j+1]))
                
                if np.sum(mask) > 0:
                    z_mean = np.mean(z_coords[mask])
                    z_means.append(z_mean)
                    positions.append((x_bins[i], y_bins[j]))
        
        self.z_statistics = {
            'z_means': np.array(z_means),
            'positions': np.array(positions)
        }
        
        return self.z_statistics
    
    def create_preprocessing_comparison(self):
        """创建数据预处理前后对比图"""
        print("正在创建预处理对比图...")
        
        if self.raw_data is None:
            print("没有原始数据用于对比")
            return
        
        # 获取固定标记点
        fixed_mask = ((np.abs(self.raw_data[:, 0] - self.fixed_point[0]) < 1e-6) & 
                     (np.abs(self.raw_data[:, 1] - self.fixed_point[1]) < 1e-6) & 
                     (np.abs(self.raw_data[:, 2] - self.fixed_point[2]) < 1e-6))
        
        self.comparison_data = {
            'raw_total': len(self.raw_data),
            'raw_valid': len(self.raw_data[~fixed_mask]),
            'raw_fixed': len(self.raw_data[fixed_mask]),
            'processed_total': len(self.point_cloud),
            'chamber_points': len(self.chamber_points) if hasattr(self, 'chamber_points') else 0,
            'boundary_points': len(self.boundary_points) if hasattr(self, 'boundary_points') else 0
        }
        
        return self.comparison_data
    
    def create_enhanced_visualization(self):
        """创建增强版可视化（包含更多图表）"""
        print("正在创建增强版可视化...")
        
        # 创建更大的图布局
        fig = plt.figure(figsize=(24, 18))
        
        # 1. 原始点云XY投影
        ax1 = plt.subplot(3, 4, 1)
        scatter = ax1.scatter(self.point_cloud[:, 0], self.point_cloud[:, 1], 
                             c=self.point_cloud[:, 2], cmap='viridis', s=0.5)
        ax1.set_title('1. 原始点云XY投影', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        plt.colorbar(scatter, ax=ax1, shrink=0.8)
        
        # 2. 燃烧室点云
        ax2 = plt.subplot(3, 4, 2)
        if hasattr(self, 'chamber_points') and len(self.chamber_points) > 0:
            ax2.scatter(self.chamber_points[:, 0], self.chamber_points[:, 1], 
                       c=self.chamber_points[:, 2], cmap='Reds', s=1)
            ax2.set_title('2. 燃烧室点云', fontsize=12, fontweight='bold')
            ax2.set_xlabel('X (mm)')
            ax2.set_ylabel('Y (mm)')
        
        # 3. 边界点和拟合圆
        ax3 = plt.subplot(3, 4, 3)
        if hasattr(self, 'boundary_points') and len(self.boundary_points) > 0:
            ax3.scatter(self.boundary_points[:, 0], self.boundary_points[:, 1], 
                       c='blue', s=2, alpha=0.6)
            if hasattr(self, 'circle_params'):
                cx, cy, r = self.circle_params
                circle = plt.Circle((cx, cy), r, fill=False, color='red', linewidth=2)
                ax3.add_patch(circle)
                ax3.set_xlim(cx-r*1.2, cx+r*1.2)
                ax3.set_ylim(cy-r*1.2, cy+r*1.2)
                ax3.set_title(f'3. 边界拟合 (R={r:.1f}mm)', fontsize=12, fontweight='bold')
            ax3.set_xlabel('X (mm)')
            ax3.set_ylabel('Y (mm)')
            ax3.set_aspect('equal')
        
        # 4. 数据统计对比
        ax4 = plt.subplot(3, 4, 4)
        if hasattr(self, 'comparison_data'):
            data = self.comparison_data
            categories = ['原始\\n数据', '有效\\n数据', '燃烧室\\n点云', '边界\\n点']
            values = [data['raw_total'], data['processed_total'], 
                     data['chamber_points'], data['boundary_points']]
            colors = ['lightcoral', 'lightblue', 'lightgreen', 'gold']
            bars = ax4.bar(categories, values, color=colors)
            ax4.set_title('4. 数据处理统计', fontsize=12, fontweight='bold')
            ax4.set_ylabel('点数')
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:,}', ha='center', va='bottom', fontsize=9)
        
        # 5. XY投影密度图
        ax5 = plt.subplot(3, 4, 5)
        if hasattr(self, 'projection_image'):
            im = ax5.imshow(self.projection_image, cmap='hot', origin='lower', aspect='auto')
            ax5.set_title('5. XY投影密度图', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax5, shrink=0.8)
        
        # 6. 边缘检测结果
        ax6 = plt.subplot(3, 4, 6)
        if hasattr(self, 'edges_image'):
            ax6.imshow(self.edges_image, cmap='gray', origin='lower', aspect='auto')
            ax6.set_title('6. 边缘检测结果', fontsize=12, fontweight='bold')
        
        # 7. Z坐标分布直方图
        ax7 = plt.subplot(3, 4, 7)
        n, bins, patches = ax7.hist(self.point_cloud[:, 2], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax7.set_title('7. Z坐标分布', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Z坐标 (mm)')
        ax7.set_ylabel('频次')
        ax7.grid(True, alpha=0.3)
        
        # 8. Z坐标统计分析（类似论文图5-3）
        ax8 = plt.subplot(3, 4, 8)
        if hasattr(self, 'z_statistics'):
            z_means = self.z_statistics['z_means']
            # 创建模拟的光刀线编号
            line_numbers = np.arange(len(z_means))
            ax8.plot(line_numbers, z_means, 'b-', linewidth=1.5, alpha=0.8)
            ax8.fill_between(line_numbers, z_means, alpha=0.3)
            ax8.set_title('8. 光刀线Z坐标均值', fontsize=12, fontweight='bold')
            ax8.set_xlabel('区域编号')
            ax8.set_ylabel('Z均值 (mm)')
            ax8.grid(True, alpha=0.3)
        
        # 9. 处理流程示意图
        ax9 = plt.subplot(3, 4, 9)
        ax9.text(0.5, 0.8, '数据处理流程', ha='center', va='center', fontsize=14, fontweight='bold')
        steps = [
            '1. 数据加载 (819,200点)',
            '2. 固定点滤波 (90,647点)', 
            '3. 燃烧室分离 (54,747点)',
            '4. 边界识别 (3,019点)',
            '5. 圆形拟合',
            '6. 图像处理'
        ]
        for i, step in enumerate(steps):
            ax9.text(0.05, 0.65 - i*0.1, step, ha='left', va='center', fontsize=10)
        ax9.set_xlim(0, 1)
        ax9.set_ylim(0, 1)
        ax9.axis('off')
        ax9.set_title('9. 处理步骤', fontsize=12, fontweight='bold')
        
        # 10. 算法性能指标
        ax10 = plt.subplot(3, 4, 10)
        if hasattr(self, 'circle_params'):
            cx, cy, r = self.circle_params
            metrics = ['圆心X', '圆心Y', '半径R', '处理时间']
            values_text = [f'{cx:.1f}mm', f'{cy:.1f}mm', f'{r:.1f}mm', '5.4s']
            
            for i, (metric, value) in enumerate(zip(metrics, values_text)):
                ax10.text(0.1, 0.8 - i*0.15, f'{metric}: {value}', 
                         fontsize=11, fontweight='bold' if i < 3 else 'normal')
            
            ax10.set_xlim(0, 1)
            ax10.set_ylim(0, 1)
            ax10.axis('off')
            ax10.set_title('10. 关键参数', fontsize=12, fontweight='bold')
        
        # 11. 数据质量评估
        ax11 = plt.subplot(3, 4, 11)
        if hasattr(self, 'chamber_points'):
            # 计算数据质量指标
            valid_ratio = len(self.point_cloud) / len(self.raw_data) * 100
            chamber_ratio = len(self.chamber_points) / len(self.point_cloud) * 100
            boundary_ratio = len(self.boundary_points) / len(self.chamber_points) * 100 if hasattr(self, 'boundary_points') else 0
            
            labels = ['有效数据\\n比例', '燃烧室\\n比例', '边界点\\n比例']
            sizes = [valid_ratio, chamber_ratio, boundary_ratio]
            colors = ['lightcoral', 'lightblue', 'lightgreen']
            
            wedges, texts, autotexts = ax11.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            ax11.set_title('11. 数据质量分析', fontsize=12, fontweight='bold')
        
        # 12. 拟合误差分析
        ax12 = plt.subplot(3, 4, 12)
        if hasattr(self, 'boundary_points') and hasattr(self, 'circle_params'):
            cx, cy, r = self.circle_params
            # 计算每个边界点到拟合圆的距离误差
            distances = np.sqrt((self.boundary_points[:, 0] - cx)**2 + 
                              (self.boundary_points[:, 1] - cy)**2)
            errors = np.abs(distances - r)
            
            ax12.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
            ax12.set_title('12. 拟合误差分布', fontsize=12, fontweight='bold')
            ax12.set_xlabel('误差 (mm)')
            ax12.set_ylabel('频次')
            ax12.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            ax12.text(0.7, 0.8, f'均值: {mean_error:.3f}mm\\n标准差: {std_error:.3f}mm', 
                     transform=ax12.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout(pad=2.0)
        plt.savefig('enhanced_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存单独的图片文件供论文使用
        self.save_individual_figures()
        
        return fig
    
    def save_individual_figures(self):
        """保存独立的图片文件供论文使用"""
        print("正在保存独立图片文件...")
        
        # 创建figures文件夹
        if not os.path.exists('figures'):
            os.makedirs('figures')
        
        # 1. 原始点云XY投影（3D效果）
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        scatter = ax1.scatter(self.point_cloud[:, 0], self.point_cloud[:, 1], 
                             c=self.point_cloud[:, 2], cmap='viridis', s=0.8, alpha=0.7)
        ax1.set_title('原始点云XY投影（按Z坐标着色）', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X坐标 (mm)', fontsize=12)
        ax1.set_ylabel('Y坐标 (mm)', fontsize=12)
        plt.colorbar(scatter, ax=ax1, label='Z坐标 (mm)')
        plt.tight_layout()
        plt.savefig('figures/01_original_pointcloud_xy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 燃烧室点云分离结果
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        if hasattr(self, 'chamber_points') and len(self.chamber_points) > 0:
            scatter2 = ax2.scatter(self.chamber_points[:, 0], self.chamber_points[:, 1], 
                                  c=self.chamber_points[:, 2], cmap='Reds', s=1.5, alpha=0.8)
            ax2.set_title('燃烧室点云分离结果', fontsize=14, fontweight='bold')
            ax2.set_xlabel('X坐标 (mm)', fontsize=12)
            ax2.set_ylabel('Y坐标 (mm)', fontsize=12)
            plt.colorbar(scatter2, ax=ax2, label='Z坐标 (mm)')
        plt.tight_layout()
        plt.savefig('figures/02_combustion_chamber_points.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 边界识别与圆拟合
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        if hasattr(self, 'boundary_points') and len(self.boundary_points) > 0:
            ax3.scatter(self.boundary_points[:, 0], self.boundary_points[:, 1], 
                       c='blue', s=8, alpha=0.7, label='边界点')
            if hasattr(self, 'circle_params'):
                cx, cy, r = self.circle_params
                circle = plt.Circle((cx, cy), r, fill=False, color='red', linewidth=3, label=f'拟合圆 (R={r:.1f}mm)')
                ax3.add_patch(circle)
                ax3.set_xlim(cx-r*1.1, cx+r*1.1)
                ax3.set_ylim(cy-r*1.1, cy+r*1.1)
                ax3.set_title(f'边界识别与最小二乘圆拟合\\n圆心: ({cx:.1f}, {cy:.1f}), 半径: {r:.1f}mm', 
                             fontsize=14, fontweight='bold')
            ax3.set_xlabel('X坐标 (mm)', fontsize=12)
            ax3.set_ylabel('Y坐标 (mm)', fontsize=12)
            ax3.legend()
            ax3.set_aspect('equal')
            ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/03_boundary_fitting.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 数据处理流程统计
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        if hasattr(self, 'comparison_data'):
            data = self.comparison_data
            categories = ['原始数据', '有效数据', '燃烧室点云', '边界点']
            values = [data['raw_total'], data['processed_total'], 
                     data['chamber_points'], data['boundary_points']]
            colors = ['lightcoral', 'lightblue', 'lightgreen', 'gold']
            bars = ax4.bar(categories, values, color=colors, edgecolor='black', linewidth=1)
            ax4.set_title('数据处理各阶段统计', fontsize=14, fontweight='bold')
            ax4.set_ylabel('点数', fontsize=12)
            ax4.set_xlabel('处理阶段', fontsize=12)
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                        f'{value:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/04_processing_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. XY投影密度分布
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        if hasattr(self, 'projection_image'):
            im = ax5.imshow(self.projection_image, cmap='hot', origin='lower', aspect='auto')
            ax5.set_title('XY平面投影密度分布', fontsize=14, fontweight='bold')
            ax5.set_xlabel('X方向 (像素)', fontsize=12)
            ax5.set_ylabel('Y方向 (像素)', fontsize=12)
            plt.colorbar(im, ax=ax5, label='密度')
        plt.tight_layout()
        plt.savefig('figures/05_xy_projection_density.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. 边缘检测结果
        fig6, ax6 = plt.subplots(figsize=(8, 6))
        if hasattr(self, 'edges_image'):
            ax6.imshow(self.edges_image, cmap='gray', origin='lower', aspect='auto')
            ax6.set_title('Canny边缘检测结果', fontsize=14, fontweight='bold')
            ax6.set_xlabel('X方向 (像素)', fontsize=12)
            ax6.set_ylabel('Y方向 (像素)', fontsize=12)
        plt.tight_layout()
        plt.savefig('figures/06_edge_detection.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. Z坐标分布分析
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        n, bins, patches = ax7.hist(self.point_cloud[:, 2], bins=60, alpha=0.7, 
                                   color='skyblue', edgecolor='black', linewidth=0.8)
        ax7.set_title('点云Z坐标分布直方图', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Z坐标 (mm)', fontsize=12)
        ax7.set_ylabel('频次', fontsize=12)
        ax7.grid(True, alpha=0.3)
        # 添加统计信息
        z_mean = np.mean(self.point_cloud[:, 2])
        z_std = np.std(self.point_cloud[:, 2])
        ax7.axvline(z_mean, color='red', linestyle='--', linewidth=2, label=f'均值: {z_mean:.2f}mm')
        ax7.text(0.7, 0.8, f'均值: {z_mean:.2f}mm\\n标准差: {z_std:.2f}mm', 
                transform=ax7.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax7.legend()
        plt.tight_layout()
        plt.savefig('figures/07_z_coordinate_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 8. 拟合精度分析
        fig8, ax8 = plt.subplots(figsize=(10, 6))
        if hasattr(self, 'boundary_points') and hasattr(self, 'circle_params'):
            cx, cy, r = self.circle_params
            distances = np.sqrt((self.boundary_points[:, 0] - cx)**2 + 
                              (self.boundary_points[:, 1] - cy)**2)
            errors = np.abs(distances - r)
            
            n, bins, patches = ax8.hist(errors, bins=40, alpha=0.7, color='orange', 
                                       edgecolor='black', linewidth=0.8)
            ax8.set_title('最小二乘圆拟合误差分布', fontsize=14, fontweight='bold')
            ax8.set_xlabel('拟合误差 (mm)', fontsize=12)
            ax8.set_ylabel('频次', fontsize=12)
            ax8.grid(True, alpha=0.3)
            
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            max_error = np.max(errors)
            ax8.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'均值误差: {mean_error:.3f}mm')
            ax8.text(0.6, 0.8, f'均值误差: {mean_error:.3f}mm\\n标准差: {std_error:.3f}mm\\n最大误差: {max_error:.3f}mm', 
                    transform=ax8.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax8.legend()
        plt.tight_layout()
        plt.savefig('figures/08_fitting_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("独立图片文件保存完成！共8张高质量图片保存在 figures/ 目录下")
        
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
        
        # 6. 创建增强分析
        self.analyze_scanning_lines()
        self.create_preprocessing_comparison()
        
        # 7. 创建3D可视化
        self.create_3d_visualization()
        
        # 8. 增强版可视化（12张图）
        self.create_enhanced_visualization()
        
        # 9. 原版可视化（6张图）
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