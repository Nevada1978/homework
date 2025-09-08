"""
燃烧室分离算法实现
按照理论文档5.1.1实现燃烧室有效点云数据提取
"""
import numpy as np
from typing import Tuple, Dict, Optional
import logging
from config import COMBUSTION_CONFIG

class CombustionChamberExtractor:
    """燃烧室点云分离器"""
    
    def __init__(self):
        self.effective_radius_multiplier = COMBUSTION_CONFIG['effective_radius_multiplier']
        self.boundary_z_threshold = COMBUSTION_CONFIG['boundary_z_threshold']
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def extract_combustion_chamber(self, point_cloud: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        5.1.1 燃烧室有效点云数据提取的主流程
        
        Args:
            point_cloud: 滤波后的点云数据 (1280, 640, 3)
            
        Returns:
            (燃烧室点云, 提取信息字典)
        """
        self.logger.info("开始燃烧室点云分离")
        
        # 步骤1: 识别边界行和列
        boundary_rows, boundary_cols = self._identify_boundary_regions(point_cloud)
        
        # 步骤2: 提取边界点
        boundary_points = self._extract_boundary_points(point_cloud, boundary_rows, boundary_cols)
        
        # 步骤3: 最小二乘法圆拟合
        circle_params = self._least_squares_circle_fit(boundary_points)
        
        # 步骤4: 提取燃烧室点云
        chamber_cloud, extraction_info = self._extract_chamber_points(
            point_cloud, circle_params
        )
        
        # 整合信息
        info = {
            'boundary_rows': boundary_rows,
            'boundary_cols': boundary_cols,
            'boundary_points_count': len(boundary_points),
            'circle_center': circle_params['center'],
            'circle_radius': circle_params['radius'],
            'effective_radius': circle_params['radius'] * self.effective_radius_multiplier,
            **extraction_info
        }
        
        self.logger.info("燃烧室分离完成:")
        self.logger.info(f"  边界点数量: {info['boundary_points_count']}")
        self.logger.info(f"  拟合圆心: ({info['circle_center'][0]:.3f}, {info['circle_center'][1]:.3f})")
        self.logger.info(f"  拟合半径: {info['circle_radius']:.3f}")
        self.logger.info(f"  有效半径: {info['effective_radius']:.3f}")
        self.logger.info(f"  燃烧室点数: {info['chamber_points_count']}")
        
        return chamber_cloud, info
    
    def _identify_boundary_regions(self, point_cloud: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        识别燃烧室边界的行列范围
        通过分析每行/列的Z坐标平均值来确定边界
        
        Args:
            point_cloud: 点云数据
            
        Returns:
            ((start_row, end_row), (start_col, end_col))
        """
        # 计算每行的Z坐标平均值（忽略NaN）
        row_z_means = np.nanmean(point_cloud[:, :, 2], axis=1)  # (1280,)
        
        # 找到Z坐标从接近0开始下降的区域
        valid_rows = ~np.isnan(row_z_means)
        if not np.any(valid_rows):
            return (0, point_cloud.shape[0]), (0, point_cloud.shape[1])
        
        # 找到有效Z值的范围
        valid_indices = np.where(valid_rows)[0]
        start_row, end_row = valid_indices[0], valid_indices[-1]
        
        # 计算每列的Z坐标平均值
        col_z_means = np.nanmean(point_cloud[:, :, 2], axis=0)  # (640,)
        valid_cols = ~np.isnan(col_z_means)
        
        if np.any(valid_cols):
            valid_col_indices = np.where(valid_cols)[0]
            start_col, end_col = valid_col_indices[0], valid_col_indices[-1]
        else:
            start_col, end_col = 0, point_cloud.shape[1]
        
        self.logger.info(f"识别边界区域: 行[{start_row}:{end_row}], 列[{start_col}:{end_col}]")
        
        return (start_row, end_row), (start_col, end_col)
    
    def _extract_boundary_points(self, point_cloud: np.ndarray, 
                               boundary_rows: Tuple[int, int],
                               boundary_cols: Tuple[int, int]) -> np.ndarray:
        """
        提取燃烧室边界点
        在边界区域内寻找Z坐标从正值变为负值的转换点
        
        Args:
            point_cloud: 点云数据
            boundary_rows: 边界行范围
            boundary_cols: 边界列范围
            
        Returns:
            边界点坐标数组 (N, 3)
        """
        boundary_points = []
        
        start_row, end_row = boundary_rows
        start_col, end_col = boundary_cols
        
        # 方法1: 行方向搜索边界点
        for row in range(start_row, min(end_row + 1, point_cloud.shape[0])):
            row_data = point_cloud[row, start_col:end_col+1]  # 该行的点云数据
            valid_mask = ~np.isnan(row_data).any(axis=2)
            
            if not np.any(valid_mask):
                continue
            
            z_values = row_data[:, 2]
            
            # 寻找Z坐标的边界点（从正值变为0附近的点）
            for col_offset in range(len(z_values) - 1):
                if valid_mask[col_offset] and valid_mask[col_offset + 1]:
                    z1, z2 = z_values[col_offset], z_values[col_offset + 1]
                    # 寻找Z值穿越0的点
                    if abs(z1) < self.boundary_z_threshold and abs(z2) > self.boundary_z_threshold:
                        boundary_points.append(row_data[col_offset])
                    elif abs(z2) < self.boundary_z_threshold and abs(z1) > self.boundary_z_threshold:
                        boundary_points.append(row_data[col_offset + 1])
        
        # 方法2: 列方向搜索边界点
        for col in range(start_col, min(end_col + 1, point_cloud.shape[1])):
            col_data = point_cloud[start_row:end_row+1, col]  # 该列的点云数据
            valid_mask = ~np.isnan(col_data).any(axis=1)
            
            if not np.any(valid_mask):
                continue
            
            z_values = col_data[:, 2]
            
            for row_offset in range(len(z_values) - 1):
                if valid_mask[row_offset] and valid_mask[row_offset + 1]:
                    z1, z2 = z_values[row_offset], z_values[row_offset + 1]
                    if abs(z1) < self.boundary_z_threshold and abs(z2) > self.boundary_z_threshold:
                        boundary_points.append(col_data[row_offset])
                    elif abs(z2) < self.boundary_z_threshold and abs(z1) > self.boundary_z_threshold:
                        boundary_points.append(col_data[row_offset + 1])
        
        if len(boundary_points) == 0:
            # 如果没找到边界点，使用Z值接近0的点作为边界
            self.logger.warning("未找到明显边界点，使用Z值接近0的点")
            for row in range(start_row, min(end_row + 1, point_cloud.shape[0])):
                for col in range(start_col, min(end_col + 1, point_cloud.shape[1])):
                    point = point_cloud[row, col]
                    if not np.isnan(point).any() and abs(point[2]) < self.boundary_z_threshold * 5:
                        boundary_points.append(point)
        
        boundary_points = np.array(boundary_points) if boundary_points else np.empty((0, 3))
        self.logger.info(f"提取边界点: {len(boundary_points)}个")
        
        return boundary_points
    
    def _least_squares_circle_fit(self, boundary_points: np.ndarray) -> Dict:
        """
        最小二乘法圆拟合
        按照文档5.1.1的数学公式实现
        
        Args:
            boundary_points: 边界点坐标 (N, 3)
            
        Returns:
            包含圆心和半径的字典
        """
        if len(boundary_points) < 3:
            self.logger.warning("边界点数量不足，使用默认圆参数")
            return {'center': (0.0, 0.0), 'radius': 10.0}
        
        # 使用XY坐标进行圆拟合
        x = boundary_points[:, 0]
        y = boundary_points[:, 1]
        N = len(boundary_points)
        
        # 按照文档公式计算
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_x2 = np.sum(x**2)
        sum_y2 = np.sum(y**2)
        sum_x3 = np.sum(x**3)
        sum_y3 = np.sum(y**3)
        sum_xy = np.sum(x * y)
        sum_x2y = np.sum(x**2 * y)
        sum_xy2 = np.sum(x * y**2)
        sum_x2_y2 = sum_x2 + sum_y2
        
        # 计算系数
        C = N * sum_x2 - sum_x**2
        D = N * sum_xy - sum_x * sum_y
        E = N * sum_x3 + N * sum_xy2 - sum_x2_y2 * sum_x
        G = N * sum_y2 - sum_y**2
        H = N * sum_x2y + N * sum_y3 - sum_x2_y2 * sum_y
        
        # 防止除零错误
        denominator = C * G - D**2
        if abs(denominator) < 1e-10:
            self.logger.warning("圆拟合计算奇异，使用备用方法")
            # 使用重心作为圆心，最大距离作为半径
            center_x, center_y = np.mean(x), np.mean(y)
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            radius = np.max(distances)
        else:
            # 计算圆参数
            a = (H * D - E * G) / denominator
            b = (H * C - E * D) / (D**2 - G * C)
            c = -(sum_x2_y2 + a * sum_x + b * sum_y) / N
            
            center_x = a / 2
            center_y = b / 2
            radius = 0.5 * np.sqrt(a**2 + b**2 - 4*c)
        
        # 验证结果合理性
        if radius <= 0 or radius > 100:  # 半径应该在合理范围内
            self.logger.warning(f"拟合半径异常: {radius}，使用备用方法")
            center_x, center_y = np.mean(x), np.mean(y)
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            radius = np.mean(distances)
        
        self.logger.info(f"圆拟合完成: 圆心({center_x:.3f}, {center_y:.3f}), 半径{radius:.3f}")
        
        return {
            'center': (float(center_x), float(center_y)),
            'radius': float(radius)
        }
    
    def _extract_chamber_points(self, point_cloud: np.ndarray, 
                              circle_params: Dict) -> Tuple[np.ndarray, Dict]:
        """
        提取燃烧室区域内的点云
        
        Args:
            point_cloud: 原始点云
            circle_params: 拟合圆参数
            
        Returns:
            (燃烧室点云, 提取统计信息)
        """
        center_x, center_y = circle_params['center']
        radius = circle_params['radius']
        effective_radius = radius * self.effective_radius_multiplier
        
        chamber_points = []
        chamber_indices = []
        
        # 遍历所有点，提取圆内的点
        for row in range(point_cloud.shape[0]):
            for col in range(point_cloud.shape[1]):
                point = point_cloud[row, col]
                
                # 跳过无效点
                if np.isnan(point).any():
                    continue
                
                # 计算到圆心的距离
                distance = np.sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
                
                # 如果在有效半径内，则加入燃烧室点云
                if distance <= effective_radius:
                    chamber_points.append(point)
                    chamber_indices.append((row, col))
        
        chamber_points = np.array(chamber_points) if chamber_points else np.empty((0, 3))
        
        info = {
            'chamber_points_count': len(chamber_points),
            'extraction_radius': effective_radius,
            'chamber_indices': chamber_indices
        }
        
        return chamber_points, info
    
    def analyze_chamber_geometry(self, chamber_points: np.ndarray) -> Dict:
        """
        分析燃烧室几何特性
        
        Args:
            chamber_points: 燃烧室点云
            
        Returns:
            几何特性字典
        """
        if len(chamber_points) == 0:
            return {'error': '无燃烧室点云数据'}
        
        # 基本统计
        x_range = [float(chamber_points[:, 0].min()), float(chamber_points[:, 0].max())]
        y_range = [float(chamber_points[:, 1].min()), float(chamber_points[:, 1].max())]
        z_range = [float(chamber_points[:, 2].min()), float(chamber_points[:, 2].max())]
        
        # 燃烧室深度（Z坐标的绝对值最大值）
        chamber_depth = float(abs(chamber_points[:, 2]).max())
        
        # 投影面积（XY平面上的包围盒面积）
        projection_area = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
        
        # 粗略体积估算（简化为平均深度×投影面积）
        avg_depth = float(abs(chamber_points[:, 2]).mean())
        estimated_volume = projection_area * avg_depth
        
        geometry = {
            'point_count': len(chamber_points),
            'x_range': x_range,
            'y_range': y_range,
            'z_range': z_range,
            'chamber_depth': chamber_depth,
            'projection_area': projection_area,
            'average_depth': avg_depth,
            'estimated_volume': estimated_volume
        }
        
        self.logger.info("燃烧室几何分析:")
        self.logger.info(f"  点云数量: {geometry['point_count']}")
        self.logger.info(f"  燃烧室深度: {geometry['chamber_depth']:.3f}")
        self.logger.info(f"  投影面积: {geometry['projection_area']:.3f}")
        self.logger.info(f"  估算体积: {geometry['estimated_volume']:.6f}")
        
        return geometry


if __name__ == "__main__":
    # 测试燃烧室分离算法
    extractor = CombustionChamberExtractor()
    
    # 创建模拟数据
    test_cloud = np.random.randn(20, 20, 3) * 0.01
    test_cloud[:, :, 2] += 0.02  # Z坐标偏移
    
    # 添加一些燃烧室特征
    center_x, center_y = 10, 10
    for i in range(20):
        for j in range(20):
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if dist < 5:  # 燃烧室区域
                test_cloud[i, j, 2] = -0.02 * (1 - dist/5)  # 负Z值，越靠近中心越深
    
    chamber_cloud, info = extractor.extract_combustion_chamber(test_cloud)
    geometry = extractor.analyze_chamber_geometry(chamber_cloud)
    
    print("测试结果:")
    print(f"提取信息: {info}")
    print(f"几何特性: {geometry}")