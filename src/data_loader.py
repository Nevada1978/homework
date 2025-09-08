"""
数据加载和预处理模块
实现点云数据的加载、重构和基础预处理功能
"""
import numpy as np
from typing import Tuple, Optional
import logging
from config import DATA_CONFIG

class PointCloudLoader:
    """点云数据加载器"""
    
    def __init__(self):
        self.num_lines = DATA_CONFIG['num_laser_lines']
        self.points_per_line = DATA_CONFIG['points_per_line']
        self.invalid_point = DATA_CONFIG['invalid_point']
        self.total_points = DATA_CONFIG['total_points']
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, filename: str) -> np.ndarray:
        """
        加载点云数据文件
        
        Args:
            filename: 数据文件路径
            
        Returns:
            shape为(1280, 640, 3)的点云数组，按激光线组织
        """
        self.logger.info(f"开始加载数据文件: {filename}")
        
        try:
            # 加载所有点云数据
            raw_data = np.loadtxt(filename)
            
            if raw_data.shape[0] != self.total_points:
                raise ValueError(f"数据行数不匹配！期望{self.total_points}，实际{raw_data.shape[0]}")
                
            if raw_data.shape[1] != 3:
                raise ValueError(f"数据列数不匹配！期望3列(X,Y,Z)，实际{raw_data.shape[1]}列")
            
            # 重构为激光线格式：(1280条线, 640点/线, 3坐标)
            point_cloud = raw_data.reshape(self.num_lines, self.points_per_line, 3)
            
            self.logger.info(f"数据加载完成，形状: {point_cloud.shape}")
            return point_cloud
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise
    
    def identify_invalid_points(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        识别无效点位置
        
        Args:
            point_cloud: 点云数据
            
        Returns:
            bool数组，True表示无效点
        """
        # 使用逐点比较来识别无效点
        # 计算每个点与无效点的差异
        diff = point_cloud - self.invalid_point.reshape(1, 1, 3)
        
        # 检查XYZ三个坐标是否都接近无效点坐标（使用容差）
        close_mask = np.abs(diff) < 1e-6  # (1280, 640, 3)
        
        # 只有当XYZ三个坐标都匹配时才认为是无效点
        invalid_mask = np.all(close_mask, axis=2)
        
        invalid_count = np.sum(invalid_mask)
        total_points = point_cloud.shape[0] * point_cloud.shape[1]
        
        self.logger.info(f"识别出无效点: {invalid_count}/{total_points} ({invalid_count/total_points*100:.1f}%)")
        
        return invalid_mask
    
    def identify_theoretical_points(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        识别理论测量点位置
        根据"从第一个点开始，每间隔四个点均为理论测量点"
        
        Args:
            point_cloud: 点云数据
            
        Returns:
            bool数组，True表示理论测量点
        """
        theoretical_mask = np.zeros((self.num_lines, self.points_per_line), dtype=bool)
        
        # 每条激光线独立处理
        for line_idx in range(self.num_lines):
            # 从第1个点开始(索引0)，每间隔4个点取1个理论测量点
            # 即索引: 0, 5, 10, 15, 20, ...
            theoretical_indices = np.arange(0, self.points_per_line, 5)
            theoretical_mask[line_idx, theoretical_indices] = True
        
        theoretical_count = np.sum(theoretical_mask)
        self.logger.info(f"识别出理论测量点: {theoretical_count}个")
        
        return theoretical_mask
    
    def get_valid_points(self, point_cloud: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取有效点云数据
        
        Args:
            point_cloud: 原始点云数据
            
        Returns:
            (有效点坐标数组, 有效点在原数组中的索引)
        """
        invalid_mask = self.identify_invalid_points(point_cloud)
        valid_mask = ~invalid_mask
        
        # 提取有效点的坐标
        valid_points = point_cloud[valid_mask]
        
        # 获取有效点的索引位置
        valid_indices = np.where(valid_mask)
        
        self.logger.info(f"有效点数量: {len(valid_points)}")
        
        return valid_points, valid_indices
    
    def analyze_point_cloud_stats(self, point_cloud: np.ndarray) -> dict:
        """
        分析点云统计信息
        
        Args:
            point_cloud: 点云数据
            
        Returns:
            包含统计信息的字典
        """
        valid_points, _ = self.get_valid_points(point_cloud)
        
        if len(valid_points) == 0:
            return {"error": "没有有效点"}
        
        stats = {
            'total_points': point_cloud.shape[0] * point_cloud.shape[1],
            'valid_points': len(valid_points),
            'valid_ratio': len(valid_points) / (point_cloud.shape[0] * point_cloud.shape[1]),
            'x_range': [float(valid_points[:, 0].min()), float(valid_points[:, 0].max())],
            'y_range': [float(valid_points[:, 1].min()), float(valid_points[:, 1].max())],
            'z_range': [float(valid_points[:, 2].min()), float(valid_points[:, 2].max())],
            'x_mean': float(valid_points[:, 0].mean()),
            'y_mean': float(valid_points[:, 1].mean()),
            'z_mean': float(valid_points[:, 2].mean()),
        }
        
        self.logger.info("点云统计信息:")
        self.logger.info(f"  总点数: {stats['total_points']}")
        self.logger.info(f"  有效点数: {stats['valid_points']}")
        self.logger.info(f"  有效率: {stats['valid_ratio']*100:.1f}%")
        self.logger.info(f"  X范围: [{stats['x_range'][0]:.3f}, {stats['x_range'][1]:.3f}]")
        self.logger.info(f"  Y范围: [{stats['y_range'][0]:.3f}, {stats['y_range'][1]:.3f}]")
        self.logger.info(f"  Z范围: [{stats['z_range'][0]:.3f}, {stats['z_range'][1]:.3f}]")
        
        return stats


if __name__ == "__main__":
    # 测试数据加载器
    loader = PointCloudLoader()
    
    # 使用测试数据
    test_file = "../大作业实验数据/test.txt"
    try:
        # 注意：测试文件只有675行，需要特殊处理
        test_data = np.loadtxt(test_file)
        print(f"测试数据形状: {test_data.shape}")
        
        # 创建模拟的完整数据用于测试
        full_data = np.tile(loader.invalid_point, (819200, 1))
        full_data[:len(test_data)] = test_data
        
        point_cloud = full_data.reshape(1280, 640, 3)
        
        # 分析数据
        stats = loader.analyze_point_cloud_stats(point_cloud)
        print("统计信息:", stats)
        
    except Exception as e:
        print(f"测试失败: {e}")