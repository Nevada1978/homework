"""
三种滤波算法实现
按照理论文档3.2.1-3.2.3实现距离阈值滤波、孤立点滤波和K近邻统计滤波
"""
import numpy as np
from typing import Tuple, Optional
import logging
from sklearn.neighbors import NearestNeighbors
from config import FILTER_CONFIG
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
import os
import signal

class PointCloudFilters:
    """点云滤波算法集合"""
    
    def __init__(self):
        self.distance_threshold = FILTER_CONFIG['distance_threshold']
        self.isolate_min_continuous = FILTER_CONFIG['isolate_min_continuous']
        self.knn_neighbors = FILTER_CONFIG['knn_neighbors']
        self.knn_std_multiplier = FILTER_CONFIG['knn_std_multiplier']
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def distance_threshold_filter(self, point_cloud: np.ndarray, 
                                invalid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        3.2.1 距离阈值滤波
        滤除Z坐标大于设定阈值的点（距离缸盖顶面过远的点）
        
        Args:
            point_cloud: 输入点云 (1280, 640, 3)
            invalid_mask: 无效点掩码，如果提供则保持无效点不变
            
        Returns:
            滤波后的点云，超过阈值的点设为NaN
        """
        self.logger.info(f"开始距离阈值滤波，阈值: {self.distance_threshold*100}cm")
        
        filtered_cloud = point_cloud.copy()
        
        # 创建Z坐标超过阈值的掩码
        z_coords = point_cloud[:, :, 2]
        over_threshold_mask = z_coords > self.distance_threshold
        
        # 如果提供了无效点掩码，则保持原无效点不变
        if invalid_mask is not None:
            # 只对原本有效的点进行阈值检查
            over_threshold_mask = over_threshold_mask & (~invalid_mask)
        
        # 将超过阈值的点设为NaN
        filtered_cloud[over_threshold_mask] = np.nan
        
        filtered_count = np.sum(over_threshold_mask)
        self.logger.info(f"距离阈值滤波完成，滤除{filtered_count}个点")
        
        return filtered_cloud
    
    def isolate_point_filter(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        3.2.2 孤立点滤波
        检测每条激光线上连续非NaN点的个数，如果小于阈值则认为是孤立点
        
        Args:
            point_cloud: 输入点云 (1280, 640, 3)
            
        Returns:
            滤波后的点云，孤立点设为NaN
        """
        self.logger.info(f"开始孤立点滤波，最小连续点数阈值: {self.isolate_min_continuous}")
        
        filtered_cloud = point_cloud.copy()
        total_filtered = 0
        
        # 逐行处理每条激光线
        for line_idx in range(point_cloud.shape[0]):
            line_data = filtered_cloud[line_idx]  # (640, 3)
            
            # 检查每个点是否为NaN
            is_nan = np.isnan(line_data).any(axis=1)  # (640,)
            
            # 找到连续的非NaN点段
            continuous_segments = self._find_continuous_segments(~is_nan)
            
            # 对于长度小于阈值的连续段，将其标记为孤立点
            for start, end in continuous_segments:
                segment_length = end - start
                if segment_length < self.isolate_min_continuous:
                    # 将该段所有点设为NaN
                    filtered_cloud[line_idx, start:end] = np.nan
                    total_filtered += segment_length
        
        self.logger.info(f"孤立点滤波完成，滤除{total_filtered}个孤立点")
        
        return filtered_cloud
    
    def _find_continuous_segments(self, mask: np.ndarray) -> list:
        """
        找到布尔数组中所有连续的True段
        
        Args:
            mask: 布尔数组
            
        Returns:
            [(start, end), ...] 连续段的起止索引列表
        """
        if len(mask) == 0:
            return []
        
        segments = []
        start = None
        
        for i, value in enumerate(mask):
            if value and start is None:
                # 连续段开始
                start = i
            elif not value and start is not None:
                # 连续段结束
                segments.append((start, i))
                start = None
        
        # 处理末尾的连续段
        if start is not None:
            segments.append((start, len(mask)))
        
        return segments
    
    def _process_chunk_knn(self, args):
        """
        多线程处理单个数据块的K近邻计算
        """
        chunk_points, all_points, k_neighbors, chunk_id = args
        
        try:
            # 构建KD树
            nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='ball_tree')
            nbrs.fit(all_points)
            
            # 计算当前块的K近邻距离
            distances, _ = nbrs.kneighbors(chunk_points)
            
            # 计算平均距离（排除自身）
            neighbor_distances = distances[:, 1:]
            mean_distances = np.mean(neighbor_distances, axis=1)
            
            return chunk_id, mean_distances
            
        except Exception as e:
            return chunk_id, f"Error: {e}"
    
    def knn_statistical_filter(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        3.2.3 基于统计的K近邻滤波（多线程优化版本）
        使用多线程分块处理，加上实时进度条
        
        Args:
            point_cloud: 输入点云 (1280, 640, 3)
            
        Returns:
            滤波后的点云，离群点设为NaN
        """
        # 保存进程ID用于kill脚本
        with open('.homework_pid', 'w') as f:
            f.write(str(os.getpid()))
            
        self.logger.info(f"开始多线程K近邻统计滤波，K={self.knn_neighbors}, 阈值={self.knn_std_multiplier}σ")
        self.logger.info(f"进程ID: {os.getpid()}")
        
        filtered_cloud = point_cloud.copy()
        
        # 提取所有有效点
        valid_mask = ~np.isnan(point_cloud).any(axis=2)
        valid_points = point_cloud[valid_mask]
        
        if len(valid_points) < self.knn_neighbors + 1:
            self.logger.warning("有效点数量不足，跳过K近邻滤波")
            return filtered_cloud
        
        self.logger.info(f"有效点数量: {len(valid_points)}")
        
        # 确定线程数和块大小
        num_threads = min(mp.cpu_count(), 8)  # 最多8个线程
        chunk_size = max(1000, len(valid_points) // (num_threads * 4))  # 每块至少1000个点
        
        self.logger.info(f"使用 {num_threads} 个线程，每块 {chunk_size} 个点")
        
        # 分割数据为块
        chunks = []
        for i in range(0, len(valid_points), chunk_size):
            chunk_end = min(i + chunk_size, len(valid_points))
            chunk_points = valid_points[i:chunk_end]
            chunks.append((chunk_points, valid_points, self.knn_neighbors, i // chunk_size))
        
        self.logger.info(f"数据分成 {len(chunks)} 块进行并行处理")
        
        # 多线程处理
        all_mean_distances = {}
        completed_chunks = 0
        
        # 使用tqdm显示进度条
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 提交所有任务
            future_to_chunk = {
                executor.submit(self._process_chunk_knn, chunk): chunk_id 
                for chunk, _, _, chunk_id in chunks
            }
            
            # 使用tqdm创建进度条
            with tqdm(total=len(chunks), desc="K近邻计算进度", unit="块") as pbar:
                for future in as_completed(future_to_chunk):
                    chunk_id = future_to_chunk[future]
                    try:
                        result_chunk_id, mean_distances = future.result()
                        
                        if isinstance(mean_distances, str):  # 错误情况
                            self.logger.error(f"块 {result_chunk_id} 处理失败: {mean_distances}")
                        else:
                            all_mean_distances[result_chunk_id] = mean_distances
                            completed_chunks += 1
                        
                        # 更新进度条
                        pbar.update(1)
                        pbar.set_postfix({
                            "已完成": f"{completed_chunks}/{len(chunks)}",
                            "当前块": result_chunk_id
                        })
                        
                    except Exception as e:
                        self.logger.error(f"块 {chunk_id} 处理异常: {e}")
                        pbar.update(1)
        
        # 合并所有结果
        self.logger.info("合并多线程计算结果...")
        sorted_chunks = sorted(all_mean_distances.keys())
        mean_distances = np.concatenate([all_mean_distances[chunk_id] for chunk_id in sorted_chunks])
        
        # 计算全局统计量
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        threshold = global_mean + self.knn_std_multiplier * global_std
        
        self.logger.info(f"统计计算完成:")
        self.logger.info(f"  全局平均距离: {global_mean:.4f}")
        self.logger.info(f"  全局标准差: {global_std:.4f}")
        self.logger.info(f"  3σ阈值: {threshold:.4f}")
        
        # 应用阈值滤除离群点
        self.logger.info("应用阈值滤除离群点...")
        outlier_mask = mean_distances > threshold
        outlier_count = 0
        
        valid_indices = np.where(valid_mask)
        
        # 使用tqdm显示离群点处理进度
        with tqdm(total=len(mean_distances), desc="处理离群点", unit="点") as pbar:
            for i, is_outlier in enumerate(outlier_mask):
                if is_outlier:
                    if i < len(valid_indices[0]):
                        row_idx = valid_indices[0][i]
                        col_idx = valid_indices[1][i]
                        filtered_cloud[row_idx, col_idx] = np.nan
                        outlier_count += 1
                
                if i % 5000 == 0:  # 每5000个点更新一次进度
                    pbar.update(5000)
            
            # 更新剩余进度
            remaining = len(mean_distances) % 5000
            if remaining > 0:
                pbar.update(remaining)
        
        self.logger.info(f"K近邻统计滤波完成:")
        self.logger.info(f"  处理点数: {len(valid_points)}")
        self.logger.info(f"  滤除离群点: {outlier_count}个")
        self.logger.info(f"  离群点比例: {outlier_count/len(valid_points)*100:.2f}%")
        
        # 清理临时文件
        try:
            os.remove('.homework_pid')
        except:
            pass
            
        return filtered_cloud
    
    def apply_all_filters(self, point_cloud: np.ndarray, 
                         invalid_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        按顺序应用所有三种滤波算法
        
        Args:
            point_cloud: 输入点云
            invalid_mask: 无效点掩码
            
        Returns:
            (滤波后点云, 滤波统计信息)
        """
        self.logger.info("开始应用完整滤波流水线")
        
        # 统计原始有效点数量
        if invalid_mask is not None:
            original_valid = np.sum(~invalid_mask)
        else:
            original_valid = np.sum(~np.isnan(point_cloud).any(axis=2))
        
        # 阶段1: 距离阈值滤波
        filtered_cloud = self.distance_threshold_filter(point_cloud, invalid_mask)
        after_distance = np.sum(~np.isnan(filtered_cloud).any(axis=2))
        
        # 阶段2: 孤立点滤波
        filtered_cloud = self.isolate_point_filter(filtered_cloud)
        after_isolate = np.sum(~np.isnan(filtered_cloud).any(axis=2))
        
        # 阶段3: K近邻统计滤波
        filtered_cloud = self.knn_statistical_filter(filtered_cloud)
        after_knn = np.sum(~np.isnan(filtered_cloud).any(axis=2))
        
        # 统计信息
        stats = {
            'original_valid_points': original_valid,
            'after_distance_filter': after_distance,
            'after_isolate_filter': after_isolate,
            'after_knn_filter': after_knn,
            'distance_filtered': original_valid - after_distance,
            'isolate_filtered': after_distance - after_isolate,
            'knn_filtered': after_isolate - after_knn,
            'total_filtered': original_valid - after_knn,
            'final_valid_ratio': after_knn / original_valid if original_valid > 0 else 0
        }
        
        self.logger.info("滤波流水线完成:")
        self.logger.info(f"  原始有效点: {stats['original_valid_points']}")
        self.logger.info(f"  距离滤波后: {stats['after_distance_filter']} (滤除{stats['distance_filtered']})")
        self.logger.info(f"  孤立点滤波后: {stats['after_isolate_filter']} (滤除{stats['isolate_filtered']})")
        self.logger.info(f"  K近邻滤波后: {stats['after_knn_filter']} (滤除{stats['knn_filtered']})")
        self.logger.info(f"  最终保留率: {stats['final_valid_ratio']*100:.1f}%")
        
        return filtered_cloud, stats


if __name__ == "__main__":
    # 测试滤波算法
    from data_loader import PointCloudLoader
    
    # 创建测试数据
    loader = PointCloudLoader()
    filters = PointCloudFilters()
    
    # 生成模拟点云数据进行测试
    test_cloud = np.random.randn(10, 10, 3) * 0.05  # 小范围随机点云
    test_cloud[:, :, 2] += 0.05  # Z坐标偏移
    
    # 添加一些离群点
    test_cloud[0, 0, 2] = 0.5  # 距离离群点
    test_cloud[1, 1:3] = np.nan  # 创建孤立点
    
    print("测试滤波算法...")
    filtered_cloud, stats = filters.apply_all_filters(test_cloud)
    print("滤波统计:", stats)