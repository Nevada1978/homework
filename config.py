# 大作业配置文件
import numpy as np

# 数据参数
DATA_CONFIG = {
    'num_laser_lines': 1280,        # 激光线数量
    'points_per_line': 640,         # 每条线的点数
    'total_points': 819200,         # 总点数
    'invalid_point': np.array([-0.017861, 0.228109, 18.9038]),  # 无效点标记
    'theoretical_point_interval': 5,  # 理论测量点间隔
}

# 滤波参数
FILTER_CONFIG = {
    'distance_threshold': 0.11,      # 距离阈值滤波：11cm
    'isolate_min_continuous': 15,    # 孤立点滤波：最小连续点数
    'knn_neighbors': 50,             # K近邻滤波：邻居数量
    'knn_std_multiplier': 3.0,       # K近邻滤波：3σ法则
}

# 燃烧室分离参数
COMBUSTION_CONFIG = {
    'effective_radius_multiplier': 1.1,  # 有效区域半径放大倍数
    'boundary_z_threshold': 0.01,        # Z坐标边界阈值
}

# 图像处理参数
IMAGE_CONFIG = {
    'output_size': (800, 800),       # 输出图像尺寸
    'gaussian_blur_kernel': (5, 5),  # 高斯模糊核大小
    'canny_low_threshold': 50,       # Canny边缘检测低阈值
    'canny_high_threshold': 150,     # Canny边缘检测高阈值
}

# 文件路径
FILE_PATHS = {
    'input_data': '大作业实验数据/8.txt',
    'output_dir': 'output/',
    'report_dir': 'report/',
}