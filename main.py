"""
汽车发动机缸盖燃烧室容积数字化计量技术大作业
主程序：完整的点云处理和图像分析流水线
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import PointCloudLoader
from filters import PointCloudFilters
from combustion_chamber import CombustionChamberExtractor
from image_processor import ImageProcessor
from config import FILE_PATHS
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('homework_process.log'),
        logging.StreamHandler()
    ]
)

class CombustionChamberAnalyzer:
    """燃烧室分析主类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.loader = PointCloudLoader()
        self.filters = PointCloudFilters()
        self.extractor = CombustionChamberExtractor()
        self.image_processor = ImageProcessor()
        
        # 确保输出目录存在
        os.makedirs(FILE_PATHS['output_dir'], exist_ok=True)
        os.makedirs(FILE_PATHS['report_dir'], exist_ok=True)
        
    def run_complete_analysis(self, data_file: str = None) -> dict:
        """
        运行完整的燃烧室分析流程
        
        Args:
            data_file: 数据文件路径，如果为None则使用默认路径
            
        Returns:
            完整分析结果字典
        """
        if data_file is None:
            data_file = FILE_PATHS['input_data']
        
        self.logger.info("=" * 60)
        self.logger.info("开始燃烧室数字化计量分析")
        self.logger.info(f"数据文件: {data_file}")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        results = {}
        
        try:
            # 阶段1: 数据加载
            self.logger.info("\n阶段1: 数据加载和预处理")
            point_cloud = self.loader.load_data(data_file)
            invalid_mask = self.loader.identify_invalid_points(point_cloud)
            data_stats = self.loader.analyze_point_cloud_stats(point_cloud)
            
            results['data_loading'] = {
                'point_cloud_shape': point_cloud.shape,
                'invalid_points': int(np.sum(invalid_mask)),
                'stats': data_stats
            }
            
            # 阶段2: 点云滤波
            self.logger.info("\n阶段2: 点云滤波处理")
            filtered_cloud, filter_stats = self.filters.apply_all_filters(point_cloud, invalid_mask)
            
            results['filtering'] = filter_stats
            
            # 阶段3: 燃烧室分离
            self.logger.info("\n阶段3: 燃烧室点云分离")
            chamber_points, extraction_info = self.extractor.extract_combustion_chamber(filtered_cloud)
            chamber_geometry = self.extractor.analyze_chamber_geometry(chamber_points)
            
            results['combustion_chamber'] = {
                'extraction': extraction_info,
                'geometry': chamber_geometry
            }
            
            # 阶段4: 图像处理分析
            self.logger.info("\n阶段4: 图像数据分析")
            image_results = self.image_processor.complete_image_analysis(
                chamber_points, 
                FILE_PATHS['output_dir']
            )
            
            results['image_analysis'] = image_results
            
            # 计算总耗时
            total_time = time.time() - start_time
            results['processing_time'] = total_time
            
            self.logger.info(f"\n分析完成！总耗时: {total_time:.2f}秒")
            
            return results
            
        except Exception as e:
            self.logger.error(f"分析过程出现错误: {e}")
            raise
    
    def generate_comprehensive_report(self, results: dict) -> str:
        """
        生成综合分析报告
        
        Args:
            results: 完整分析结果
            
        Returns:
            报告文本
        """
        report_lines = []
        
        # 报告头部
        report_lines.extend([
            "=" * 80,
            "汽车发动机缸盖燃烧室容积数字化计量技术分析报告",
            "=" * 80,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"处理耗时: {results.get('processing_time', 0):.2f}秒",
            "",
        ])
        
        # 1. 数据加载部分
        if 'data_loading' in results:
            data = results['data_loading']
            report_lines.extend([
                "1. 数据加载和预处理",
                "-" * 30,
                f"点云数据形状: {data['point_cloud_shape']}",
                f"总点数: {data['point_cloud_shape'][0] * data['point_cloud_shape'][1]}",
                f"无效点数量: {data['invalid_points']}",
                ""
            ])
            
            if 'stats' in data and 'error' not in data['stats']:
                stats = data['stats']
                report_lines.extend([
                    "数据统计信息:",
                    f"  有效点数: {stats['valid_points']}",
                    f"  有效率: {stats['valid_ratio']*100:.2f}%",
                    f"  X坐标范围: [{stats['x_range'][0]:.3f}, {stats['x_range'][1]:.3f}]",
                    f"  Y坐标范围: [{stats['y_range'][0]:.3f}, {stats['y_range'][1]:.3f}]",
                    f"  Z坐标范围: [{stats['z_range'][0]:.3f}, {stats['z_range'][1]:.3f}]",
                    ""
                ])
        
        # 2. 滤波处理部分
        if 'filtering' in results:
            filt = results['filtering']
            report_lines.extend([
                "2. 点云滤波处理",
                "-" * 30,
                f"原始有效点数: {filt['original_valid_points']}",
                f"距离阈值滤波后: {filt['after_distance_filter']} (滤除 {filt['distance_filtered']})",
                f"孤立点滤波后: {filt['after_isolate_filter']} (滤除 {filt['isolate_filtered']})",
                f"K近邻滤波后: {filt['after_knn_filter']} (滤除 {filt['knn_filtered']})",
                f"最终保留率: {filt['final_valid_ratio']*100:.2f}%",
                ""
            ])
        
        # 3. 燃烧室分离部分
        if 'combustion_chamber' in results:
            chamber = results['combustion_chamber']
            
            if 'extraction' in chamber:
                ext = chamber['extraction']
                report_lines.extend([
                    "3. 燃烧室点云分离",
                    "-" * 30,
                    f"边界点数量: {ext.get('boundary_points_count', 0)}",
                    f"拟合圆心: ({ext['circle_center'][0]:.3f}, {ext['circle_center'][1]:.3f})",
                    f"拟合半径: {ext['circle_radius']:.3f}",
                    f"有效半径: {ext['effective_radius']:.3f}",
                    f"燃烧室点数: {ext['chamber_points_count']}",
                    ""
                ])
            
            if 'geometry' in chamber and 'error' not in chamber['geometry']:
                geo = chamber['geometry']
                report_lines.extend([
                    "燃烧室几何特性:",
                    f"  燃烧室深度: {geo['chamber_depth']:.4f}mm",
                    f"  投影面积: {geo['projection_area']:.6f}mm²",
                    f"  平均深度: {geo['average_depth']:.4f}mm",
                    f"  估算体积: {geo['estimated_volume']:.8f}mm³",
                    ""
                ])
        
        # 4. 图像分析部分
        if 'image_analysis' in results:
            # 添加图像处理器生成的报告
            image_report = self.image_processor.generate_analysis_report(results['image_analysis'])
            report_lines.extend(["4. " + image_report, ""])
        
        # 报告结尾
        report_lines.extend([
            "=" * 80,
            "分析完成",
            "输出文件:",
            "  - homework_process.log: 处理日志",
            "  - output/grayscale_image.png: 灰度投影图像",
            "  - output/edges.png: 边缘检测结果",
            "  - output/result_visualization.png: 可视化结果",
            "  - report/analysis_report.txt: 本报告文件",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def save_results(self, results: dict) -> None:
        """
        保存分析结果
        
        Args:
            results: 分析结果字典
        """
        # 保存完整报告
        report = self.generate_comprehensive_report(results)
        report_file = os.path.join(FILE_PATHS['report_dir'], 'analysis_report.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"分析报告已保存: {report_file}")
        
        # 保存数值结果（如果需要进一步处理）
        import json
        
        # 将numpy数组转换为列表以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        # 保存可序列化的结果（排除图像数据）
        serializable_results = {}
        for key, value in results.items():
            if key != 'image_analysis':  # 图像数据太大，不保存到JSON
                serializable_results[key] = convert_numpy(value)
            else:
                # 只保存图像分析的元数据
                serializable_results[key] = {
                    'projection': convert_numpy(value.get('projection', {})),
                    'edge_detection': convert_numpy(value.get('edge_detection', {})),
                    'contour_extraction': convert_numpy(value.get('contour_extraction', {}))
                }
        
        json_file = os.path.join(FILE_PATHS['output_dir'], 'analysis_results.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"数值结果已保存: {json_file}")


def main():
    """主函数"""
    analyzer = CombustionChamberAnalyzer()
    
    try:
        # 运行完整分析
        results = analyzer.run_complete_analysis()
        
        # 保存结果
        analyzer.save_results(results)
        
        # 打印报告摘要
        print("\n" + "=" * 60)
        print("分析完成！主要结果:")
        print("=" * 60)
        
        if 'filtering' in results:
            filt = results['filtering']
            print(f"滤波保留率: {filt['final_valid_ratio']*100:.2f}%")
        
        if 'combustion_chamber' in results and 'geometry' in results['combustion_chamber']:
            geo = results['combustion_chamber']['geometry']
            if 'error' not in geo:
                print(f"燃烧室深度: {geo['chamber_depth']:.4f}mm")
                print(f"估算体积: {geo['estimated_volume']:.8f}mm³")
        
        if 'image_analysis' in results and 'contour_extraction' in results['image_analysis']:
            contour = results['image_analysis']['contour_extraction']
            print(f"检测到轮廓: {contour.get('contour_count', 0)}个")
        
        print("\n详细报告请查看: report/analysis_report.txt")
        
    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)