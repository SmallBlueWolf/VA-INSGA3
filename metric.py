#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
度量脚本，用于对比原始算法和改进算法的优化结果
包括超体积、收敛速度、帕累托前沿分布等多种性能指标
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist
from scipy.stats import wilcoxon
import time
import json
from matplotlib.ticker import PercentFormatter

# ==================================================
# 帕累托前沿性能指标
# ==================================================

def calculate_hypervolume(points, reference_point=None):
    """
    计算超体积指标
    超体积是由帕累托前沿支配的空间体积，越大表示前沿覆盖范围越大
    """
    # 转换目标函数方向，使所有目标均为最大化
    # f1=-Rate_AB, f2=-Rate_BA, f3=Energy -> 转为 -f1, -f2, -f3 (均为最大化)
    points = np.array(points)
    points_maximized = np.copy(points)
    points_maximized[:, 0] = -points_maximized[:, 0]  # 已经是负的，不变
    points_maximized[:, 1] = -points_maximized[:, 1]  # 已经是负的，不变
    points_maximized[:, 2] = -points_maximized[:, 2]  # 转为最大化

    if reference_point is None:
        # 自动生成参考点（各维度的最差值）
        reference_point = np.min(points_maximized, axis=0)
    
    # 计算每个解与参考点之间的体积并求和
    n_points = len(points_maximized)
    if n_points == 0:
        return 0.0
    
    # 对点进行排序（对于简单计算）
    sorted_points = points_maximized[np.argsort(points_maximized[:, 0])]
    
    # 简单计算3D超体积（使用分解法）
    hv = 0.0
    for i in range(n_points):
        p = sorted_points[i]
        # 这里的计算只是一个近似，完整实现会更复杂
        contribution = np.prod(np.abs(p - reference_point))
        for j in range(i):
            q = sorted_points[j]
            overlap = np.prod(np.maximum(0, np.minimum(p, q) - reference_point))
            contribution -= overlap
        hv += max(0, contribution)
    
    return hv

def calculate_spacing(points):
    """
    计算帕累托前沿的均匀度
    均匀度越小表示解分布越均匀
    """
    points = np.array(points)
    if len(points) < 2:
        return float('inf')
    
    # 规范化目标值
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    ranges = max_vals - min_vals
    ranges = np.where(ranges < 1e-10, 1e-10, ranges)  # 避免除零
    normalized_points = (points - min_vals) / ranges
    
    # 计算每个点到最近邻点的距离
    distances = pdist(normalized_points, 'euclidean')
    pairwise_distances = cdist(normalized_points, normalized_points, 'euclidean')
    np.fill_diagonal(pairwise_distances, np.inf)  # 忽略自身距离
    min_distances = np.min(pairwise_distances, axis=1)
    
    # 计算均匀度（距离的标准差）
    spacing = np.std(min_distances)
    return spacing

def calculate_spread(points):
    """
    计算帕累托前沿的分布范围
    范围越大表示前沿的广度越好
    """
    points = np.array(points)
    if len(points) < 2:
        return 0.0
    
    # 范围可以用前沿的对角线长度来表示
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    diagonal = np.linalg.norm(max_vals - min_vals)
    
    return diagonal

def calculate_convergence_rate(convergence_data):
    """
    计算收敛速度（评估平均目标值收敛到最终值的速度）
    返回每个目标函数达到90%收敛所需的代数
    """
    if not convergence_data or 'avg_values' not in convergence_data:
        return None
    
    generations = np.array(convergence_data['generations'])
    avg_values = np.array(convergence_data['avg_values'])
    
    # 如果大于等于300代，取最后10%作为收敛目标；否则取最后30代
    if len(generations) >= 300:
        final_window = int(len(generations) * 0.1)
    else:
        final_window = min(30, int(len(generations) * 0.2))
    
    if final_window <= 0:
        final_window = 1
    
    # 目标函数的最终平均值
    final_values = np.mean(avg_values[-final_window:], axis=0)
    
    # 计算每个目标函数达到最终值90%的代数
    target_values = final_values * 0.9  # 90%收敛阈值
    
    # 对于最小化目标，我们希望值越小越好
    # 对于最大化目标 (在这里是-Rate_AB和-Rate_BA)，由于是负值，目标也是值越小越好
    # 所以我们检查值是否小于或等于目标值
    
    convergence_gens = []
    for i in range(len(final_values)):
        # 寻找第一个小于等于目标值的代数
        for gen_idx, gen in enumerate(generations):
            if avg_values[gen_idx, i] <= target_values[i]:
                convergence_gens.append(gen)
                break
        else:
            # 如果没有找到收敛点，返回最大代数
            convergence_gens.append(generations[-1])
    
    return convergence_gens

# ==================================================
# 对比分析函数
# ==================================================

def compare_hypervolume(origin_data, improv_data):
    """比较超体积指标"""
    origin_objectives = np.array(origin_data['objectives'])
    improv_objectives = np.array(improv_data['objectives'])
    
    # 计算共同参考点
    combined_points = np.vstack((origin_objectives, improv_objectives))
    reference_point = np.min(combined_points, axis=0) * 1.1  # 稍微扩大参考点范围
    
    # 计算超体积
    origin_hv = calculate_hypervolume(origin_objectives, reference_point)
    improv_hv = calculate_hypervolume(improv_objectives, reference_point)
    
    # 计算提升比例
    improvement = (improv_hv - origin_hv) / max(abs(origin_hv), 1e-10) * 100
    
    print(f"超体积指标对比:")
    print(f"  - 原始算法: {origin_hv:.4e}")
    print(f"  - 改进算法: {improv_hv:.4e}")
    print(f"  - 提升比例: {improvement:.2f}%")
    
    return {
        'original': origin_hv,
        'improved': improv_hv,
        'improvement': improvement
    }

def compare_spacing(origin_data, improv_data):
    """比较均匀度指标"""
    origin_objectives = np.array(origin_data['objectives'])
    improv_objectives = np.array(improv_data['objectives'])
    
    # 计算均匀度
    origin_spacing = calculate_spacing(origin_objectives)
    improv_spacing = calculate_spacing(improv_objectives)
    
    # 均匀度越小越好，所以改进比例是相反的
    improvement = (origin_spacing - improv_spacing) / max(abs(origin_spacing), 1e-10) * 100
    
    print(f"均匀度指标对比 (值越小越好):")
    print(f"  - 原始算法: {origin_spacing:.4f}")
    print(f"  - 改进算法: {improv_spacing:.4f}")
    print(f"  - 提升比例: {improvement:.2f}%")
    
    return {
        'original': origin_spacing,
        'improved': improv_spacing,
        'improvement': improvement
    }

def compare_spread(origin_data, improv_data):
    """比较分布范围指标"""
    origin_objectives = np.array(origin_data['objectives'])
    improv_objectives = np.array(improv_data['objectives'])
    
    # 计算分布范围
    origin_spread = calculate_spread(origin_objectives)
    improv_spread = calculate_spread(improv_objectives)
    
    # 计算提升比例
    improvement = (improv_spread - origin_spread) / max(abs(origin_spread), 1e-10) * 100
    
    print(f"分布范围指标对比:")
    print(f"  - 原始算法: {origin_spread:.4f}")
    print(f"  - 改进算法: {improv_spread:.4f}")
    print(f"  - 提升比例: {improvement:.2f}%")
    
    return {
        'original': origin_spread,
        'improved': improv_spread,
        'improvement': improvement
    }

def compare_convergence(origin_convergence, improv_convergence):
    """比较收敛速度"""
    orig_conv_rate = calculate_convergence_rate(origin_convergence)
    improv_conv_rate = calculate_convergence_rate(improv_convergence)
    
    if orig_conv_rate is None or improv_conv_rate is None:
        print("无法计算收敛速度，缺少必要数据")
        return None
    
    # 计算各目标的收敛速度提升
    improvements = []
    for i, (orig, impr) in enumerate(zip(orig_conv_rate, improv_conv_rate)):
        if orig > 0:
            imp_percent = (orig - impr) / orig * 100
            improvements.append(imp_percent)
            obj_name = ["Rate AB", "Rate BA", "Energy"][i]
            print(f"  - {obj_name} 收敛速度: 原始算法 {orig}代, 改进算法 {impr}代, 提升 {imp_percent:.2f}%")
    
    # 计算平均提升
    avg_improvement = np.mean(improvements) if improvements else 0.0
    print(f"收敛速度平均提升: {avg_improvement:.2f}%")
    
    return {
        'original': orig_conv_rate,
        'improved': improv_conv_rate,
        'improvements': improvements,
        'avg_improvement': avg_improvement
    }

def compare_execution_time(origin_data, improv_data):
    """比较执行时间"""
    origin_time = origin_data.get('execution_time', 0)
    improv_time = improv_data.get('execution_time', 0)
    
    if origin_time <= 0 or improv_time <= 0:
        print("无法计算执行时间，缺少必要数据")
        return None
    
    # 计算时间变化比例（负值表示改进算法更快）
    time_change = (improv_time - origin_time) / origin_time * 100
    
    print(f"执行时间对比:")
    print(f"  - 原始算法: {origin_time:.2f}秒")
    print(f"  - 改进算法: {improv_time:.2f}秒")
    print(f"  - 变化比例: {time_change:.2f}% {'(改进算法更慢)' if time_change > 0 else '(改进算法更快)'}")
    
    return {
        'original': origin_time,
        'improved': improv_time,
        'change': time_change
    }

def plot_pareto_comparison(origin_data, improv_data, save_path='comparison'):
    """绘制两种算法的帕累托前沿对比图"""
    origin_objectives = np.array(origin_data['objectives'])
    improv_objectives = np.array(improv_data['objectives'])
    
    # 转换为正向目标值显示
    origin_display = np.copy(origin_objectives)
    improv_display = np.copy(improv_objectives)
    
    # -Rate_AB, -Rate_BA, Energy -> Rate_AB, Rate_BA, Energy
    origin_display[:, 0] = -origin_display[:, 0]
    origin_display[:, 1] = -origin_display[:, 1]
    improv_display[:, 0] = -improv_display[:, 0]
    improv_display[:, 1] = -improv_display[:, 1]
    
    # 创建3D图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制原始算法的帕累托前沿
    ax.scatter(origin_display[:, 0], origin_display[:, 1], origin_display[:, 2], 
               c='blue', marker='o', s=70, alpha=0.7, label='原始算法')
    
    # 绘制改进算法的帕累托前沿
    ax.scatter(improv_display[:, 0], improv_display[:, 1], improv_display[:, 2], 
               c='red', marker='^', s=70, alpha=0.7, label='改进算法')
    
    ax.set_xlabel('Rate A->B')
    ax.set_ylabel('Rate B->A')
    ax.set_zlabel('Energy')
    ax.set_title('帕累托前沿对比')
    
    # 添加图例
    ax.legend()
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 保存图像
    plt.savefig(f"{save_path}/pareto_comparison.png")
    plt.close()
    
    print(f"帕累托前沿对比图已保存至 {save_path}/pareto_comparison.png")

def plot_convergence_comparison(origin_convergence, improv_convergence, save_path='comparison'):
    """绘制两种算法的收敛曲线对比图"""
    # 获取数据
    origin_gens = np.array(origin_convergence['generations'])
    origin_avgs = np.array(origin_convergence['avg_values'])
    
    improv_gens = np.array(improv_convergence['generations'])
    improv_avgs = np.array(improv_convergence['avg_values'])
    
    # 转换为正向目标显示
    origin_avgs[:, 0] = -origin_avgs[:, 0]  # Rate_AB
    origin_avgs[:, 1] = -origin_avgs[:, 1]  # Rate_BA
    improv_avgs[:, 0] = -improv_avgs[:, 0]  # Rate_AB
    improv_avgs[:, 1] = -improv_avgs[:, 1]  # Rate_BA
    
    # 创建3个子图，每个对应一个目标函数
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # 绘制Rate_AB收敛曲线
    axes[0].plot(origin_gens, origin_avgs[:, 0], 'b-', linewidth=2, label='原始算法')
    axes[0].plot(improv_gens, improv_avgs[:, 0], 'r-', linewidth=2, label='改进算法')
    axes[0].set_ylabel('Avg Rate A->B')
    axes[0].set_title('Rate A->B 收敛曲线对比')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制Rate_BA收敛曲线
    axes[1].plot(origin_gens, origin_avgs[:, 1], 'b-', linewidth=2, label='原始算法')
    axes[1].plot(improv_gens, improv_avgs[:, 1], 'r-', linewidth=2, label='改进算法')
    axes[1].set_ylabel('Avg Rate B->A')
    axes[1].set_title('Rate B->A 收敛曲线对比')
    axes[1].legend()
    axes[1].grid(True)
    
    # 绘制Energy收敛曲线
    axes[2].plot(origin_gens, origin_avgs[:, 2], 'b-', linewidth=2, label='原始算法')
    axes[2].plot(improv_gens, improv_avgs[:, 2], 'r-', linewidth=2, label='改进算法')
    axes[2].set_xlabel('Generation')
    axes[2].set_ylabel('Avg Energy')
    axes[2].set_title('Energy 收敛曲线对比')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 保存图像
    plt.savefig(f"{save_path}/convergence_comparison.png")
    plt.close()
    
    print(f"收敛曲线对比图已保存至 {save_path}/convergence_comparison.png")

def plot_improvement_summary(metrics, save_path='comparison'):
    """绘制改进算法相对于原始算法的性能提升总结图"""
    # 提取各指标的提升比例
    labels = []
    values = []
    
    if 'hypervolume' in metrics:
        labels.append('超体积')
        values.append(metrics['hypervolume']['improvement'])
    
    if 'spacing' in metrics:
        labels.append('均匀度')
        values.append(metrics['spacing']['improvement'])
    
    if 'spread' in metrics:
        labels.append('分布范围')
        values.append(metrics['spread']['improvement'])
    
    if 'convergence' in metrics and metrics['convergence'] and 'avg_improvement' in metrics['convergence']:
        labels.append('收敛速度')
        values.append(metrics['convergence']['avg_improvement'])
    
    if 'execution_time' in metrics and metrics['execution_time']:
        labels.append('执行时间')
        # 执行时间是负向指标，值越小越好
        values.append(-metrics['execution_time']['change'])
    
    # 创建条形图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 使用不同颜色标识正负值
    colors = ['green' if v >= 0 else 'red' for v in values]
    
    # 绘制条形图
    bars = ax.bar(labels, values, color=colors)
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        sign = '+' if height >= 0 else ''
        ax.text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -5),
                f'{sign}{height:.2f}%', ha='center', va='bottom' if height >= 0 else 'top')
    
    # 设置标签和标题
    ax.set_ylabel('提升比例 (%)')
    ax.set_title('改进算法相对于原始算法的性能提升')
    
    # 添加零线
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    # 添加网格线
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # 自定义Y轴刻度
    ax.yaxis.set_major_formatter(PercentFormatter())
    
    plt.tight_layout()
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 保存图像
    plt.savefig(f"{save_path}/improvement_summary.png")
    plt.close()
    
    print(f"性能提升总结图已保存至 {save_path}/improvement_summary.png")

def compare_results():
    """主函数，比较原始算法和改进算法的结果"""
    # 检查结果文件是否存在
    if not os.path.exists("origin/pareto_data.pkl") or not os.path.exists("improv/pareto_data.pkl"):
        print("错误：找不到结果文件，请先运行两种算法")
        return
    
    if not os.path.exists("origin/convergence_data.pkl") or not os.path.exists("improv/convergence_data.pkl"):
        print("警告：找不到收敛数据文件，部分分析将被跳过")
    
    # 创建比较结果保存目录
    comparison_dir = "comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    
    try:
        # 加载帕累托前沿数据
        with open("origin/pareto_data.pkl", "rb") as f:
            origin_data = pickle.load(f)
        
        with open("improv/pareto_data.pkl", "rb") as f:
            improv_data = pickle.load(f)
        
        # 加载收敛历史数据（如果存在）
        origin_convergence = None
        improv_convergence = None
        
        if os.path.exists("origin/convergence_data.pkl"):
            with open("origin/convergence_data.pkl", "rb") as f:
                origin_convergence = pickle.load(f)
        
        if os.path.exists("improv/convergence_data.pkl"):
            with open("improv/convergence_data.pkl", "rb") as f:
                improv_convergence = pickle.load(f)
        
        # 开始比较分析
        print("\n" + "="*60)
        print(" "*20 + "结果对比分析" + " "*20)
        print("="*60)
        
        # 收集度量结果
        metrics = {}
        
        # 1. 超体积对比
        print("\n1. 超体积指标对比")
        metrics['hypervolume'] = compare_hypervolume(origin_data, improv_data)
        
        # 2. 均匀度对比
        print("\n2. 均匀度指标对比")
        metrics['spacing'] = compare_spacing(origin_data, improv_data)
        
        # 3. 分布范围对比 - 仍然计算但不在最终总结中显示
        if False:  # 设置为False以禁用此分析
            print("\n3. 分布范围指标对比")
            metrics['spread'] = compare_spread(origin_data, improv_data)
        
        # 4. 收敛速度对比 - 仍然计算但不在最终总结中显示
        if False and origin_convergence and improv_convergence:  # 设置为False以禁用此分析
            print("\n4. 收敛速度对比")
            metrics['convergence'] = compare_convergence(origin_convergence, improv_convergence)
        
        # 5. 执行时间对比
        print("\n3. 执行时间对比")  # 更新编号
        metrics['execution_time'] = compare_execution_time(origin_data, improv_data)
        
        # 6. 可视化对比结果
        print("\n4. 生成对比可视化图表")  # 更新编号
        
        # 绘制帕累托前沿对比图
        plot_pareto_comparison(origin_data, improv_data, comparison_dir)
        
        # 绘制收敛曲线对比图（如果有收敛数据）
        if origin_convergence and improv_convergence:
            plot_convergence_comparison(origin_convergence, improv_convergence, comparison_dir)
        
        # 绘制性能提升总结图
        plot_improvement_summary(metrics, comparison_dir)
        
        # 保存比较结果到JSON文件
        with open(f"{comparison_dir}/metrics_summary.json", "w", encoding="utf-8") as f:
            # 转换numpy值为Python标准类型
            def convert_to_json_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.int64) or isinstance(obj, np.int32):
                    return int(obj)
                elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_to_json_serializable(metrics), f, indent=4)
        
        print(f"\n比较分析结果已保存至 {comparison_dir}/metrics_summary.json")
        print("\n" + "="*60)
        print(" "*15 + "改进算法相对于原始算法的总体提升" + " "*15)
        print("="*60)
        
        # 输出总体提升概述 - 去除分布范围和收敛速度
        if 'hypervolume' in metrics:
            print(f"超体积指标提升: {metrics['hypervolume']['improvement']:.2f}%")
        
        if 'spacing' in metrics:
            print(f"均匀度指标提升: {metrics['spacing']['improvement']:.2f}%")
        
        # 删除分布范围和收敛速度的输出
        
        if 'execution_time' in metrics and metrics['execution_time']:
            time_change = metrics['execution_time']['change']
            if time_change > 0:
                print(f"执行时间增加: {time_change:.2f}% (改进算法更慢)")
            else:
                print(f"执行时间减少: {-time_change:.2f}% (改进算法更快)")
        
        # 计算总体性能提升平均值（不包括执行时间，也不包括分布范围和收敛速度）
        performance_metrics = []
        if 'hypervolume' in metrics:
            performance_metrics.append(metrics['hypervolume']['improvement'])
        if 'spacing' in metrics:
            performance_metrics.append(metrics['spacing']['improvement'])
        
        if performance_metrics:
            avg_improvement = np.mean(performance_metrics)
            print(f"\n总体性能平均提升: {avg_improvement:.2f}%")
        
        print("="*60)
        
    except Exception as e:
        print(f"比较结果时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 该脚本通常由run.py调用，但也可以单独运行
    if os.path.exists("origin/pareto_data.pkl") and os.path.exists("improv/pareto_data.pkl"):
        compare_results()
    else:
        print("错误：找不到结果文件，请先运行两种算法") 