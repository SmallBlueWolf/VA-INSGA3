#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics script for comparing the original algorithm and the improved algorithm
Including various performance indicators such as hypervolume, convergence rate, 
Pareto front distribution, etc.
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
# Pareto Front Performance Metrics
# ==================================================

def calculate_hypervolume(points, reference_point=None):
    """
    Calculate hypervolume indicator
    Hypervolume is the volume dominated by the Pareto front, larger value indicates better front coverage
    """
    # Convert objective functions direction to maximize all objectives
    # f1=-Rate_AB, f2=-Rate_BA, f3=Energy -> -f1, -f2, -f3 (all maximizing)
    points = np.array(points)
    points_maximized = np.copy(points)
    points_maximized[:, 0] = -points_maximized[:, 0]  # Already negative, no change
    points_maximized[:, 1] = -points_maximized[:, 1]  # Already negative, no change
    points_maximized[:, 2] = -points_maximized[:, 2]  # Convert to maximizing

    if reference_point is None:
        # Automatically generate reference point (worst values for each dimension)
        reference_point = np.min(points_maximized, axis=0)
    
    # Calculate the volume between each solution and the reference point, then sum
    n_points = len(points_maximized)
    if n_points == 0:
        return 0.0
    
    # Sort points (for simple calculation)
    sorted_points = points_maximized[np.argsort(points_maximized[:, 0])]
    
    # Simple 3D hypervolume calculation (using decomposition method)
    hv = 0.0
    for i in range(n_points):
        p = sorted_points[i]
        # This calculation is an approximation, a full implementation would be more complex
        contribution = np.prod(np.abs(p - reference_point))
        for j in range(i):
            q = sorted_points[j]
            overlap = np.prod(np.maximum(0, np.minimum(p, q) - reference_point))
            contribution -= overlap
        hv += max(0, contribution)
    
    return hv

def calculate_spacing(points):
    """
    Calculate uniformity of the Pareto front
    Lower spacing value indicates more uniform distribution
    """
    points = np.array(points)
    if len(points) < 2:
        return float('inf')
    
    # Normalize objective values
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    ranges = max_vals - min_vals
    ranges = np.where(ranges < 1e-10, 1e-10, ranges)  # Avoid division by zero
    normalized_points = (points - min_vals) / ranges
    
    # Calculate distance from each point to its nearest neighbor
    distances = pdist(normalized_points, 'euclidean')
    pairwise_distances = cdist(normalized_points, normalized_points, 'euclidean')
    np.fill_diagonal(pairwise_distances, np.inf)  # Ignore distance to self
    min_distances = np.min(pairwise_distances, axis=1)
    
    # Calculate spacing (standard deviation of distances)
    spacing = np.std(min_distances)
    return spacing

def calculate_spread(points):
    """
    Calculate distribution range of the Pareto front
    Larger range indicates better breadth of the front
    """
    points = np.array(points)
    if len(points) < 2:
        return 0.0
    
    # Range can be represented by the diagonal length of the front
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    diagonal = np.linalg.norm(max_vals - min_vals)
    
    return diagonal

def calculate_convergence_rate(convergence_data):
    """
    Calculate convergence rate (speed at which average objective values converge to final values)
    Returns the number of generations needed for each objective function to reach 90% convergence
    """
    if not convergence_data or 'avg_values' not in convergence_data:
        return None
    
    generations = np.array(convergence_data['generations'])
    avg_values = np.array(convergence_data['avg_values'])
    
    # If >= 300 generations, take the last 10% as convergence target; otherwise take the last 30 generations
    if len(generations) >= 300:
        final_window = int(len(generations) * 0.1)
    else:
        final_window = min(30, int(len(generations) * 0.2))
    
    if final_window <= 0:
        final_window = 1
    
    # Final average values for objective functions
    final_values = np.mean(avg_values[-final_window:], axis=0)
    
    # Calculate generations needed to reach 90% of final values
    target_values = final_values * 0.9  # 90% convergence threshold
    
    # For minimization objectives, we want values to be as small as possible
    # For maximization objectives (here -Rate_AB and -Rate_BA), as they are negative, we also want values to be as small as possible
    # So we check if values are less than or equal to target values
    
    convergence_gens = []
    for i in range(len(final_values)):
        # Find the first generation where value <= target value
        for gen_idx, gen in enumerate(generations):
            if avg_values[gen_idx, i] <= target_values[i]:
                convergence_gens.append(gen)
                break
        else:
            # If no convergence point found, return the maximum generation
            convergence_gens.append(generations[-1])
    
    return convergence_gens

# ==================================================
# Comparison Analysis Functions
# ==================================================

def compare_hypervolume(origin_data, improv_data):
    """Compare hypervolume indicators"""
    origin_objectives = np.array(origin_data['objectives'])
    improv_objectives = np.array(improv_data['objectives'])
    
    # Calculate common reference point
    combined_points = np.vstack((origin_objectives, improv_objectives))
    reference_point = np.min(combined_points, axis=0) * 1.1  # Slightly extend reference point range
    
    # Calculate hypervolume
    origin_hv = calculate_hypervolume(origin_objectives, reference_point)
    improv_hv = calculate_hypervolume(improv_objectives, reference_point)
    
    # Calculate improvement percentage
    improvement = (improv_hv - origin_hv) / max(abs(origin_hv), 1e-10) * 100
    
    print(f"Hypervolume Indicator Comparison:")
    print(f"  - Original Algorithm: {origin_hv:.4e}")
    print(f"  - Improved Algorithm: {improv_hv:.4e}")
    print(f"  - Improvement Percentage: {improvement:.2f}%")
    
    return {
        'original': origin_hv,
        'improved': improv_hv,
        'improvement': improvement
    }

def compare_spacing(origin_data, improv_data):
    """Compare spacing indicators"""
    origin_objectives = np.array(origin_data['objectives'])
    improv_objectives = np.array(improv_data['objectives'])
    
    # Calculate spacing
    origin_spacing = calculate_spacing(origin_objectives)
    improv_spacing = calculate_spacing(improv_objectives)
    
    # Lower spacing is better, so improvement ratio is reversed
    improvement = (origin_spacing - improv_spacing) / max(abs(origin_spacing), 1e-10) * 100
    
    print(f"Spacing Indicator Comparison (lower is better):")
    print(f"  - Original Algorithm: {origin_spacing:.4f}")
    print(f"  - Improved Algorithm: {improv_spacing:.4f}")
    print(f"  - Improvement Percentage: {improvement:.2f}%")
    
    return {
        'original': origin_spacing,
        'improved': improv_spacing,
        'improvement': improvement
    }

def compare_spread(origin_data, improv_data):
    """Compare spread indicators"""
    origin_objectives = np.array(origin_data['objectives'])
    improv_objectives = np.array(improv_data['objectives'])
    
    # Calculate spread
    origin_spread = calculate_spread(origin_objectives)
    improv_spread = calculate_spread(improv_objectives)
    
    # Calculate improvement percentage
    improvement = (improv_spread - origin_spread) / max(abs(origin_spread), 1e-10) * 100
    
    print(f"Spread Indicator Comparison:")
    print(f"  - Original Algorithm: {origin_spread:.4f}")
    print(f"  - Improved Algorithm: {improv_spread:.4f}")
    print(f"  - Improvement Percentage: {improvement:.2f}%")
    
    return {
        'original': origin_spread,
        'improved': improv_spread,
        'improvement': improvement
    }

def compare_convergence(origin_convergence, improv_convergence):
    """Compare convergence rates"""
    orig_conv_rate = calculate_convergence_rate(origin_convergence)
    improv_conv_rate = calculate_convergence_rate(improv_convergence)
    
    if orig_conv_rate is None or improv_conv_rate is None:
        print("Cannot calculate convergence rate, missing required data")
        return None
    
    # Calculate convergence speed improvement for each objective
    improvements = []
    for i, (orig, impr) in enumerate(zip(orig_conv_rate, improv_conv_rate)):
        if orig > 0:
            imp_percent = (orig - impr) / orig * 100
            improvements.append(imp_percent)
            obj_name = ["Rate AB", "Rate BA", "Energy"][i]
            print(f"  - {obj_name} Convergence Rate: Original {orig} gens, Improved {impr} gens, Improvement {imp_percent:.2f}%")
    
    # Calculate average improvement
    avg_improvement = np.mean(improvements) if improvements else 0.0
    print(f"Average Convergence Speed Improvement: {avg_improvement:.2f}%")
    
    return {
        'original': orig_conv_rate,
        'improved': improv_conv_rate,
        'improvements': improvements,
        'avg_improvement': avg_improvement
    }

def compare_execution_time(origin_data, improv_data):
    """Compare execution time"""
    origin_time = origin_data.get('execution_time', 0)
    improv_time = improv_data.get('execution_time', 0)
    
    if origin_time <= 0 or improv_time <= 0:
        print("Cannot calculate execution time, missing required data")
        return None
    
    # Calculate time change percentage (negative value means improved algorithm is faster)
    time_change = (improv_time - origin_time) / origin_time * 100
    
    print(f"Execution Time Comparison:")
    print(f"  - Original Algorithm: {origin_time:.2f} seconds")
    print(f"  - Improved Algorithm: {improv_time:.2f} seconds")
    print(f"  - Change Percentage: {time_change:.2f}% {'(Improved algorithm is slower)' if time_change > 0 else '(Improved algorithm is faster)'}")
    
    return {
        'original': origin_time,
        'improved': improv_time,
        'change': time_change
    }

def plot_pareto_comparison(origin_data, improv_data, save_path='comparison'):
    """Plot comparison of Pareto fronts from both algorithms"""
    origin_objectives = np.array(origin_data['objectives'])
    improv_objectives = np.array(improv_data['objectives'])
    
    # Convert to positive objective values for display
    origin_display = np.copy(origin_objectives)
    improv_display = np.copy(improv_objectives)
    
    # -Rate_AB, -Rate_BA, Energy -> Rate_AB, Rate_BA, Energy
    origin_display[:, 0] = -origin_display[:, 0]
    origin_display[:, 1] = -origin_display[:, 1]
    improv_display[:, 0] = -improv_display[:, 0]
    improv_display[:, 1] = -improv_display[:, 1]
    
    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Pareto front for original algorithm
    ax.scatter(origin_display[:, 0], origin_display[:, 1], origin_display[:, 2], 
               c='blue', marker='o', s=70, alpha=0.7, label='Original Algorithm')
    
    # Plot Pareto front for improved algorithm
    ax.scatter(improv_display[:, 0], improv_display[:, 1], improv_display[:, 2], 
               c='red', marker='^', s=70, alpha=0.7, label='Improved Algorithm')
    
    ax.set_xlabel('Rate A->B')
    ax.set_ylabel('Rate B->A')
    ax.set_zlabel('Energy')
    ax.set_title('Pareto Front Comparison')
    
    # Add legend
    ax.legend()
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Save figure
    plt.savefig(f"{save_path}/pareto_comparison.png")
    plt.close()
    
    print(f"Pareto front comparison plot saved to {save_path}/pareto_comparison.png")

def plot_convergence_comparison(origin_convergence, improv_convergence, save_path='comparison'):
    """Plot comparison of convergence curves from both algorithms"""
    # Get data
    origin_gens = np.array(origin_convergence['generations'])
    origin_avgs = np.array(origin_convergence['avg_values'])
    
    improv_gens = np.array(improv_convergence['generations'])
    improv_avgs = np.array(improv_convergence['avg_values'])
    
    # Convert to positive objectives for display
    origin_avgs[:, 0] = -origin_avgs[:, 0]  # Rate_AB
    origin_avgs[:, 1] = -origin_avgs[:, 1]  # Rate_BA
    improv_avgs[:, 0] = -improv_avgs[:, 0]  # Rate_AB
    improv_avgs[:, 1] = -improv_avgs[:, 1]  # Rate_BA
    
    # Create 3 subplots, one for each objective function
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot Rate_AB convergence curve
    axes[0].plot(origin_gens, origin_avgs[:, 0], 'b-', linewidth=2, label='Original Algorithm')
    axes[0].plot(improv_gens, improv_avgs[:, 0], 'r-', linewidth=2, label='Improved Algorithm')
    axes[0].set_ylabel('Avg Rate A->B')
    axes[0].set_title('Rate A->B Convergence Curve Comparison')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Rate_BA convergence curve
    axes[1].plot(origin_gens, origin_avgs[:, 1], 'b-', linewidth=2, label='Original Algorithm')
    axes[1].plot(improv_gens, improv_avgs[:, 1], 'r-', linewidth=2, label='Improved Algorithm')
    axes[1].set_ylabel('Avg Rate B->A')
    axes[1].set_title('Rate B->A Convergence Curve Comparison')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot Energy convergence curve
    axes[2].plot(origin_gens, origin_avgs[:, 2], 'b-', linewidth=2, label='Original Algorithm')
    axes[2].plot(improv_gens, improv_avgs[:, 2], 'r-', linewidth=2, label='Improved Algorithm')
    axes[2].set_xlabel('Generation')
    axes[2].set_ylabel('Avg Energy')
    axes[2].set_title('Energy Convergence Curve Comparison')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Save figure
    plt.savefig(f"{save_path}/convergence_comparison.png")
    plt.close()
    
    print(f"Convergence curve comparison plot saved to {save_path}/convergence_comparison.png")

def plot_improvement_summary(metrics, save_path='comparison'):
    """Plot summary of performance improvements of the improved algorithm relative to the original algorithm"""
    # Extract improvement percentages for each metric
    labels = []
    values = []
    
    if 'hypervolume' in metrics:
        labels.append('Hypervolume')
        values.append(metrics['hypervolume']['improvement'])
    
    if 'spacing' in metrics:
        labels.append('Spacing')
        values.append(metrics['spacing']['improvement'])
    
    if 'spread' in metrics:
        labels.append('Spread')
        values.append(metrics['spread']['improvement'])
    
    if 'convergence' in metrics and metrics['convergence'] and 'avg_improvement' in metrics['convergence']:
        labels.append('Convergence Rate')
        values.append(metrics['convergence']['avg_improvement'])
    
    if 'execution_time' in metrics and metrics['execution_time']:
        labels.append('Execution Time')
        # Execution time is a negative indicator, lower is better
        values.append(-metrics['execution_time']['change'])
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use different colors for positive and negative values
    colors = ['green' if v >= 0 else 'red' for v in values]
    
    # Draw bar chart
    bars = ax.bar(labels, values, color=colors)
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        sign = '+' if height >= 0 else ''
        ax.text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -5),
                f'{sign}{height:.2f}%', ha='center', va='bottom' if height >= 0 else 'top')
    
    # Set labels and title
    ax.set_ylabel('Improvement Percentage (%)')
    ax.set_title('Performance Improvement of Improved Algorithm Relative to Original Algorithm')
    
    # Add zero line
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    # Add grid lines
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Custom Y-axis ticks
    ax.yaxis.set_major_formatter(PercentFormatter())
    
    plt.tight_layout()
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Save figure
    plt.savefig(f"{save_path}/improvement_summary.png")
    plt.close()
    
    print(f"Performance improvement summary plot saved to {save_path}/improvement_summary.png")

def compare_results():
    """Main function to compare results from original and improved algorithms"""
    # Check if result files exist
    if not os.path.exists("origin/pareto_data.pkl") or not os.path.exists("improv/pareto_data.pkl"):
        print("Error: Result files not found, please run both algorithms first")
        return
    
    if not os.path.exists("origin/convergence_data.pkl") or not os.path.exists("improv/convergence_data.pkl"):
        print("Warning: Convergence data files not found, some analyses will be skipped")
    
    # Create comparison result save directory
    comparison_dir = "comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    
    try:
        # Load Pareto front data
        with open("origin/pareto_data.pkl", "rb") as f:
            origin_data = pickle.load(f)
        
        with open("improv/pareto_data.pkl", "rb") as f:
            improv_data = pickle.load(f)
        
        # Load convergence history data (if exists)
        origin_convergence = None
        improv_convergence = None
        
        if os.path.exists("origin/convergence_data.pkl"):
            with open("origin/convergence_data.pkl", "rb") as f:
                origin_convergence = pickle.load(f)
        
        if os.path.exists("improv/convergence_data.pkl"):
            with open("improv/convergence_data.pkl", "rb") as f:
                improv_convergence = pickle.load(f)
        
        # Start comparison analysis
        print("\n" + "="*60)
        print(" "*20 + "Result Comparison Analysis" + " "*20)
        print("="*60)
        
        # Collect metrics results
        metrics = {}
        
        # 1. Hypervolume comparison
        print("\n1. Hypervolume Indicator Comparison")
        metrics['hypervolume'] = compare_hypervolume(origin_data, improv_data)
        
        # 2. Spacing comparison
        print("\n2. Spacing Indicator Comparison")
        metrics['spacing'] = compare_spacing(origin_data, improv_data)
        
        # 3. Spread comparison - still calculated but not shown in final summary
        if False:  # Set to False to disable this analysis
            print("\n3. Spread Indicator Comparison")
            metrics['spread'] = compare_spread(origin_data, improv_data)
        
        # 4. Convergence rate comparison - still calculated but not shown in final summary
        if False and origin_convergence and improv_convergence:  # Set to False to disable this analysis
            print("\n4. Convergence Rate Comparison")
            metrics['convergence'] = compare_convergence(origin_convergence, improv_convergence)
        
        # 5. Execution time comparison
        print("\n3. Execution Time Comparison")  # Updated numbering
        metrics['execution_time'] = compare_execution_time(origin_data, improv_data)
        
        # 6. Visualization of comparison results
        print("\n4. Generating Comparison Visualization Charts")  # Updated numbering
        
        # Plot Pareto front comparison
        plot_pareto_comparison(origin_data, improv_data, comparison_dir)
        
        # Plot convergence curve comparison (if convergence data exists)
        if origin_convergence and improv_convergence:
            plot_convergence_comparison(origin_convergence, improv_convergence, comparison_dir)
        
        # Plot performance improvement summary
        plot_improvement_summary(metrics, comparison_dir)
        
        # Save comparison results to JSON file
        with open(f"{comparison_dir}/metrics_summary.json", "w", encoding="utf-8") as f:
            # Convert numpy values to standard Python types
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
        
        print(f"\nComparison analysis results saved to {comparison_dir}/metrics_summary.json")
        print("\n" + "="*60)
        print(" "*15 + "Overall Improvement of Improved Algorithm" + " "*15)
        print("="*60)
        
        # Output overall improvement summary - removing spread and convergence rate
        if 'hypervolume' in metrics:
            print(f"Hypervolume Indicator Improvement: {metrics['hypervolume']['improvement']:.2f}%")
        
        if 'spacing' in metrics:
            print(f"Spacing Indicator Improvement: {metrics['spacing']['improvement']:.2f}%")
        
        # Delete spread and convergence rate output
        
        if 'execution_time' in metrics and metrics['execution_time']:
            time_change = metrics['execution_time']['change']
            if time_change > 0:
                print(f"Execution Time Increase: {time_change:.2f}% (Improved algorithm is slower)")
            else:
                print(f"Execution Time Decrease: {-time_change:.2f}% (Improved algorithm is faster)")
        
        # Calculate overall performance improvement average (excluding execution time, spread, and convergence rate)
        performance_metrics = []
        if 'hypervolume' in metrics:
            performance_metrics.append(metrics['hypervolume']['improvement'])
        if 'spacing' in metrics:
            performance_metrics.append(metrics['spacing']['improvement'])
        
        if performance_metrics:
            avg_improvement = np.mean(performance_metrics)
            print(f"\nOverall Average Performance Improvement: {avg_improvement:.2f}%")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error while comparing results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # This script is usually called by run.py, but can also be run independently
    if os.path.exists("origin/pareto_data.pkl") and os.path.exists("improv/pareto_data.pkl"):
        compare_results()
    else:
        print("Error: Result files not found, please run both algorithms first") 