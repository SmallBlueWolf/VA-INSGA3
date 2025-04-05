# Placeholder for visualization functions 

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os

from config import NUM_UAV_A, NUM_UAV_B, REC_A_IDX, REC_B_IDX, X_BOUNDS_A, X_BOUNDS_B
from problem import extract_uav_info, INITIAL_POSITIONS_A, INITIAL_POSITIONS_B

# 确保图像保存目录存在
if not os.path.exists('plots'):
    os.makedirs('plots')

def plot_pareto_front(population, gen=-1, result_dir="plots"):
    """绘制帕累托前沿 (3D)
    目标: (-Rate AB, -Rate BA, Energy)
    绘制: (Rate AB, Rate BA, Energy)
    """
    front = np.array([ind.fitness.values for ind in population if ind.fitness.valid])
    if front.size == 0:
        print("没有有效的个体可以绘制帕累托前沿。")
        return

    # 将目标转换回原始形式 (Rate AB, Rate BA, Energy)
    rate_ab = -front[:, 0] / 1e6 # 转换为 Mbps
    rate_ba = -front[:, 1] / 1e6 # 转换为 Mbps
    energy = front[:, 2] / 1e8   # 转换为 10^8 J

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 使用颜色映射或其他方式区分帕累托前沿点
    scatter = ax.scatter(rate_ab, rate_ba, energy, c=energy, cmap='viridis', marker='o')

    ax.set_xlabel('Rate A->B (Mbps)')
    ax.set_ylabel('Rate B->A (Mbps)')
    ax.set_zlabel('Total Energy (1e8 J)')
    ax.set_title(f'Pareto Front at Generation {gen}' if gen >= 0 else 'Final Pareto Front')

    # 添加色条
    fig.colorbar(scatter, label='Total Energy (1e8 J)')

    plt.tight_layout()
    
    # 确保目录存在
    plots_dir = f"{result_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    file_path = f"{plots_dir}/pareto_front_gen_{gen}.png" if gen >= 0 else f"{plots_dir}/final_pareto_front.png"
    plt.savefig(file_path)
    print(f"帕累托前沿图已保存至 {file_path}")
    # plt.show() # 取消注释以显示图像
    plt.close(fig)

def plot_convergence(logbook, result_dir="plots"):
    """绘制收敛曲线 (各目标的平均值)"""
    gen = logbook.select("gen")
    avg_values = np.array(logbook.select("avg"))
    avg_rate_ab = -avg_values[:, 0] / 1e6  # 第一个目标的平均值，转为正Mbps
    avg_rate_ba = -avg_values[:, 1] / 1e6  # 第二个目标的平均值，转为正Mbps
    avg_energy = avg_values[:, 2] / 1e8    # 第三个目标的平均值，转为1e8 J

    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax[0].plot(gen, avg_rate_ab, marker='o', linestyle='-', label='Avg Rate A->B')
    ax[0].set_ylabel('Avg Rate A->B (Mbps)')
    ax[0].set_title('Convergence Plot')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(gen, avg_rate_ba, marker='s', linestyle='--', label='Avg Rate B->A', color='orange')
    ax[1].set_ylabel('Avg Rate B->A (Mbps)')
    ax[1].legend()
    ax[1].grid(True)

    ax[2].plot(gen, avg_energy, marker='^', linestyle=':', label='Avg Total Energy', color='green')
    ax[2].set_ylabel('Avg Total Energy (1e8 J)')
    ax[2].set_xlabel('Generation')
    ax[2].legend()
    ax[2].grid(True)

    plt.tight_layout()
    
    # 确保目录存在
    plots_dir = f"{result_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    file_path = f"{plots_dir}/convergence.png"
    plt.savefig(file_path)
    print(f"收敛曲线图已保存至 {file_path}")
    # plt.show()
    plt.close(fig)

def plot_deployment(individual, gen, result_dir="plots"):
    """绘制单个个体的无人机部署情况 (3D)"""
    if INITIAL_POSITIONS_A is None or INITIAL_POSITIONS_B is None:
        print(f"Warning: 无法绘制部署图(gen={gen})，初始位置未设置。")
        return

    # 提取信息
    (pos_a, _, pos_b, _, pos_send_a, _, pos_rec_a,
     pos_send_b, _, pos_rec_b, rec_a_idx, rec_b_idx) = extract_uav_info(individual)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制初始位置和轨迹 (A群)
    for i in range(NUM_UAV_A):
        ax.plot([INITIAL_POSITIONS_A[i, 0], pos_a[i, 0]],
                [INITIAL_POSITIONS_A[i, 1], pos_a[i, 1]],
                [INITIAL_POSITIONS_A[i, 2], pos_a[i, 2]], 'b:', alpha=0.5) # 轨迹
        ax.scatter(INITIAL_POSITIONS_A[i, 0], INITIAL_POSITIONS_A[i, 1], INITIAL_POSITIONS_A[i, 2],
                   marker='d', color='blue', alpha=0.6, label='Initial A' if i == 0 else "") # 初始点

    # 绘制初始位置和轨迹 (B群)
    for i in range(NUM_UAV_B):
        ax.plot([INITIAL_POSITIONS_B[i, 0], pos_b[i, 0]],
                [INITIAL_POSITIONS_B[i, 1], pos_b[i, 1]],
                [INITIAL_POSITIONS_B[i, 2], pos_b[i, 2]], 'g:', alpha=0.5) # 轨迹
        ax.scatter(INITIAL_POSITIONS_B[i, 0], INITIAL_POSITIONS_B[i, 1], INITIAL_POSITIONS_B[i, 2],
                   marker='d', color='green', alpha=0.6, label='Initial B' if i == 0 else "") # 初始点

    # 绘制最终位置 (发送端 A)
    if len(pos_send_a) > 0:
        ax.scatter(pos_send_a[:, 0], pos_send_a[:, 1], pos_send_a[:, 2],
                   marker='o', color='cyan', s=50, label='Sender A')

    # 绘制最终位置 (发送端 B)
    if len(pos_send_b) > 0:
        ax.scatter(pos_send_b[:, 0], pos_send_b[:, 1], pos_send_b[:, 2],
                   marker='o', color='magenta', s=50, label='Sender B')

    # 绘制最终位置 (接收端 A)
    ax.scatter(pos_rec_a[0], pos_rec_a[1], pos_rec_a[2],
               marker='*', color='red', s=150, label=f'Receiver A (idx {rec_b_idx})')

    # 绘制最终位置 (接收端 B)
    ax.scatter(pos_rec_b[0], pos_rec_b[1], pos_rec_b[2],
               marker='*', color='yellow', s=150, label=f'Receiver B (idx {rec_a_idx})')

    # 设置坐标轴范围和标签
    all_x = np.concatenate((INITIAL_POSITIONS_A[:, 0], INITIAL_POSITIONS_B[:, 0], pos_a[:, 0], pos_b[:, 0]))
    all_y = np.concatenate((INITIAL_POSITIONS_A[:, 1], INITIAL_POSITIONS_B[:, 1], pos_a[:, 1], pos_b[:, 1]))
    all_z = np.concatenate((INITIAL_POSITIONS_A[:, 2], INITIAL_POSITIONS_B[:, 2], pos_a[:, 2], pos_b[:, 2]))

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # 保持比例，否则距离关系会失真
    max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
    mid_x = (all_x.max()+all_x.min()) * 0.5
    mid_y = (all_y.max()+all_y.min()) * 0.5
    mid_z = (all_z.max()+all_z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_title(f'UAV Deployment at Generation {gen}')
    ax.legend()
    plt.tight_layout()
    
    # 确保目录存在
    plots_dir = f"{result_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    file_path = f"{plots_dir}/deployment_gen_{gen}.png"
    plt.savefig(file_path)
    print(f"无人机部署图 (Gen {gen}) 已保存至 {file_path}")
    # plt.show()
    plt.close(fig) 