# Placeholder for visualization functions 

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os

from config import NUM_UAV_A, NUM_UAV_B, REC_A_IDX, REC_B_IDX, X_BOUNDS_A, X_BOUNDS_B
from problem import extract_uav_info, INITIAL_POSITIONS_A, INITIAL_POSITIONS_B

from matplotlib import font_manager as fm, rcParams

# Remove Chinese font settings
# plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
# plt.rcParams['axes.unicode_minus']=False   

# Ensure plot directory exists
if not os.path.exists('plots'):
    os.makedirs('plots')

def plot_pareto_front(population, gen=-1, result_dir="plots"):
    """Draw Pareto front (3D)
    Objectives: (-Rate AB, -Rate BA, Energy)
    Plot: (Rate AB, Rate BA, Energy)
    """
    front = np.array([ind.fitness.values for ind in population if ind.fitness.valid])
    if front.size == 0:
        print("No valid individuals to plot Pareto front.")
        return

    # Convert objectives back to original form (Rate AB, Rate BA, Energy)
    rate_ab = -front[:, 0] / 1e6 # Convert to Mbps
    rate_ba = -front[:, 1] / 1e6 # Convert to Mbps
    energy = front[:, 2] / 1e8   # Convert to 10^8 J

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Use color mapping or other methods to distinguish Pareto front points
    scatter = ax.scatter(rate_ab, rate_ba, energy, c=energy, cmap='viridis', marker='o')

    ax.set_xlabel('Rate A->B (Mbps)')
    ax.set_ylabel('Rate B->A (Mbps)')
    ax.set_zlabel('Total Energy (1e8 J)')
    ax.set_title(f'Pareto Front at Generation {gen}' if gen >= 0 else 'Final Pareto Front')

    # Add color bar
    fig.colorbar(scatter, label='Total Energy (1e8 J)')

    plt.tight_layout()
    
    # Ensure directory exists
    plots_dir = f"{result_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    file_path = f"{plots_dir}/pareto_front_gen_{gen}.png" if gen >= 0 else f"{plots_dir}/final_pareto_front.png"
    plt.savefig(file_path)
    print(f"Pareto front plot saved to {file_path}")
    # plt.show() # Uncomment to display image
    plt.close(fig)

def plot_convergence(logbook, result_dir="plots"):
    """Plot convergence curves (average values for each objective)"""
    gen = logbook.select("gen")
    avg_values = np.array(logbook.select("avg"))
    avg_rate_ab = -avg_values[:, 0] / 1e6  # Average of first objective, converted to positive Mbps
    avg_rate_ba = -avg_values[:, 1] / 1e6  # Average of second objective, converted to positive Mbps
    avg_energy = avg_values[:, 2] / 1e8    # Average of third objective, converted to 1e8 J

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
    
    # Ensure directory exists
    plots_dir = f"{result_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    file_path = f"{plots_dir}/convergence.png"
    plt.savefig(file_path)
    print(f"Convergence plot saved to {file_path}")
    # plt.show()
    plt.close(fig)

def plot_deployment(individual, gen, result_dir="plots"):
    """Plot UAV deployment for a single individual (3D)"""
    if INITIAL_POSITIONS_A is None or INITIAL_POSITIONS_B is None:
        print(f"Warning: Cannot plot deployment (gen={gen}), initial positions not set.")
        return

    # Extract information
    (pos_a, _, pos_b, _, pos_send_a, _, pos_rec_a,
     pos_send_b, _, pos_rec_b, rec_a_idx, rec_b_idx) = extract_uav_info(individual)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot initial positions and trajectories (Group A)
    for i in range(NUM_UAV_A):
        ax.plot([INITIAL_POSITIONS_A[i, 0], pos_a[i, 0]],
                [INITIAL_POSITIONS_A[i, 1], pos_a[i, 1]],
                [INITIAL_POSITIONS_A[i, 2], pos_a[i, 2]], 'b:', alpha=0.5) # Trajectory
        ax.scatter(INITIAL_POSITIONS_A[i, 0], INITIAL_POSITIONS_A[i, 1], INITIAL_POSITIONS_A[i, 2],
                   marker='d', color='blue', alpha=0.6, label='Initial A' if i == 0 else "") # Initial point

    # Plot initial positions and trajectories (Group B)
    for i in range(NUM_UAV_B):
        ax.plot([INITIAL_POSITIONS_B[i, 0], pos_b[i, 0]],
                [INITIAL_POSITIONS_B[i, 1], pos_b[i, 1]],
                [INITIAL_POSITIONS_B[i, 2], pos_b[i, 2]], 'g:', alpha=0.5) # Trajectory
        ax.scatter(INITIAL_POSITIONS_B[i, 0], INITIAL_POSITIONS_B[i, 1], INITIAL_POSITIONS_B[i, 2],
                   marker='d', color='green', alpha=0.6, label='Initial B' if i == 0 else "") # Initial point

    # Plot final positions (Sender A)
    if len(pos_send_a) > 0:
        ax.scatter(pos_send_a[:, 0], pos_send_a[:, 1], pos_send_a[:, 2],
                   marker='o', color='cyan', s=50, label='Sender A')

    # Plot final positions (Sender B)
    if len(pos_send_b) > 0:
        ax.scatter(pos_send_b[:, 0], pos_send_b[:, 1], pos_send_b[:, 2],
                   marker='o', color='magenta', s=50, label='Sender B')

    # Plot final positions (Receiver A)
    ax.scatter(pos_rec_a[0], pos_rec_a[1], pos_rec_a[2],
               marker='*', color='red', s=150, label=f'Receiver A (idx {rec_b_idx})')

    # Plot final positions (Receiver B)
    ax.scatter(pos_rec_b[0], pos_rec_b[1], pos_rec_b[2],
               marker='*', color='yellow', s=150, label=f'Receiver B (idx {rec_a_idx})')

    # Set axis ranges and labels
    all_x = np.concatenate((INITIAL_POSITIONS_A[:, 0], INITIAL_POSITIONS_B[:, 0], pos_a[:, 0], pos_b[:, 0]))
    all_y = np.concatenate((INITIAL_POSITIONS_A[:, 1], INITIAL_POSITIONS_B[:, 1], pos_a[:, 1], pos_b[:, 1]))
    all_z = np.concatenate((INITIAL_POSITIONS_A[:, 2], INITIAL_POSITIONS_B[:, 2], pos_a[:, 2], pos_b[:, 2]))

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Maintain proportions to avoid distorting distance relationships
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
    
    # Ensure directory exists
    plots_dir = f"{result_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    file_path = f"{plots_dir}/deployment_gen_{gen}.png"
    plt.savefig(file_path)
    print(f"UAV deployment plot (Gen {gen}) saved to {file_path}")
    # plt.show()
    plt.close(fig) 