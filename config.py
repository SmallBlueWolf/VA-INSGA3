"""
Configuration file for the A2ACMOP simulation.
Stores parameters related to the problem definition, algorithm settings,
and visualization preferences. Sourced from relevant literature where possible.
"""
import numpy as np

# ==========================================
# A2ACMOP Problem Parameters
# ==========================================

# --- UAV Configuration ---
NUM_UAV_A = 5  # Number of UAVs in group A (transmitters)
NUM_UAV_B = 5  # Number of UAVs in group B (transmitters)
TOTAL_UAVS = NUM_UAV_A + NUM_UAV_B

# --- Activity Area ---
# Boundaries for group A UAVs (meters)
X_BOUNDS_A = [0, 100]
Y_BOUNDS_A = [0, 100]
Z_BOUNDS_A = [50, 150]
# Boundaries for group B UAVs (meters)
X_BOUNDS_B = [100, 200] # Example: Group B starts further away
Y_BOUNDS_B = [0, 100]
Z_BOUNDS_B = [50, 150]

# --- Constraints ---
MIN_DISTANCE = 1.0  # Minimum allowed distance between any two UAVs (meters)

# --- Communication Parameters ---
CARRIER_FREQUENCY_HZ = 2.4e9  # Carrier frequency (Hz), e.g., 2.4 GHz
WAVELENGTH = 3e8 / CARRIER_FREQUENCY_HZ # Wavelength (meters), lambda = c / f
# TRANSMIT_POWER_W = 1.0 # Total transmit power for each group's virtual array (Watts) - Placeholder
TRANSMIT_POWER_W = 0.1 # Transmit power per drone (W) - From Lit 1, Table I. Assuming array power is sum? Needs clarification. Let's assume this is per group for now.
BANDWIDTH_HZ = 10e6 # Communication bandwidth (Hz) - Ref Lit 1 (10MHz mentioned)
NOISE_POWER_SPECTRAL_DENSITY_DBM_HZ = -160.0 # Noise power spectral density (dBm/Hz) - From Lit 1, Table I
NOISE_POWER_W = (10**(NOISE_POWER_SPECTRAL_DENSITY_DBM_HZ / 10) / 1000) * BANDWIDTH_HZ # Noise power (Watts), N = N0 * B
SNR_GAMMA_0 = 1e6 # Reference SNR at 1m distance (linear scale, 60dB) - From Lit 2, Sec V Simulation Setup
ANTENNA_GAIN_DBI = 0.0 # Antenna gain (dBi) - Assuming isotropic for now
ANTENNA_EFFICIENCY = 1.0 # Antenna efficiency (0 to 1) - Assuming ideal

# --- Energy Model Parameters (Rotary Wing UAV - Based on Lit 2, Appendix A & Table I) ---
# Note: These parameters are based on a specific UAV model (W=100N) from Lit 2.
# Using these requires implementing the corresponding propulsion power model in problem.py
AIRCRAFT_WEIGHT_N = 100.0 # Aircraft weight (Newtons) - Lit 2, Table I
GRAVITY_ACCEL = 9.81 # Gravitational acceleration (m/s^2)
MASS_KG = AIRCRAFT_WEIGHT_N / GRAVITY_ACCEL # Mass of each drone (kg) - Derived from Lit 2 Weight
AIR_DENSITY = 1.225 # Air density (kg/m^3) - Lit 2, Table I
ROTOR_RADIUS = 0.5 # Rotor radius (m) - Lit 2, Table I
ROTOR_AREA = np.pi * ROTOR_RADIUS**2 # Rotor disc area (m^2) - Lit 2, Table I
ROTOR_OMEGA = 400.0 # Blade angular velocity (rad/s) - Lit 2, Table I
ROTOR_TIP_SPEED = ROTOR_OMEGA * ROTOR_RADIUS # Tip speed of the rotor blade (m/s) - Lit 2, Table I
NUM_ROTOR_BLADES = 4 # Number of blades - Lit 2, Table I
ROTOR_SOLIDITY = 0.05 # Rotor solidity (dimensionless) - Lit 2, Table I
FUSELAGE_DRAG_RATIO = 0.3 # Fuselage drag ratio (dimensionless) - Lit 2, Table I (Note: Lit 2 text calculation differs from table value, using table value)
INDUCED_POWER_FACTOR = 0.1 # Incremental correction factor to induced power (k) - Lit 2, Table I
MEAN_ROTOR_INDUCED_VELOCITY = 7.2 # Mean rotor induced velocity in hover (v0) (m/s) - Lit 2, Table I
PROFILE_DRAG_COEFF = 0.012 # Profile drag coefficient (delta) - Lit 2, Table I
# Calculated Hover Power Components (Based on Lit 2, Eq. 61 and Table I parameters)
P0_HOVER_POWER = 573.3 # Blade profile power in hover (Watts) - Calculated based on Lit 2 params
Pi_INDUCED_POWER = 790.8 # Induced power in hover (Watts) - Calculated based on Lit 2 params
HOVER_POWER = P0_HOVER_POWER + Pi_INDUCED_POWER # Total power in hover (Watts)

COMMUNICATION_POWER_W = 50.0 # Communication circuitry power consumption (Watts) - From Lit 2, Sec V Simulation Setup
MAX_COMMUNICATION_TIME_S = 10.0 # Placeholder: Maximum time duration for communication phase

# --- UAV Dynamics (Placeholders/Defaults) ---
MAX_SPEED = 60.0  # Maximum speed of UAVs (m/s) - From Lit 2, Sec V Simulation Setup
MAX_ACCELERATION = 5.0 # Maximum acceleration of UAVs (m/s^2) - Placeholder, adjust as needed


# ==========================================
# INSGA-III Algorithm Parameters
# ==========================================

POP_SIZE = 100      # Population size
MAX_GEN = 200       # Maximum number of generations
NUM_OBJECTIVES = 3  # Number of objectives (R_AB, R_BA, E_total)

# --- NSGA-III Specific ---
# Number of divisions for reference point generation along each objective axis
# For 3 objectives, p divisions lead to H = C(p + M - 1, M - 1) reference points
# E.g., p=12 -> H = C(12+3-1, 3-1) = C(14, 2) = 91 reference points
NUM_DIVISIONS = 12 # NSGA-III parameter 'p'

# --- Operator Probabilities ---
CXPB = 0.9          # Crossover probability (for DBC)
MUTPB = 0.1         # Mutation probability (for DBM)
ALOPB = 0.3         # Probability of applying ALO update - Needs tuning based on paper's findings
BH_PROB = 0.2       # Probability of applying BH operator - Placeholder, needs tuning

# --- Custom Operator Parameters ---
# ALO Parameters (if needed, otherwise handled within algorithm.py)
# BH Parameters
BH_FACTOR = 0.5    # Black Hole attraction strength factor (example) - Needs tuning

# --- Opposition-Based Learning ---
USE_OBL = True      # Whether to use OBL for initialization


# ==========================================
# Visualization Parameters
# ==========================================

PLOT_PARETO_FRONT = True    # Plot the Pareto front (final and intermediate)
PLOT_CONVERGENCE = True     # Plot the convergence curves (average objectives vs generations)
PLOT_DEPLOYMENT = True      # Plot UAV deployment snapshots
DEPLOYMENT_SNAPSHOT_GEN = [0, MAX_GEN // 2, MAX_GEN] # Generations to plot deployment


# ==========================================
# Derived Parameters (DO NOT MODIFY directly)
# ==========================================

# Calculate the size of the individual's chromosome
# Pos A (3*N_A) + Pos B (3*N_B) + Current A (N_A) + Current B (N_B) + Rec A (1) + Rec B (1)
# Note: Incentive currents are not used in the current simplified problem.py
# IND_SIZE = 3 * NUM_UAV_A + 3 * NUM_UAV_B + NUM_UAV_A + NUM_UAV_B + 1 + 1
IND_SIZE = 3 * NUM_UAV_A + 3 * NUM_UAV_B + 1 + 1 # Size without currents

# Define bounds for decision variables (location and receiver index)
LOWER_BOUNDS = []
UPPER_BOUNDS = []

# Bounds for UAV A positions (x, y, z for each UAV)
for _ in range(NUM_UAV_A):
    LOWER_BOUNDS.extend([X_BOUNDS_A[0], Y_BOUNDS_A[0], Z_BOUNDS_A[0]])
    UPPER_BOUNDS.extend([X_BOUNDS_A[1], Y_BOUNDS_A[1], Z_BOUNDS_A[1]])

# Bounds for UAV B positions (x, y, z for each UAV)
for _ in range(NUM_UAV_B):
    LOWER_BOUNDS.extend([X_BOUNDS_B[0], Y_BOUNDS_B[0], Z_BOUNDS_B[0]])
    UPPER_BOUNDS.extend([X_BOUNDS_B[1], Y_BOUNDS_B[1], Z_BOUNDS_B[1]])

# Bounds for Receiver Index for A -> B transmission (index within Group B)
LOWER_BOUNDS.append(0)
UPPER_BOUNDS.append(NUM_UAV_B - 1)
REC_A_IDX = len(LOWER_BOUNDS) - 1 # Index in the individual list

# Bounds for Receiver Index for B -> A transmission (index within Group A)
LOWER_BOUNDS.append(0)
UPPER_BOUNDS.append(NUM_UAV_A - 1)
REC_B_IDX = len(LOWER_BOUNDS) - 1 # Index in the individual list

# Sanity check
assert len(LOWER_BOUNDS) == IND_SIZE, "Bounds list length mismatch"
assert len(UPPER_BOUNDS) == IND_SIZE, "Bounds list length mismatch"

print("Configuration loaded.")

# ==========================================
# 算法改进配置参数
# ==========================================

# 主开关参数：选择使用哪种算法版本
USE_IMPROVED_ALGORITHM = True  # 设置为True使用改进版本，False使用原始INSGA-III

# --- 改进点1: 基于变分分布的动态参考点自适应机制 ---
ENABLE_ADAPTIVE_REF_POINTS = True  # 是否启用自适应参考点
ADAPTIVE_REF_POINTS_FREQ = 10      # 每隔多少代更新一次参考点
ADAPTIVE_REF_POINTS_METHOD = 'gmm' # 参考点生成方法：'gmm'(高斯混合模型)或'kde'(核密度估计)
GMM_COMPONENTS = 5                 # GMM模型的组件数量
SMOOTH_FACTOR = 0.7                # 平滑因子：新旧参考点混合比例(0-1)，越大表示保留越多旧参考点

# --- 改进点2: 进化阶段自适应的变分分布采样算子 ---
ENABLE_VARIATIONAL_SAMPLING = True  # 是否启用变分分布采样
VAR_SAMPLING_PROB = 0.3             # 使用变分分布采样的概率
STAGE_THRESHOLDS = [0.3, 0.7]       # 进化阶段划分阈值，如[0.3, 0.7]将进化分为三个阶段
SAMPLING_METHODS = ['random', 'gmm', 'gmm'] # 各阶段使用的采样方法
VAR_SAMPLING_RATIO = 0.2            # 每代使用变分采样生成的个体比例

# --- 改进点3: 变分不确定性估计的代理模型 ---
ENABLE_SURROGATE_MODEL = False       # 是否启用代理模型
ENABLE_UNCERTAINTY = True            # 是否启用不确定性估计
UNCERTAINTY_THRESHOLD = 0.2          # 不确定性阈值，超过此值将使用真实模型评估
RETRAINING_FREQ = 50                 # 代理模型重新训练频率(每隔多少代)
RETRAINING_SAMPLES = 1000            # 用于重训练的样本数量
MC_DROPOUT_SAMPLES = 10              # Monte Carlo Dropout的采样次数
ONLINE_LEARNING_BATCH = 20           # 在线学习的批量大小

print("算法改进配置已加载。") 