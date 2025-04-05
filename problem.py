# Placeholder for A2ACMOP problem definition 

import numpy as np
from scipy import integrate
from scipy.spatial.distance import pdist, squareform
import random
from scipy.integrate import dblquad
from scipy.constants import c as speed_of_light

# 从配置文件导入参数
from config import (
    NUM_UAV_A, NUM_UAV_B, TOTAL_UAVS, WAVELENGTH, 
    BANDWIDTH_HZ, NOISE_POWER_W, ANTENNA_EFFICIENCY,
    MIN_DISTANCE, MAX_COMMUNICATION_TIME_S, REC_A_IDX, REC_B_IDX,
    X_BOUNDS_A, Y_BOUNDS_A, Z_BOUNDS_A, X_BOUNDS_B, Y_BOUNDS_B, Z_BOUNDS_B,
    TRANSMIT_POWER_W, SNR_GAMMA_0,
    P0_HOVER_POWER, Pi_INDUCED_POWER, ROTOR_TIP_SPEED, MEAN_ROTOR_INDUCED_VELOCITY,
    FUSELAGE_DRAG_RATIO, AIR_DENSITY, ROTOR_SOLIDITY, ROTOR_AREA,
    COMMUNICATION_POWER_W
)

# 全局变量存储初始位置（应在主程序中初始化一次）
INITIAL_POSITIONS_A = None
INITIAL_POSITIONS_B = None
# 跟踪警告消息计数，避免过多输出
WARNING_COUNT = 0
MAX_WARNINGS = 3

# 定义一些在配置中没有的常量
PATH_LOSS_EXPONENT = 2.0  # 自由空间路径损耗指数
CONSTANT_PATH_LOSS = 1.0  # 路径损耗常数
ENERGY_FACTOR_HORIZONTAL = 10.0  # 水平移动能耗因子
ENERGY_FACTOR_VERTICAL = 15.0    # 垂直移动能耗因子
ENERGY_FACTOR_HOVERING = 5.0     # 悬停能耗因子
COMM_TIME = MAX_COMMUNICATION_TIME_S  # 通信时间
TX_POWER_A = TRANSMIT_POWER_W    # A组发射功率
TX_POWER_B = TRANSMIT_POWER_W    # B组发射功率
BANDWIDTH = BANDWIDTH_HZ         # 通信带宽
NOISE_POWER_WATT = NOISE_POWER_W # 噪声功率

# 添加全局变量用于缓存积分计算结果
GAIN_CACHE = {}

def set_initial_positions():
    """生成并设置无人机的随机初始位置"""
    global INITIAL_POSITIONS_A, INITIAL_POSITIONS_B, WARNING_COUNT
    WARNING_COUNT = 0  # 重置警告计数
    INITIAL_POSITIONS_A = np.random.rand(NUM_UAV_A, 3)
    INITIAL_POSITIONS_A[:, 0] = INITIAL_POSITIONS_A[:, 0] * (X_BOUNDS_A[1] - X_BOUNDS_A[0]) + X_BOUNDS_A[0]
    INITIAL_POSITIONS_A[:, 1] = INITIAL_POSITIONS_A[:, 1] * (Y_BOUNDS_A[1] - Y_BOUNDS_A[0]) + Y_BOUNDS_A[0]
    INITIAL_POSITIONS_A[:, 2] = INITIAL_POSITIONS_A[:, 2] * (Z_BOUNDS_A[1] - Z_BOUNDS_A[0]) + Z_BOUNDS_A[0]

    INITIAL_POSITIONS_B = np.random.rand(NUM_UAV_B, 3)
    INITIAL_POSITIONS_B[:, 0] = INITIAL_POSITIONS_B[:, 0] * (X_BOUNDS_B[1] - X_BOUNDS_B[0]) + X_BOUNDS_B[0]
    INITIAL_POSITIONS_B[:, 1] = INITIAL_POSITIONS_B[:, 1] * (Y_BOUNDS_B[1] - Y_BOUNDS_B[0]) + Y_BOUNDS_B[0]
    INITIAL_POSITIONS_B[:, 2] = INITIAL_POSITIONS_B[:, 2] * (Z_BOUNDS_B[1] - Z_BOUNDS_B[0]) + Z_BOUNDS_B[0]
    
    return INITIAL_POSITIONS_A.copy(), INITIAL_POSITIONS_B.copy()

def set_positions_from_arrays(init_pos_a, init_pos_b):
    """从现有数组设置初始位置（用于多进程环境）"""
    global INITIAL_POSITIONS_A, INITIAL_POSITIONS_B
    if init_pos_a is not None:
        INITIAL_POSITIONS_A = init_pos_a.copy()
    if init_pos_b is not None:
        INITIAL_POSITIONS_B = init_pos_b.copy()

def extract_uav_info(individual):
    """从个体解中提取无人机信息"""
    # 确保接收端索引是整数且在有效范围内
    rec_a_idx_raw = individual[REC_A_IDX]
    rec_b_idx_raw = individual[REC_B_IDX]

    rec_a_idx = int(round(np.clip(rec_a_idx_raw, 0, NUM_UAV_B - 1)))
    rec_b_idx = int(round(np.clip(rec_b_idx_raw, 0, NUM_UAV_A - 1)))

    # 提取位置（注意没有电流信息的情况）
    pos_a = np.array(individual[0 : 3 * NUM_UAV_A]).reshape(NUM_UAV_A, 3)
    pos_b = np.array(individual[3 * NUM_UAV_A : 3 * NUM_UAV_A + 3 * NUM_UAV_B]).reshape(NUM_UAV_B, 3)

    # 设置默认电流为1.0（当没有电流参数时）
    current_a = np.ones(NUM_UAV_A)
    current_b = np.ones(NUM_UAV_B)

    # 分离发送端和接收端
    send_indices_a = np.arange(NUM_UAV_A) != rec_b_idx
    send_indices_b = np.arange(NUM_UAV_B) != rec_a_idx

    pos_send_a = pos_a[send_indices_a]
    current_send_a = current_a[send_indices_a]
    pos_rec_a = pos_a[rec_b_idx]

    pos_send_b = pos_b[send_indices_b]
    current_send_b = current_b[send_indices_b]
    pos_rec_b = pos_b[rec_a_idx]

    return pos_a, current_a, pos_b, current_b, pos_send_a, current_send_a, pos_rec_a, pos_send_b, current_send_b, pos_rec_b, rec_a_idx, rec_b_idx


def cartesian_to_spherical(xyz):
    """将笛卡尔坐标转换为球坐标 (r, theta, phi)"""
    x, y, z = xyz
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r) if r > 0 else 0 # 仰角 [0, pi]
    phi = np.arctan2(y, x) # 方位角 [-pi, pi]
    return r, theta, phi

def array_factor(theta, phi, positions, currents):
    """计算给定方向的天线阵因子 (公式 2.2)"""
    # 如果位置为空，则直接返回0
    if len(positions) == 0:
        return 0.0 + 0.0j
    
    k = 2 * np.pi / WAVELENGTH
    array_center = np.mean(positions, axis=0)
    
    # 使用向量化操作代替循环
    rel_pos = positions - array_center
    
    # 计算方向余弦
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    # 计算相位项
    dot_products = (rel_pos[:, 0] * sin_theta * cos_phi +
                   rel_pos[:, 1] * sin_theta * sin_phi +
                   rel_pos[:, 2] * cos_theta)
    
    phases = k * dot_products
    
    # 使用向量化操作计算复数和
    af = np.sum(currents * np.exp(1j * phases))
    
    return af

def radiation_pattern(theta, phi):
    """单个天线单元的辐射方向图，假设全向"""
    # w(theta, phi) = 1 for omnidirectional
    return 1.0

def gain_integrand(theta, phi, positions, currents):
    """增益公式 (2.3) 分母中的被积函数"""
    af_val = array_factor(theta, phi, positions, currents)
    w_val = radiation_pattern(theta, phi)
    return np.abs(af_val * w_val)**2 * np.sin(theta)

def split_integrate_region(func, phi_min, phi_max, theta_min, theta_max, args=(), n_phi=4, n_theta=4, **kwargs):
    """
    通过将积分区域分割成更小的子区域来计算双重积分。
    这对于处理复杂的被积函数特别有用，可以避免积分收敛问题。
    
    参数:
    func -- 被积函数
    phi_min, phi_max -- phi方向的积分范围
    theta_min, theta_max -- theta方向的积分范围
    args -- 传递给被积函数的额外参数
    n_phi, n_theta -- 在phi和theta方向上的分割数量
    kwargs -- 传递给dblquad的其他参数
    
    返回:
    积分结果
    """
    # scipy.integrate.dblquad的参数顺序:
    # dblquad(func, a, b, gfun, hfun)，其中:
    # - func: 被积函数，接收参数顺序为 (y, x, *args)
    # - a, b: x的积分范围
    # - gfun, hfun: 返回y的下限和上限的函数，它们是x的函数
    
    dphi = (phi_max - phi_min) / n_phi
    dtheta = (theta_max - theta_min) / n_theta
    
    integral_sum = 0.0
    
    for i in range(n_phi):
        phi_start = phi_min + i * dphi
        phi_end = phi_min + (i + 1) * dphi
        
        for j in range(n_theta):
            theta_start = theta_min + j * dtheta
            theta_end = theta_min + (j + 1) * dtheta
            
            # 注意dblquad的参数顺序: dblquad(func, x_min, x_max, y_min(x), y_max(x))
            # 我们的积分是: int_phi_start^phi_end int_theta_start^theta_end func(theta, phi) dtheta dphi
            # 因此我们需要将func(theta, phi)转换为func(phi, theta)
            def integrand_wrapper(phi, theta, *wrapper_args):
                return func(theta, phi, *wrapper_args)
            
            result, _ = integrate.dblquad(
                integrand_wrapper, phi_start, phi_end,
                lambda phi: theta_start, lambda phi: theta_end,
                args=args, **kwargs
            )
            
            integral_sum += result
    
    return integral_sum

def clear_gain_cache():
    """清除增益计算缓存"""
    global GAIN_CACHE
    GAIN_CACHE = {}

def gain_integrand_vectorized(thetas, phis, positions, currents):
    """
    天线增益积分的向量化版本
    可以同时计算多个(theta, phi)点的函数值
    
    参数:
    thetas -- theta角度值的数组
    phis -- phi角度值的数组
    positions -- 发送器位置数组
    currents -- 发送器电流数组
    
    返回:
    每个(theta, phi)点的函数值数组
    """
    k = 2 * np.pi / WAVELENGTH
    array_center = np.mean(positions, axis=0)
    rel_pos = positions - array_center
    
    result = np.zeros(len(thetas))
    
    for i in range(len(thetas)):
        theta = thetas[i]
        phi = phis[i]
        
        # 计算方向余弦
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        # 计算相位项
        dot_products = (rel_pos[:, 0] * sin_theta * cos_phi +
                       rel_pos[:, 1] * sin_theta * sin_phi +
                       rel_pos[:, 2] * cos_theta)
        
        phases = k * dot_products
        
        # 计算阵因子
        af = np.sum(currents * np.exp(1j * phases))
        
        # 计算辐射方向图 (全向模式 = 1.0)
        w = 1.0
        
        # 计算积分被积函数
        result[i] = np.abs(af * w)**2 * sin_theta
    
    return result

def monte_carlo_integration(func, args=(), n_samples=200):
    """
    使用蒙特卡洛积分方法计算球面上的积分
    利用随机采样来逼近积分值，对高维积分尤其有效
    
    参数:
    func -- 被积函数
    args -- 传递给被积函数的额外参数
    n_samples -- 随机采样点的数量
    
    返回:
    积分近似值
    """
    # 在球坐标系中生成均匀随机点
    # 均匀分布在[0, pi]的余弦值
    cos_theta = np.random.uniform(-1.0, 1.0, n_samples)
    theta = np.arccos(cos_theta)  # 这样产生的theta在球面上是均匀分布的
    phi = np.random.uniform(0, 2 * np.pi, n_samples)
    
    # 使用向量化函数一次性计算所有采样点的函数值
    function_values = gain_integrand_vectorized(theta, phi, *args)
    
    # 蒙特卡洛积分估计值 = (4π) * 函数值的平均值
    # 4π是球面的总面积
    integral_estimate = 4 * np.pi * np.mean(function_values)
    
    return integral_estimate

def discrete_sampling_approximation(func, args=(), n_phi=12, n_theta=6):
    """
    使用离散采样点近似计算球面积分，避免使用耗时的数值积分
    使用向量化操作加速计算
    
    参数:
    func -- 被积函数
    args -- 传递给被积函数的额外参数
    n_phi -- phi方向的采样点数
    n_theta -- theta方向的采样点数
    
    返回:
    积分近似值
    """
    # 均匀采样点
    dphi = 2 * np.pi / n_phi
    dtheta = np.pi / n_theta
    
    # 预先计算所有theta和phi值
    theta_values = np.linspace(dtheta/2, np.pi - dtheta/2, n_theta)
    phi_values = np.linspace(dphi/2, 2*np.pi - dphi/2, n_phi)
    
    # 预计算sin(theta)权重
    sin_theta_weights = np.sin(theta_values) * dtheta
    
    # 初始化积分结果
    integral_sum = 0.0
    
    # 为每组phi采样点创建theta数组
    all_thetas = []
    all_phis = []
    
    for phi in phi_values:
        for theta in theta_values:
            all_thetas.append(theta)
            all_phis.append(phi)
    
    all_thetas = np.array(all_thetas)
    all_phis = np.array(all_phis)
    
    # 一次性计算所有采样点的函数值
    all_values = gain_integrand_vectorized(all_thetas, all_phis, *args)
    
    # 重新组织和求和
    idx = 0
    for i, phi in enumerate(phi_values):
        for j, theta in enumerate(theta_values):
            integral_sum += all_values[idx] * sin_theta_weights[j] * dphi
            idx += 1
    
    return integral_sum

def calculate_gain_with_method(receiver_pos, sender_positions, sender_currents, method='discrete'):
    """
    使用不同的积分方法计算天线阵列增益
    
    参数:
    receiver_pos -- 接收器位置
    sender_positions -- 发送器位置数组
    sender_currents -- 发送器电流数组
    method -- 积分方法: 'discrete'使用离散采样, 'monte_carlo'使用蒙特卡洛方法
    
    返回:
    增益值
    """
    if len(sender_positions) == 0:
        return 0.0
        
    array_center = np.mean(sender_positions, axis=0)
    vec_to_receiver = receiver_pos - array_center

    # 检查距离是否过小，避免数值问题
    dist_to_receiver = np.linalg.norm(vec_to_receiver)
    if dist_to_receiver < 1e-6:
        return 0.0
        
    # 计算接收端相对于阵列中心的方向
    _, theta_rev, phi_rev = cartesian_to_spherical(vec_to_receiver)

    # 计算接收方向的阵因子
    af_rev = array_factor(theta_rev, phi_rev, sender_positions, sender_currents)
    w_rev = radiation_pattern(theta_rev, phi_rev)
    
    # 选择积分方法
    if method == 'monte_carlo':
        # 忽略func参数，直接使用向量化计算
        # 在球坐标系中生成均匀随机点
        n_samples = 200
        cos_theta = np.random.uniform(-1.0, 1.0, n_samples)
        theta = np.arccos(cos_theta)
        phi = np.random.uniform(0, 2 * np.pi, n_samples)
        
        # 使用向量化函数计算积分
        function_values = gain_integrand_vectorized(theta, phi, sender_positions, sender_currents)
        integral_result = 4 * np.pi * np.mean(function_values)
    else:  # 默认使用离散采样
        # 离散采样近似
        n_phi, n_theta = 12, 6
        
        # 均匀采样点
        dphi = 2 * np.pi / n_phi
        dtheta = np.pi / n_theta
        
        # 预先计算所有theta和phi值
        theta_values = np.linspace(dtheta/2, np.pi - dtheta/2, n_theta)
        phi_values = np.linspace(dphi/2, 2*np.pi - dphi/2, n_phi)
        
        # 预计算sin(theta)权重
        sin_theta_weights = np.sin(theta_values) * dtheta
        
        # 为每组phi采样点创建theta和phi数组
        all_thetas = []
        all_phis = []
        
        for phi in phi_values:
            for theta in theta_values:
                all_thetas.append(theta)
                all_phis.append(phi)
        
        all_thetas = np.array(all_thetas)
        all_phis = np.array(all_phis)
        
        # 一次性计算所有采样点的函数值
        all_values = gain_integrand_vectorized(all_thetas, all_phis, sender_positions, sender_currents)
        
        # 重新组织和求和
        integral_sum = 0.0
        idx = 0
        for i, phi in enumerate(phi_values):
            for j, theta in enumerate(theta_values):
                integral_sum += all_values[idx] * sin_theta_weights[j] * dphi
                idx += 1
        
        integral_result = integral_sum
    
    if integral_result < 1e-9:
        gain = 0.0
    else:
        numerator = 4 * np.pi * np.abs(af_rev * w_rev)**2
        gain = (numerator / integral_result) * ANTENNA_EFFICIENCY
    
    return gain

def calculate_gain(receiver_pos, sender_positions, sender_currents):
    """计算虚拟天线阵列在接收端方向上的增益 (公式 2.3)，使用缓存和采样近似加速"""
    global GAIN_CACHE
    
    if len(sender_positions) == 0:
        return 0.0

    # 生成缓存键
    # 注意：我们必须使用不可变对象作为字典键
    cache_key = (
        tuple(map(float, receiver_pos)),
        tuple(tuple(map(float, pos)) for pos in sender_positions),
        tuple(map(float, sender_currents))
    )
    
    # 检查缓存
    if cache_key in GAIN_CACHE:
        return GAIN_CACHE[cache_key]
    
    # 选择积分方法
    # 对于少量发送器(<5)，使用离散采样方法，速度更快且足够准确
    # 对于较多发送器，使用蒙特卡洛方法，更稳定
    method = 'discrete' if len(sender_positions) < 5 else 'monte_carlo'
    
    # 计算增益
    gain = calculate_gain_with_method(receiver_pos, sender_positions, sender_currents, method)
    
    # 保存结果到缓存
    GAIN_CACHE[cache_key] = gain
    
    return gain

def calculate_rate(gain, distance, tx_power):
    """计算传输速率 (公式 2.4)"""
    if distance < 1e-6 or gain < 1e-9: # 避免无效值
        return 0.0

    # 路径损耗 K * d^(-alpha)，这里假设 K=1，实际应根据模型调整
    path_loss_factor = CONSTANT_PATH_LOSS
    received_power = tx_power * path_loss_factor * gain * (distance ** (-PATH_LOSS_EXPONENT))

    # 信噪比 SNR
    snr = received_power / NOISE_POWER_WATT

    # 速率 R = B * log2(1 + SNR)
    rate = BANDWIDTH * np.log2(1 + snr)
    return rate

def calculate_simplified_energy(initial_pos, final_pos):
    """计算简化的推进能耗"""
    # WARNING: 这是基于config中能量因子的简化模型
    # 需要替换为公式 (2.5) 和 (2.6) 的精确实现和所需参数

    # 移动能耗
    movement_energy = 0
    for i in range(len(final_pos)):
        init_p = initial_pos[i]
        final_p = final_pos[i]
        delta_xy = np.linalg.norm(init_p[:2] - final_p[:2])
        delta_z = np.abs(init_p[2] - final_p[2])
        movement_energy += delta_xy * ENERGY_FACTOR_HORIZONTAL + delta_z * ENERGY_FACTOR_VERTICAL

    # 悬停能耗 (所有无人机在通信期间悬停)
    hovering_energy = len(final_pos) * ENERGY_FACTOR_HOVERING * COMM_TIME

    return movement_energy + hovering_energy

def check_constraints(pos_a, pos_b):
    """检查约束条件，主要是最小距离"""
    # 合并所有无人机的位置
    all_pos = np.vstack((pos_a, pos_b))

    # 计算所有无人机之间的距离
    distances = pdist(all_pos)

    # 检查是否有距离小于 MIN_DISTANCE
    if np.any(distances < MIN_DISTANCE):
        return False # 违反最小距离约束

    # 可以在这里添加其他约束检查，例如边界检查（虽然通常由算子处理）

    return True # 所有约束满足

def calculate_propulsion_power(speed):
    """根据文献2的公式(6)计算旋翼无人机的推进功率。"""
    V = speed
    V2 = V * V
    V4 = V2 * V2

    # 处理 V=0 的情况，避免除零
    if V < 1e-6:
        # 根据公式(61)，悬停功率 P_h = P0 + Pi
        return P0_HOVER_POWER + Pi_INDUCED_POWER

    # 剖面功率项
    power_profile = P0_HOVER_POWER * (1 + 3 * V2 / (ROTOR_TIP_SPEED**2))

    # 诱导功率项
    sqrt_term_inner = 1 + V4 / (4 * MEAN_ROTOR_INDUCED_VELOCITY**4)
    induced_term_inner = np.sqrt(sqrt_term_inner) - V2 / (2 * MEAN_ROTOR_INDUCED_VELOCITY**2)
    # 确保根号内非负 (理论上应非负)
    induced_term_inner = max(0, induced_term_inner)
    power_induced = Pi_INDUCED_POWER * np.sqrt(induced_term_inner)

    # 寄生功率项
    power_parasite = 0.5 * FUSELAGE_DRAG_RATIO * AIR_DENSITY * ROTOR_SOLIDITY * ROTOR_AREA * (V**3)

    total_power = power_profile + power_induced + power_parasite
    return total_power

def E_calc(pos_a, pos_b):
    """计算维持当前位置的总能量消耗 (推进+通信)。"""
    total_energy = 0.0

    # 假设无人机在通信阶段悬停 (速度 V=0)
    hover_propulsion_power = calculate_propulsion_power(0.0)

    # 计算所有无人机的总悬停推进功率 + 通信功率
    total_power_per_uav = hover_propulsion_power + COMMUNICATION_POWER_W
    total_power_all_uavs = TOTAL_UAVS * total_power_per_uav

    # 能量 = 总功率 * 通信时间
    total_energy = total_power_all_uavs * MAX_COMMUNICATION_TIME_S

    return total_energy

def evaluate_a2acmop(individual):
    """
    A2ACMOP 问题的评估函数 (目标函数计算).
    返回: (f1, f2, -f3)，因为我们要最大化f1, f2，最小化f3
    """
    global INITIAL_POSITIONS_A, INITIAL_POSITIONS_B, WARNING_COUNT
    # 如果初始位置未设置，则尝试使用个体信息
    if INITIAL_POSITIONS_A is None or INITIAL_POSITIONS_B is None:
        # 在多进程环境中，尝试提取当前个体的位置作为初始位置
        pos_a, _, pos_b, _, _, _, _, _, _, _, _, _ = extract_uav_info(individual)
        
        if INITIAL_POSITIONS_A is None:
            INITIAL_POSITIONS_A = pos_a.copy()
            if WARNING_COUNT < MAX_WARNINGS:
                print("警告: 使用当前个体位置作为A组初始位置")
                WARNING_COUNT += 1
                if WARNING_COUNT == MAX_WARNINGS:
                    print("更多类似警告将被抑制...")
        
        if INITIAL_POSITIONS_B is None:
            INITIAL_POSITIONS_B = pos_b.copy()
            if WARNING_COUNT < MAX_WARNINGS:
                print("警告: 使用当前个体位置作为B组初始位置")
                WARNING_COUNT += 1
                if WARNING_COUNT == MAX_WARNINGS:
                    print("更多类似警告将被抑制...")
            
        # 清空缓存
        clear_gain_cache()

    # 1. 从个体中提取信息
    (pos_a, current_a, pos_b, current_b,
     pos_send_a, current_send_a, pos_rec_a,
     pos_send_b, current_send_b, pos_rec_b,
     rec_a_idx, rec_b_idx) = extract_uav_info(individual)

    # 2. 检查约束
    if not check_constraints(pos_a, pos_b):
        # 如果违反约束，返回一个非常差的适应度值
        # 注意：DEAP期望目标是最小化，所以对于最大化目标返回0，最小化目标返回大正数
        return 0.0, 0.0, 1e18 # (Rate AB=0, Rate BA=0, Energy=很大)

    # 3. 计算目标 f1: Rate A -> B
    array_center_a = np.mean(pos_send_a, axis=0) if len(pos_send_a) > 0 else pos_rec_a # Handle case where only receiver exists
    distance_ab = np.linalg.norm(pos_rec_b - array_center_a)
    gain_ab = calculate_gain(pos_rec_b, pos_send_a, current_send_a)
    rate_ab = calculate_rate(gain_ab, distance_ab, TX_POWER_A)

    # 4. 计算目标 f2: Rate B -> A
    array_center_b = np.mean(pos_send_b, axis=0) if len(pos_send_b) > 0 else pos_rec_b # Handle case where only receiver exists
    distance_ba = np.linalg.norm(pos_rec_a - array_center_b)
    gain_ba = calculate_gain(pos_rec_a, pos_send_b, current_send_b)
    rate_ba = calculate_rate(gain_ba, distance_ba, TX_POWER_B)

    # 5. 计算目标 f3: Total Energy
    # WARNING: 使用简化能量模型
    energy_a = calculate_simplified_energy(INITIAL_POSITIONS_A, pos_a)
    energy_b = calculate_simplified_energy(INITIAL_POSITIONS_B, pos_b)
    total_energy = energy_a + energy_b

    # 6. 返回目标值 (最大化 f1, f2, 最小化 f3 => max f1, max f2, max -f3)
    # DEAP 默认最小化，所以适应度值为 (-f1, -f2, f3)
    return -rate_ab, -rate_ba, total_energy 