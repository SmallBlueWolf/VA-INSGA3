# Placeholder for INSGA-III algorithm implementation 

import random
import numpy as np
from deap import tools

from config import (
    LOWER_BOUNDS, UPPER_BOUNDS, IND_SIZE, REC_A_IDX, REC_B_IDX,
    NUM_UAV_A, NUM_UAV_B, TOTAL_UAVS, BH_FACTOR, MAX_GEN
)
from problem import extract_uav_info, INITIAL_POSITIONS_A, INITIAL_POSITIONS_B, set_initial_positions, WARNING_COUNT, MAX_WARNINGS # 需要初始 Z 坐标

# --- Helper Functions ---

def check_bounds(min_bound, max_bound):
    """创建 DEAP 边界检查装饰器所需的函数"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max_bound[i]:
                        child[i] = max_bound[i]
                    elif child[i] < min_bound[i]:
                        child[i] = min_bound[i]
            return offspring
        return wrapper
    return decorator

def get_bounds():
    """获取决策变量的边界"""
    return LOWER_BOUNDS, UPPER_BOUNDS

# --- OBL Initialization (Sec 4.2.1) ---
def generate_opposition(ind, lower_bounds, upper_bounds):
    """根据公式 4.1 生成对立个体"""
    opp_ind = ind.__class__(l + u - x for x, l, u in zip(ind, lower_bounds, upper_bounds))
    return opp_ind

# --- DBC Operator (Sec 4.2.2, Algo 4.3, 4.5) ---
def find_nearest_receiver(swarm_positions, target_swarm_positions):
    """为 swarm 选择距离 target_swarm 中心最近的接收端索引"""
    if len(swarm_positions) == 0 or len(target_swarm_positions) == 0:
        return 0 # 或者一个默认值
    target_center = np.mean(target_swarm_positions, axis=0)
    distances = np.linalg.norm(swarm_positions - target_center, axis=1)
    return np.argmin(distances)

def crossover_dbc(ind1, ind2, indpb_continuous):
    """基于距离的混合交叉算子 (DBC)
    indpb_continuous: 连续变量每个元素的交换概率
    """
    # 1. 对连续部分使用标准交叉 (这里用均匀交叉示例，可用 SBX)
    # 排除最后的两个离散接收端索引
    size = len(ind1) - 2
    tools.cxUniform(ind1[:size], ind2[:size], indpb_continuous)

    # 2. 对离散部分 (接收端) 使用 Near() 逻辑 (Algo 4.5 简化版)
    # 基于交叉后的位置信息更新接收端

    # 提取交叉后的位置信息
    pos_a1, _, pos_b1, _, _, _, _, _, _, _, _, _ = extract_uav_info(ind1)
    pos_a2, _, pos_b2, _, _, _, _, _, _, _, _, _ = extract_uav_info(ind2)

    # 更新 ind1 的接收端（B组内索引用于A->B通信，A组内索引用于B->A通信）
    rec_a1_new_idx = find_nearest_receiver(pos_b1, pos_a1) # 在B组中找A->B的接收端
    rec_b1_new_idx = find_nearest_receiver(pos_a1, pos_b1) # 在A组中找B->A的接收端
    ind1[REC_A_IDX] = rec_a1_new_idx
    ind1[REC_B_IDX] = rec_b1_new_idx

    # 更新 ind2 的接收端
    rec_a2_new_idx = find_nearest_receiver(pos_b2, pos_a2) # 在B组中找A->B的接收端
    rec_b2_new_idx = find_nearest_receiver(pos_a2, pos_b2) # 在A组中找B->A的接收端
    ind2[REC_A_IDX] = rec_a2_new_idx
    ind2[REC_B_IDX] = rec_b2_new_idx

    return ind1, ind2

# --- DBM Operator (Sec 4.2.2, Algo 4.4, 4.5) ---
def mutate_dbm(individual, low, up, mu, indpb_continuous):
    """基于距离的混合变异算子 (DBM)
    low, up: 连续变量的下界和上界列表
    mu: 拥挤度参数 (for polynomial mutation)
    indpb_continuous: 连续变量每个元素的变异概率
    """
    # 1. 对连续部分使用标准变异 (Polynomial Mutation)
    # 排除最后的两个离散接收端索引
    size = len(individual) - 2
    tools.mutPolynomialBounded(individual[:size], mu, low[:size], up[:size], indpb_continuous)

    # 2. 对离散部分 (接收端) 使用 Near() 逻辑 (Algo 4.5 简化版)
    # 基于变异后的位置信息更新接收端

    # 提取变异后的位置信息
    pos_a, _, pos_b, _, _, _, _, _, _, _, _, _ = extract_uav_info(individual)

    # 更新接收端（B组内索引用于A->B通信，A组内索引用于B->A通信）
    rec_a_new_idx = find_nearest_receiver(pos_b, pos_a) # 在B组中找A->B的接收端
    rec_b_new_idx = find_nearest_receiver(pos_a, pos_b) # 在A组中找B->A的接收端
    individual[REC_A_IDX] = rec_a_new_idx
    individual[REC_B_IDX] = rec_b_new_idx

    return individual,

# --- ALO Continuous Value Update (Sec 4.2.3) ---
def random_walk_alo(dim, max_iter, current_iter):
    """根据公式 4.2 和 4.3 生成随机游走向量"""
    # 定义边界收缩因子 I (可以根据需要调整)
    if current_iter == 0: current_iter = 1 # Avoid division by zero
    ratio = current_iter / max_iter
    if ratio <= 0.1: I = 1
    elif ratio <= 0.5: I = 10**(2 * ratio)
    elif ratio <= 0.75: I = 10**(ratio)
    elif ratio <= 0.9: I = 10**(0.5 * ratio)
    elif ratio <= 0.95: I = 10**(0.1 * ratio)
    else: I = 10**(0.01 * ratio)

    # 对应公式 (4.4) 中的上下界调整
    lb_walk = np.array(LOWER_BOUNDS) / I
    ub_walk = np.array(UPPER_BOUNDS) / I

    # 为每个维度独立生成随机游走
    normalized_walk = np.zeros(dim)
    
    for d in range(dim):
        # 进行随机游走
        X = [0] * (max_iter + 1)
        for t in range(1, max_iter + 1):
            rt = 1 if random.random() > 0.5 else 0
            X[t] = X[t-1] + (2 * rt - 1)

        # 归一化 (公式 4.4)
        a = min(X)
        b = max(X)
        if b - a < 1e-9: # 避免如果游走没有移动导致除零
            normalized_walk[d] = 0.0
        else:
            # 计算单一维度的归一化值
            X_array = np.array(X[1:max_iter+1])  # 取前max_iter个元素
            normalized_walk[d] = ((np.mean(X_array) - a) * (ub_walk[d] - lb_walk[d])) / (b - a) + lb_walk[d]

    return normalized_walk

def update_alo(ant, antlion, elite, max_iter, current_iter):
    """ALO 连续值更新策略 (Fig 4.6 简化，只做连续更新)
       ant: 当前个体 (蚂蚁)
       antlion: 轮盘赌选出的个体 (蚁狮)
       elite: 迄今为止最优的个体 (精英蚁狮)
       max_iter, current_iter: 最大/当前迭代次数
    """
    dim = len(ant) - 2 # 只更新连续部分

    # 围绕选定蚁狮的随机游走
    rw_antlion = random_walk_alo(dim, max_iter, current_iter)
    # 围绕精英蚁狮的随机游走
    rw_elite = random_walk_alo(dim, max_iter, current_iter)

    # 结合两个随机游走更新蚂蚁位置 (只更新连续部分)
    # 原ALO论文是 (rw_antlion + rw_elite) / 2
    # 这里简化为只受精英引导，更接近 BH 的思想，或随机选择一个
    # new_pos_continuous = (rw_antlion + rw_elite) / 2
    # 或者更简单：受精英影响
    # 修复: 将列表转换为NumPy数组再进行向量运算
    ant_array = np.array(ant[:dim])
    elite_array = np.array(elite[:dim])
    new_pos_continuous = ant_array + 0.5 * (elite_array - ant_array) * random.random() # 类似PSO/DE
    # 或者使用归一化后的随机游走直接更新（需要再确认论文意图）
    # new_pos_continuous = (np.array(antlion[:dim]) + np.array(elite[:dim])) / 2 # 均值引导

    # 简单实现：向精英移动一小步
    step_size = 0.1 * (1 - current_iter / max_iter) # 步长递减
    direction = elite_array - ant_array  # 修复: 使用NumPy数组进行向量运算
    norm_direction = direction / (np.linalg.norm(direction) + 1e-9)
    new_pos_continuous = ant_array + step_size * norm_direction * (np.array(UPPER_BOUNDS[:dim]) - np.array(LOWER_BOUNDS[:dim]))

    # 更新个体连续部分
    for i in range(dim):
        ant[i] = new_pos_continuous[i]

    # 离散部分不变
    # ant[REC_A_IDX] = ant[REC_A_IDX]
    # ant[REC_B_IDX] = ant[REC_B_IDX]

    # 确保边界
    lb, ub = get_bounds()
    for i in range(dim):
        ant[i] = np.clip(ant[i], lb[i], ub[i])

    return ant,

# --- BH Operator (Sec 4.2.4) ---
def apply_bh_operator(individual, current_gen, max_gen):
    """应用黑洞算子更新无人机位置 (公式 4.6, 4.7)"""
    global INITIAL_POSITIONS_A, INITIAL_POSITIONS_B, WARNING_COUNT
    
    # 如果初始位置未设置，尝试设置一次（避免多进程环境下的问题）
    if INITIAL_POSITIONS_A is None or INITIAL_POSITIONS_B is None:
        # 获取当前个体中的位置信息，作为初始位置的备选
        pos_a, _, pos_b, _, _, _, _, _, _, _, _, _ = extract_uav_info(individual)
        
        # 如果没有全局初始位置，就用当前位置作为初始位置
        if INITIAL_POSITIONS_A is None:
            INITIAL_POSITIONS_A = pos_a.copy()
            if WARNING_COUNT < MAX_WARNINGS:
                print("警告: BH算子 - 使用当前位置作为A组初始位置")
                WARNING_COUNT += 1
                if WARNING_COUNT == MAX_WARNINGS:
                    print("更多类似警告将被抑制...")
                    
        if INITIAL_POSITIONS_B is None:
            INITIAL_POSITIONS_B = pos_b.copy()
            if WARNING_COUNT < MAX_WARNINGS:
                print("警告: BH算子 - 使用当前位置作为B组初始位置")
                WARNING_COUNT += 1
                if WARNING_COUNT == MAX_WARNINGS:
                    print("更多类似警告将被抑制...")
    
    # 提取位置
    pos_a, _, pos_b, _, _, _, _, _, _, _, _, _ = extract_uav_info(individual)

    # 计算吸引强度 xi (随迭代减小)
    xi = random.random() * (1.0 - current_gen / max_gen) * BH_FACTOR

    # 水平 BH (公式 4.6)
    center_a_xy = np.mean(pos_a[:, :2], axis=0)
    center_b_xy = np.mean(pos_b[:, :2], axis=0)

    # 更新 A 组的 x, y 坐标
    for i in range(NUM_UAV_A):
        idx_x = 3 * i  # X 坐标
        idx_y = 3 * i + 1  # Y 坐标
        individual[idx_x] += xi * (center_a_xy[0] - individual[idx_x])  # X_A
        individual[idx_y] += xi * (center_a_xy[1] - individual[idx_y])  # Y_A
    
    # 更新 B 组的 x, y 坐标
    for i in range(NUM_UAV_B):
        idx_offset = 3 * NUM_UAV_A  # B 组的起始索引
        idx_x = idx_offset + 3 * i  # X 坐标
        idx_y = idx_offset + 3 * i + 1  # Y 坐标
        individual[idx_x] += xi * (center_b_xy[0] - individual[idx_x])  # X_B
        individual[idx_y] += xi * (center_b_xy[1] - individual[idx_y])  # Y_B

    # 垂直 BH (公式 4.7)
    # 使用当前位置代替初始位置进行计算，避免依赖全局变量
    if INITIAL_POSITIONS_A is not None and INITIAL_POSITIONS_B is not None:
        initial_z_a = INITIAL_POSITIONS_A[:, 2]
        initial_z_b = INITIAL_POSITIONS_B[:, 2]
    else:
        # 如果没有初始位置，则使用当前位置的z坐标
        initial_z_a = pos_a[:, 2]
        initial_z_b = pos_b[:, 2]

    # 更新 A 组的 z 坐标
    for i in range(NUM_UAV_A):
        idx_z = 3 * i + 2  # Z 坐标
        individual[idx_z] += xi * (initial_z_a[i] - individual[idx_z])  # Z_A
    
    # 更新 B 组的 z 坐标
    for i in range(NUM_UAV_B):
        idx_offset = 3 * NUM_UAV_A  # B 组的起始索引
        idx_z = idx_offset + 3 * i + 2  # Z 坐标
        individual[idx_z] += xi * (initial_z_b[i] - individual[idx_z])  # Z_B

    # 确保边界
    lb, ub = get_bounds()
    for i in range(len(individual)): # 检查所有维度，包括接收端
        individual[i] = np.clip(individual[i], lb[i], ub[i])

    return individual,

# --- INSGA-III Main Algorithm Structure (Example based on Algo 4.7) ---
# (实际执行将在 main.py 中使用 DEAP 工具箱完成)
# 此处仅为结构示意
def example_insga3_loop(pop_size, max_gen, toolbox):
    """示意性 INSGA-III 循环结构，非实际执行代码"""
    # 1. 生成参考点 (DEAP 功能)
    ref_points = tools.uniform_reference_points(toolbox.nobj, p=12) # 示例

    # 2. 初始化种群 (使用 OBL)
    population = toolbox.population_obl(n=pop_size)
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # NSGA-III 选择
    population = toolbox.select(population, pop_size)

    # --- 主循环 ---
    for gen in range(max_gen):
        # --- 生成子代 ---
        offspring = []
        # DBC (算法 4.3)
        # for _ in range(N_DBC): # 论文说使用 N 次
        # 选择两个父代
        # child1, child2 = toolbox.mate(parent1, parent2)
        # offspring.extend([child1, child2])
        # 使用概率更符合 DEAP 风格
        temp_offspring = tools.selTournament(population, len(population), tournsize=3)
        temp_offspring = [toolbox.clone(ind) for ind in temp_offspring]

        # 应用交叉 (DBC)
        for i in range(0, len(temp_offspring), 2):
            if random.random() < toolbox.cxpb and i+1 < len(temp_offspring):
                 toolbox.mate(temp_offspring[i], temp_offspring[i+1])

        # 应用变异 (DBM)
        for i in range(len(temp_offspring)):
            if random.random() < toolbox.mutpb:
                toolbox.mutate(temp_offspring[i])

        # 应用 ALO (算法 4.6)
        elite = tools.selBest(population, 1)[0] # 找到精英
        for i in range(len(temp_offspring)):
            if random.random() < toolbox.alopb:
                # 需要选择一个 antlion (例如随机选或轮盘赌)
                antlion = random.choice(population)
                toolbox.update_alo(temp_offspring[i], antlion, elite, max_gen, gen)

        offspring.extend(temp_offspring)

        # --- 应用 BH (算法 4.7, line 7) ---
        # 注意：BH 应用于整个种群还是子代？论文流程图显示更新 pop
        # 应用于合并后的种群更合理
        # for ind in population:
        #    toolbox.apply_bh(ind, gen, max_gen)

        # --- 评估新生成的无效个体 ---
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # --- 合并父代与子代 ---
        combined_pop = population + offspring

        # --- 应用 BH 到合并种群 ---
        for i in range(len(combined_pop)):
            # 假设 BH 应用有一定概率或固定次数
            # if random.random() < BH_PROB: # 或者固定次数 N_BH
            toolbox.apply_bh(combined_pop[i], gen, max_gen)
            # BH 可能改变个体，需要重新评估? 论文没说，假设不用
            # del combined_pop[i].fitness.values

        # --- NSGA-III 选择 ---
        population = toolbox.select(combined_pop, pop_size)

        # 记录统计信息等 (省略)

    return population 