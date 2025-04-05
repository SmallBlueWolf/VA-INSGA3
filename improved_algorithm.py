"""
改进的NSGA-III算法实现：
1. 基于变分分布的动态参考点自适应机制
2. 进化阶段自适应的变分分布采样算子
3. 结合变分不确定性估计的代理模型混合策略
"""

import numpy as np
import random
from deap import tools, base, creator
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

from config import (
    # 改进点1参数
    ENABLE_ADAPTIVE_REF_POINTS, ADAPTIVE_REF_POINTS_FREQ, 
    ADAPTIVE_REF_POINTS_METHOD, GMM_COMPONENTS, SMOOTH_FACTOR,
    # 改进点2参数
    ENABLE_VARIATIONAL_SAMPLING, VAR_SAMPLING_PROB, 
    STAGE_THRESHOLDS, SAMPLING_METHODS, VAR_SAMPLING_RATIO,
    # 改进点3参数
    ENABLE_SURROGATE_MODEL, ENABLE_UNCERTAINTY, UNCERTAINTY_THRESHOLD,
    RETRAINING_FREQ, RETRAINING_SAMPLES, MC_DROPOUT_SAMPLES, ONLINE_LEARNING_BATCH,
    # 其他必要配置
    NUM_OBJECTIVES, IND_SIZE, LOWER_BOUNDS, UPPER_BOUNDS, REC_A_IDX, REC_B_IDX
)

# 保存历史数据用于代理模型训练
X_history = []
y_history = []
surrogate_model = None

# ================================================
# 改进点1: 基于变分分布的动态参考点自适应机制
# ================================================

def get_current_stage(gen, max_gen):
    """根据当前代数确定算法所处阶段"""
    ratio = gen / max_gen
    for i, threshold in enumerate(STAGE_THRESHOLDS):
        if ratio <= threshold:
            return i
    return len(STAGE_THRESHOLDS)

def get_objectives_values(population):
    """获取种群中个体的目标值"""
    objectives = np.array([ind.fitness.values for ind in population if ind.fitness.valid])
    # 转换目标值，使其都为正方向(对于速率目标需要取相反数)
    objectives[:, 0] = -objectives[:, 0]  # Rate_AB
    objectives[:, 1] = -objectives[:, 1]  # Rate_BA
    # objectives[:, 2] 是能量，不需要转换
    return objectives

def normalize_objectives(objectives):
    """归一化目标值到[0,1]范围"""
    min_vals = np.min(objectives, axis=0)
    max_vals = np.max(objectives, axis=0)
    
    # 避免除零错误
    ranges = max_vals - min_vals
    ranges = np.where(ranges < 1e-10, 1e-10, ranges)
    
    normalized = (objectives - min_vals) / ranges
    return normalized, min_vals, max_vals, ranges

def denormalize_points(points, min_vals, ranges):
    """将归一化的点转换回原始尺度"""
    return points * ranges + min_vals

def generate_gmm_reference_points(objectives, n_points, n_components=GMM_COMPONENTS):
    """使用高斯混合模型生成参考点"""
    normalized, min_vals, max_vals, ranges = normalize_objectives(objectives)
    
    # 拟合GMM
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(normalized)
    
    # 从GMM采样
    samples, _ = gmm.sample(n_samples=n_points)
    
    # 确保样本在[0,1]范围内
    samples = np.clip(samples, 0, 1)
    
    # 将样本转换回原始尺度
    return denormalize_points(samples, min_vals, ranges)

def generate_kde_reference_points(objectives, n_points, bandwidth=0.1):
    """使用核密度估计生成参考点"""
    normalized, min_vals, max_vals, ranges = normalize_objectives(objectives)
    
    # 拟合KDE
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(normalized)
    
    # 生成多组候选点
    n_candidates = n_points * 5  # 生成更多候选点以便筛选
    candidates = np.random.random((n_candidates, normalized.shape[1]))
    scores = kde.score_samples(candidates)
    
    # 选择得分最高的n_points个点
    indices = np.argsort(scores)[-n_points:]
    samples = candidates[indices]
    
    # 将样本转换回原始尺度
    return denormalize_points(samples, min_vals, ranges)

def smooth_reference_points(old_points, new_points, factor=SMOOTH_FACTOR):
    """平滑新旧参考点，避免参考点剧烈变化"""
    n_points = len(old_points)
    n_new = len(new_points)
    
    # 如果新旧参考点数量不同，则不进行平滑
    if n_points != n_new:
        return new_points
    
    # 计算新旧参考点的距离矩阵
    distances = np.zeros((n_points, n_new))
    for i in range(n_points):
        for j in range(n_new):
            distances[i, j] = np.linalg.norm(old_points[i] - new_points[j])
    
    # 使用匈牙利算法进行匹配
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(distances)
    
    # 根据匹配结果平滑参考点
    smooth_points = np.zeros_like(old_points)
    for i, j in zip(row_ind, col_ind):
        smooth_points[i] = factor * old_points[i] + (1 - factor) * new_points[j]
    
    return smooth_points

def update_reference_points(population, current_ref_points, n_points):
    """根据当前种群更新参考点"""
    objectives = get_objectives_values(population)
    
    if len(objectives) < 5:  # 样本太少，无法可靠估计分布
        return current_ref_points
    
    if ADAPTIVE_REF_POINTS_METHOD == 'gmm':
        new_points = generate_gmm_reference_points(objectives, n_points)
    elif ADAPTIVE_REF_POINTS_METHOD == 'kde':
        new_points = generate_kde_reference_points(objectives, n_points)
    else:  # 默认使用传统方法
        return current_ref_points
    
    # 如果存在当前参考点，则进行平滑
    if current_ref_points is not None and len(current_ref_points) > 0:
        new_points = smooth_reference_points(current_ref_points, new_points)
    
    return new_points

# ================================================
# 改进点2: 进化阶段自适应的变分分布采样算子
# ================================================

def sample_from_distribution(population, n_samples, method='gmm', tool=None):
    """从当前种群分布中采样新个体"""
    if len(population) < 5:  # 样本太少
        return []
    
    # 提取个体的决策变量
    decision_vars = np.array([ind[:] for ind in population])
    
    # 连续部分（排除接收端索引）
    continuous_vars = decision_vars[:, :-2]  # 假设最后两个是离散接收端索引
    
    if method == 'random':
        # 简单随机采样（初期探索）
        samples = np.random.uniform(
            low=LOWER_BOUNDS[:-2],  # 除去离散索引
            high=UPPER_BOUNDS[:-2],
            size=(n_samples, len(LOWER_BOUNDS)-2)
        )
    elif method == 'gmm':
        # 使用GMM采样
        gmm = GaussianMixture(n_components=min(GMM_COMPONENTS, len(continuous_vars)//2),
                             covariance_type='full', random_state=42)
        gmm.fit(continuous_vars)
        samples, _ = gmm.sample(n_samples=n_samples)
        
        # 确保边界约束
        for i in range(samples.shape[1]):
            samples[:, i] = np.clip(samples[:, i], LOWER_BOUNDS[i], UPPER_BOUNDS[i])
    else:
        # 默认使用精英周围小扰动
        elite_indices = np.argsort([ind.fitness.values[2] for ind in population])[:max(5, len(population)//10)]
        elite_vars = continuous_vars[elite_indices]
        
        # 从精英中随机选择并添加扰动
        selected = np.random.choice(len(elite_vars), n_samples)
        samples = elite_vars[selected].copy()
        
        # 添加扰动
        noise = np.random.normal(0, 0.1, samples.shape)
        samples += noise
        
        # 确保边界约束
        for i in range(samples.shape[1]):
            samples[:, i] = np.clip(samples[:, i], LOWER_BOUNDS[i], UPPER_BOUNDS[i])
    
    # 为采样的个体创建完整个体（需要添加离散接收端索引）
    new_individuals = []
    for s in samples:
        if tool:
            ind = tool.individual_bounded()  # 使用工具箱创建个体
            # 替换连续部分
            for i in range(len(s)):
                ind[i] = s[i]
            # 离散部分保持随机生成
            new_individuals.append(ind)
    
    return new_individuals

def get_receiver_indices(pos_a, pos_b):
    """基于位置确定接收端索引"""
    from algorithm import find_nearest_receiver
    rec_a_idx = find_nearest_receiver(pos_b, pos_a)  # B组中的接收端索引
    rec_b_idx = find_nearest_receiver(pos_a, pos_b)  # A组中的接收端索引
    return rec_a_idx, rec_b_idx

def fix_discrete_variables(individual):
    """修正离散变量，确保接收端索引合法"""
    from problem import extract_uav_info, check_constraints
    
    # 提取位置信息
    pos_a, _, pos_b, _, _, _, _, _, _, _, _, _ = extract_uav_info(individual)
    
    # 根据位置确定最合适的接收端索引
    rec_a_idx, rec_b_idx = get_receiver_indices(pos_a, pos_b)
    
    # 更新接收端索引
    individual[REC_A_IDX] = rec_a_idx
    individual[REC_B_IDX] = rec_b_idx
    
    return individual

def create_variational_offspring(population, toolbox, n_samples):
    """创建基于变分分布的子代"""
    if not ENABLE_VARIATIONAL_SAMPLING or random.random() > VAR_SAMPLING_PROB:
        return []
    
    # 确定当前阶段
    current_gen = getattr(toolbox, 'current_gen', 0)
    max_gen = getattr(toolbox, 'max_gen', 100)
    stage = get_current_stage(current_gen, max_gen)
    
    # 获取当前阶段的采样方法
    method = SAMPLING_METHODS[min(stage, len(SAMPLING_METHODS)-1)]
    
    # 从分布中采样新个体
    sampled_individuals = sample_from_distribution(population, n_samples, method, toolbox)
    
    # 修正离散变量
    for ind in sampled_individuals:
        fix_discrete_variables(ind)
    
    return sampled_individuals

# ================================================
# 改进点3: 结合变分不确定性估计的代理模型
# ================================================

class MCDropout(nn.Module):
    """实现MC Dropout层"""
    def __init__(self, p=0.5):
        super(MCDropout, self).__init__()
        self.p = p
    
    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)

class RateResWithUncertainty(nn.Module):
    """带有不确定性估计的RateRes代理模型"""
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.1):
        super(RateResWithUncertainty, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            MCDropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            MCDropout(dropout_rate)
        )
        
        # 预测器 - 均值
        self.mean_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            MCDropout(dropout_rate),
            nn.Linear(hidden_dim // 2, NUM_OBJECTIVES)  # 输出所有目标的预测值
        )
        
        # 预测器 - 对数方差
        self.logvar_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            MCDropout(dropout_rate),
            nn.Linear(hidden_dim // 2, NUM_OBJECTIVES)  # 输出所有目标的方差
        )
    
    def forward(self, x):
        """前向传播，返回均值和对数方差"""
        features = self.encoder(x)
        mean = self.mean_predictor(features)
        logvar = self.logvar_predictor(features)  # 预测对数方差更稳定
        return mean, logvar
    
    def predict_with_uncertainty(self, x, n_samples=MC_DROPOUT_SAMPLES):
        """使用MC Dropout进行不确定性估计"""
        self.train()  # 启用dropout
        
        means = []
        logvars = []
        
        # 收集多次采样结果
        for _ in range(n_samples):
            mean, logvar = self.forward(x)
            means.append(mean.detach())
            logvars.append(logvar.detach())
        
        # 计算预测均值和方差
        mean_pred = torch.stack(means).mean(dim=0)
        
        # 两种不确定性来源：
        # 1. 模型本身预测的方差
        # 2. 多次采样的方差
        model_var = torch.exp(torch.stack(logvars).mean(dim=0))
        epistemic_var = torch.var(torch.stack(means), dim=0)  # 认知不确定性
        
        # 总不确定性
        total_var = model_var + epistemic_var
        
        return mean_pred, total_var

def encode_individual(individual):
    """将个体编码为模型输入"""
    # 对于复杂的个体结构，这里可能需要进一步处理
    # 这里简单地将个体的连续部分转换为张量
    x = torch.tensor(individual, dtype=torch.float32)
    return x

def train_surrogate_model(X, y, epochs=100, batch_size=32):
    """训练代理模型"""
    global surrogate_model
    
    input_dim = X.shape[1]
    
    # 创建或重用模型
    if surrogate_model is None:
        surrogate_model = RateResWithUncertainty(input_dim)
    
    # 创建数据集和数据加载器
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 优化器
    optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=0.001)
    
    # 训练循环
    surrogate_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            # 前向传播
            mean, logvar = surrogate_model(batch_x)
            
            # 负对数似然损失 (NLL)
            precision = torch.exp(-logvar)
            loss = torch.mean(0.5 * precision * (mean - batch_y) ** 2 + 0.5 * logvar)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 打印训练进度
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")
    
    return surrogate_model

def predict_with_surrogate(individual, threshold=UNCERTAINTY_THRESHOLD):
    """使用代理模型进行预测，返回预测值和是否可信"""
    global surrogate_model
    
    if surrogate_model is None:
        return None, False
    
    # 编码个体
    x = encode_individual(individual).unsqueeze(0)  # 添加批次维度
    
    # 预测
    with torch.no_grad():
        mean, var = surrogate_model.predict_with_uncertainty(x)
    
    # 计算不确定性指标 (如变异系数 CV)
    std = torch.sqrt(var)
    cv = std / (torch.abs(mean) + 1e-8)  # 避免除零
    
    # 判断预测是否可信
    is_reliable = (cv.max().item() <= threshold)
    
    # 将预测结果转换为元组形式 (与真实目标函数一致)
    pred_values = mean.squeeze().tolist()
    
    return tuple(pred_values), is_reliable

def update_surrogate_model(individuals, fitnesses, incremental=True):
    """更新代理模型"""
    global X_history, y_history, surrogate_model
    
    # 编码个体
    X_new = torch.stack([encode_individual(ind) for ind in individuals])
    y_new = torch.tensor(fitnesses, dtype=torch.float32)
    
    # 更新历史数据
    if incremental and len(X_history) > 0:
        X_history = torch.cat([X_history, X_new], dim=0)
        y_history = torch.cat([y_history, y_new], dim=0)
    else:
        X_history = X_new
        y_history = y_new
    
    # 如果样本量足够大，进行增量训练
    if incremental and len(X_new) < ONLINE_LEARNING_BATCH:
        return surrogate_model
    
    # 限制历史数据大小
    if len(X_history) > RETRAINING_SAMPLES:
        indices = torch.randperm(len(X_history))[:RETRAINING_SAMPLES]
        X_train = X_history[indices]
        y_train = y_history[indices]
    else:
        X_train = X_history
        y_train = y_history
    
    # 训练模型
    train_surrogate_model(X_train, y_train, epochs=50 if incremental else 100)
    
    return surrogate_model

# ================================================
# 主要接口函数，供main.py调用
# ================================================

def get_adaptive_reference_points(population, gen, ref_points=None):
    """获取自适应参考点，供main.py调用"""
    if not ENABLE_ADAPTIVE_REF_POINTS or gen % ADAPTIVE_REF_POINTS_FREQ != 0:
        return ref_points
    
    n_points = len(ref_points) if ref_points is not None else 91  # 默认值
    return update_reference_points(population, ref_points, n_points)

def generate_variational_offspring(population, toolbox, n_samples=None):
    """生成变分分布采样的子代，供main.py调用"""
    if n_samples is None:
        n_samples = int(len(population) * VAR_SAMPLING_RATIO)
    
    return create_variational_offspring(population, toolbox, n_samples)

def evaluate_with_surrogate(individual, real_eval_func):
    """结合代理模型和真实评估函数进行评估"""
    if not ENABLE_SURROGATE_MODEL:
        return real_eval_func(individual)
    
    # 使用代理模型预测
    pred_values, is_reliable = predict_with_surrogate(individual)
    
    # 如果预测不可靠或代理模型未初始化，使用真实评估函数
    if pred_values is None or not is_reliable:
        return real_eval_func(individual)
    
    return pred_values

def update_surrogate_with_new_data(new_individuals, new_fitnesses, gen):
    """更新代理模型的数据集"""
    if not ENABLE_SURROGATE_MODEL:
        return
    
    # 周期性完全重训练
    incremental = (gen % RETRAINING_FREQ != 0)
    update_surrogate_model(new_individuals, new_fitnesses, incremental) 