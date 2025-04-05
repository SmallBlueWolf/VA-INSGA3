# Placeholder for main execution logic

import random
import numpy as np
import time
from tqdm import tqdm
import multiprocessing
import os
import json
import pickle

from deap import base, creator, tools, algorithms

# 配置参数
from config import (
    POP_SIZE, MAX_GEN, NUM_OBJECTIVES, CXPB, MUTPB, ALOPB, USE_OBL,
    IND_SIZE, LOWER_BOUNDS, UPPER_BOUNDS, REC_A_IDX, REC_B_IDX,
    PLOT_PARETO_FRONT, PLOT_CONVERGENCE, PLOT_DEPLOYMENT, DEPLOYMENT_SNAPSHOT_GEN,
    NUM_DIVISIONS, USE_IMPROVED_ALGORITHM
)

# 问题定义和评估函数
from problem import evaluate_a2acmop, set_initial_positions, check_constraints, INITIAL_POSITIONS_A, INITIAL_POSITIONS_B

# 自定义算法算子
from algorithm import (
    generate_opposition, crossover_dbc, mutate_dbm, update_alo, apply_bh_operator,
    get_bounds, check_bounds
)

# 可视化函数
from visualization import plot_pareto_front, plot_convergence, plot_deployment

# 如果使用改进算法，导入相关功能
if USE_IMPROVED_ALGORITHM:
    from improved_algorithm import (
        get_adaptive_reference_points, generate_variational_offspring,
        evaluate_with_surrogate, update_surrogate_with_new_data
    )

# --- DEAP 设置 ---

# 创建适应度类 (最小化三个目标: -RateAB, -RateBA, Energy)
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, 1.0))

# 创建个体类 (列表，带有适应度属性)
creator.create("Individual", list, fitness=creator.FitnessMin)

# --- 工具箱注册 ---
toolbox = base.Toolbox()

# 属性生成器 (为个体生成随机浮点数)
def attr_item(low, up):
    return random.uniform(low, up)

toolbox.register("attr_float", attr_item)

# 个体生成器 (重复调用 attr_float)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_float for _ in range(IND_SIZE)), n=1)

# 调整个体生成，确保在边界内
# (直接在生成时应用边界似乎更简单)
def individual_bounded():
    """创建边界内的随机个体"""
    low, up = get_bounds()
    ind = creator.Individual(random.uniform(l, u) for l, u in zip(low, up))
    # 确保接收端索引初始时是近似整数，便于后续处理
    ind[REC_A_IDX] = random.randint(low[REC_A_IDX], up[REC_A_IDX])  # B组中的接收器索引
    ind[REC_B_IDX] = random.randint(low[REC_B_IDX], up[REC_B_IDX])  # A组中的接收器索引
    return ind
toolbox.register("individual_bounded", individual_bounded)

# 种群生成器
toolbox.register("population", tools.initRepeat, list, toolbox.individual_bounded)

# OBL 初始化
def population_obl(n):
    pop = toolbox.population(n=n // 2 if USE_OBL else n)
    if USE_OBL:
        opp_pop = []
        lb, ub = get_bounds()
        for ind in pop:
            opp_ind = generate_opposition(ind, lb, ub)
            opp_pop.append(opp_ind)
        # DEAP算法通常在初始评估后进行第一次选择，所以这里暂不筛选
        # 筛选逻辑可以在主循环开始前进行，或让NSGA3选择处理
        pop.extend(opp_pop)
    return pop
toolbox.register("population_obl", population_obl)

# 注册评估函数
toolbox.register("evaluate", evaluate_a2acmop)

# 如果使用改进算法，注册代理模型评估函数
if USE_IMPROVED_ALGORITHM:
    toolbox.register("evaluate_surrogate", evaluate_with_surrogate, real_eval_func=evaluate_a2acmop)
    # 为变分算子设置当前代数属性（初始为0）
    toolbox.current_gen = 0
    toolbox.max_gen = MAX_GEN

# 注册自定义交叉算子 (DBC)
# 注意：cxUniform 的 indpb 控制每个基因交换的概率
toolbox.register("mate", crossover_dbc, indpb_continuous=0.5) # 连续部分基因交换概率

# 注册自定义变异算子 (DBM)
# eta 是 mutPolynomialBounded 的拥挤度参数
# indpb 是每个基因发生变异的概率
lb, ub = get_bounds()
toolbox.register("mutate", mutate_dbm, low=lb, up=ub, mu=20.0, indpb_continuous=1.0/IND_SIZE)

# 注册自定义ALO更新算子
toolbox.register("update_alo", update_alo)

# 注册自定义BH算子
toolbox.register("apply_bh", apply_bh_operator)

# 生成NSGA-III所需的参考点
ref_points = tools.uniform_reference_points(nobj=NUM_OBJECTIVES, p=NUM_DIVISIONS)

# 注册选择算子 (NSGA-III)
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

# 保存ref_points为工具箱属性，以便后续更新
toolbox.ref_points = ref_points

# 注册边界检查装饰器
toolbox.decorate("mate", check_bounds(lb, ub))
toolbox.decorate("mutate", check_bounds(lb, ub))
toolbox.decorate("update_alo", check_bounds(lb, ub))
toolbox.decorate("apply_bh", check_bounds(lb, ub))

# --- 主程序 ---
def main():
    random.seed(42) # 设置随机种子以便复现
    np.random.seed(42)

    # 创建结果保存目录
    result_dir = "improv" if USE_IMPROVED_ALGORITHM else "origin"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(f"{result_dir}/plots", exist_ok=True)  # 为可视化结果创建子目录

    # 设置无人机初始位置（在创建进程池之前）
    print("生成无人机初始位置...")
    init_pos_a, init_pos_b = set_initial_positions()
    
    # 导入全局位置变量和设置函数
    from problem import set_positions_from_arrays
    
    # 定义进程池初始化函数，将初始位置传递给每个工作进程
    def init_worker():
        # 为每个子进程设置初始位置
        set_positions_from_arrays(init_pos_a, init_pos_b)
    
    # 设置多进程
    num_processes = 3  # 使用除了一个核之外的所有CPU核心
    num_processes = max(1, num_processes)  # 确保至少有一个核心
    print(f"使用 {num_processes} 个CPU核心进行并行计算...")
    
    # 替换默认的map函数为并行版本
    # 使用maxtasksperchild来限制每个进程处理的任务数，避免内存泄漏
    pool = multiprocessing.Pool(processes=num_processes, initializer=init_worker, maxtasksperchild=50)
    toolbox.register("map", pool.map)

    # 设置统计信息
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # 创建初始种群 (包含 OBL)
    print(f"初始化种群 (大小: {POP_SIZE})...")
    pop = toolbox.population_obl(n=POP_SIZE)

    # 评估初始种群中的无效个体
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    
    # 如果使用改进的代理模型
    if USE_IMPROVED_ALGORITHM and hasattr(toolbox, 'evaluate_surrogate'):
        fitnesses = [toolbox.evaluate_surrogate(ind) for ind in invalid_ind]
    else:
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        
    for ind, fit in tqdm(zip(invalid_ind, fitnesses), total=len(invalid_ind), desc="初始种群评估 [0/5]"):
        ind.fitness.values = fit

    # 如果使用改进算法，更新代理模型
    if USE_IMPROVED_ALGORITHM:
        update_surrogate_with_new_data(invalid_ind, [ind.fitness.values for ind in invalid_ind], 0)

    # 如果使用了OBL，初始种群大小可能是 POP_SIZE*2，需要第一次选择
    if USE_OBL:
        print("应用初始 NSGA-III 选择 (因使用 OBL)...")
        pop = toolbox.select(pop, POP_SIZE)

    # 记录初始统计信息
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # 创建文本日志文件
    with open(f"{result_dir}/optimization_log.txt", "w", encoding="utf-8") as log_file:
        log_file.write(f"优化算法: {'改进NSGA-III' if USE_IMPROVED_ALGORITHM else '原始NSGA-III'}\n")
        log_file.write(f"种群大小: {POP_SIZE}, 最大代数: {MAX_GEN}\n")
        log_file.write(f"交叉概率: {CXPB}, 变异概率: {MUTPB}, ALO概率: {ALOPB}\n")
        log_file.write(f"使用OBL初始化: {'是' if USE_OBL else '否'}\n\n")
        log_file.write("=== 优化日志 ===\n")
        log_file.write(f"代数 0: {logbook[0]}\n")

    # 绘制初始部署 (选择一个初始最佳个体)
    if PLOT_DEPLOYMENT and 0 in DEPLOYMENT_SNAPSHOT_GEN:
        # 确保初始位置已设置
        if INITIAL_POSITIONS_A is None or INITIAL_POSITIONS_B is None:
            print("初始位置尚未设置，无法绘制初始部署图")
        else:
            # 选择帕累托前沿中的一个点 (例如，能量最低的)
            best_ind_initial = min(pop, key=lambda x: x.fitness.values[2])
            plot_deployment(best_ind_initial, 0, result_dir=result_dir)

    # 开始进化
    print("开始进化...")
    start_time = time.time()
    for gen in tqdm(range(1, MAX_GEN + 1), desc=f"进化过程 [1/5]"):
        # 设置当前代数（供改进算法使用）
        if USE_IMPROVED_ALGORITHM:
            toolbox.current_gen = gen
            toolbox.max_gen = MAX_GEN

        # --- 生成子代 ---
        # 使用标准锦标赛选择代替DCD变体（DCD不适用于NSGA-III）
        selected_parents = tools.selTournament(pop, len(pop), tournsize=3)
        offspring = [toolbox.clone(ind) for ind in selected_parents]

        # 如果使用改进算法，使用变分分布采样生成额外子代
        if USE_IMPROVED_ALGORITHM:
            # 更新参考点（如果需要）
            if hasattr(toolbox, 'ref_points'):
                toolbox.ref_points = get_adaptive_reference_points(pop, gen, toolbox.ref_points)
                # 重新注册选择算子
                toolbox.register("select", tools.selNSGA3, ref_points=toolbox.ref_points)
            
            # 生成变分分布采样的子代
            var_offspring = generate_variational_offspring(pop, toolbox)
            if var_offspring:
                offspring.extend(var_offspring)

        # 应用交叉 (DBC)
        for i in tqdm(range(0, len(offspring), 2), desc=f"交叉操作 [2/5]", leave=False):
            if random.random() < CXPB and i+1 < len(offspring):
                 toolbox.mate(offspring[i], offspring[i+1])
                 del offspring[i].fitness.values # 交叉后适应度失效
                 del offspring[i+1].fitness.values

        # 应用变异 (DBM)
        for i in tqdm(range(len(offspring)), desc=f"变异操作 [3/5]", leave=False):
            if random.random() < MUTPB:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values # 变异后适应度失效

        # 应用 ALO
        if ALOPB > 0 and len(pop) > 0:
             elite = tools.selBest(pop, 1)[0] # 近似精英
             for i in tqdm(range(len(offspring)), desc=f"ALO操作 [4/5]", leave=False):
                 if random.random() < ALOPB:
                    antlion = random.choice(pop) # 随机选择蚁狮
                    toolbox.update_alo(offspring[i], antlion, elite, MAX_GEN, gen)
                    del offspring[i].fitness.values # ALO 更新后适应度失效

        # --- 合并父代与子代 ---
        combined_pop = pop + offspring

        # --- 应用 BH 算子到合并种群 --- (Algo 4.7, line 7)
        # BH 应用概率或次数可以配置，这里简化为对所有个体应用
        for i in tqdm(range(len(combined_pop)), desc=f"BH算子应用 [5/5]", leave=False):
            toolbox.apply_bh(combined_pop[i], gen, MAX_GEN)
            # BH 改变位置，理论上需要重新评估
            del combined_pop[i].fitness.values

        # --- 评估所有适应度无效的个体 ---
        # (经过交叉、变异、ALO、BH后)
        invalid_ind = [ind for ind in combined_pop if not ind.fitness.valid]
        
        # 如果使用改进的代理模型
        if USE_IMPROVED_ALGORITHM and hasattr(toolbox, 'evaluate_surrogate'):
            # 使用代理模型评估
            surrogate_fitnesses = []
            true_eval_inds = []
            
            for ind in invalid_ind:
                fitness = evaluate_with_surrogate(ind, toolbox.evaluate)
                surrogate_fitnesses.append(fitness)
                if not hasattr(ind, '_surrogate_evaluated') or not ind._surrogate_evaluated:
                    true_eval_inds.append(ind)
            
            # 更新代理模型
            if true_eval_inds:
                update_surrogate_with_new_data(true_eval_inds, 
                                              [ind.fitness.values for ind in true_eval_inds], 
                                              gen)
                
            # 设置适应度
            for ind, fit in zip(invalid_ind, surrogate_fitnesses):
                ind.fitness.values = fit
                ind._surrogate_evaluated = True
        else:
            # 使用真实评估函数
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in tqdm(zip(invalid_ind, fitnesses), total=len(invalid_ind), desc="适应度评估", leave=False):
                ind.fitness.values = fit

        # --- NSGA-III 选择 ---
        pop = toolbox.select(combined_pop, POP_SIZE)

        # --- 记录统计信息 ---
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
        
        # 记录到文本日志
        with open(f"{result_dir}/optimization_log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"代数 {gen}: {logbook[gen]}\n")

        # --- 可视化快照 ---
        if PLOT_DEPLOYMENT and gen in DEPLOYMENT_SNAPSHOT_GEN:
            # 选择当前帕累托前沿中能量最低的个体进行绘制
            current_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
            if current_front:
                 best_ind_snapshot = min(current_front, key=lambda x: x.fitness.values[2])
                 plot_deployment(best_ind_snapshot, gen, result_dir=result_dir)
            else:
                 print(f"第 {gen} 代没有找到非支配解用于绘制部署图。")

        if PLOT_PARETO_FRONT and gen % 50 == 0: # 每50代绘制一次帕累托前沿
            current_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
            if current_front:
                 plot_pareto_front(current_front, gen, result_dir=result_dir)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"进化完成，耗时: {execution_time:.2f} 秒")

    # --- 最终结果与可视化 ---
    print("\n最终种群帕累托前沿:")
    final_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    
    # 保存最终帕累托前沿结果到文本文件
    with open(f"{result_dir}/final_results.txt", "w", encoding="utf-8") as results_file:
        results_file.write(f"优化算法: {'改进NSGA-III' if USE_IMPROVED_ALGORITHM else '原始NSGA-III'}\n")
        results_file.write(f"总耗时: {execution_time:.2f} 秒\n\n")
        results_file.write("=== 最终帕累托前沿 ===\n")
        results_file.write("编号\tRate_AB (bps)\tRate_BA (bps)\tEnergy (J)\n")
        
        # 将前沿中的个体按能量排序
        sorted_front = sorted(final_front, key=lambda x: x.fitness.values[2])
        
        for i, ind in enumerate(sorted_front):
            rate_ab = -ind.fitness.values[0]  # 转换回原始值
            rate_ba = -ind.fitness.values[1]  # 转换回原始值
            energy = ind.fitness.values[2]
            results_file.write(f"{i}\t{rate_ab:.4e}\t{rate_ba:.4e}\t{energy:.4e}\n")
            print(f"  个体 {i}: {rate_ab:.2e} (R_AB), {rate_ba:.2e} (R_BA), {energy:.2e} (E_tot)")
    
    # 保存帕累托前沿数据用于后续分析
    pareto_data = {
        'algorithm': 'improved' if USE_IMPROVED_ALGORITHM else 'original',
        'execution_time': execution_time,
        'objectives': [(ind.fitness.values[0], ind.fitness.values[1], ind.fitness.values[2]) 
                       for ind in final_front],
        'hypervolume': None,  # 将在metric.py中计算
        'spread': None        # 将在metric.py中计算
    }
    
    with open(f"{result_dir}/pareto_data.pkl", "wb") as f:
        pickle.dump(pareto_data, f)

    # 保存收敛历史数据
    convergence_data = {
        'generations': logbook.select('gen'),
        'avg_values': logbook.select('avg'),
        'min_values': logbook.select('min'),
        'max_values': logbook.select('max'),
    }
    
    with open(f"{result_dir}/convergence_data.pkl", "wb") as f:
        pickle.dump(convergence_data, f)

    if PLOT_PARETO_FRONT:
        plot_pareto_front(final_front, gen=-1, result_dir=result_dir) # 绘制最终帕累托前沿

    if PLOT_CONVERGENCE:
        plot_convergence(logbook, result_dir=result_dir)

    if PLOT_DEPLOYMENT and MAX_GEN not in DEPLOYMENT_SNAPSHOT_GEN:
        # 绘制最终一代的最佳个体部署图
        best_ind_final = min(final_front, key=lambda x: x.fitness.values[2]) # 按能量最低选
        plot_deployment(best_ind_final, MAX_GEN, result_dir=result_dir)

    return pop, logbook

if __name__ == "__main__":
    try:
        final_pop, stats_log = main()
        # 可以在这里添加更多对 final_pop 的分析
        print("\n优化过程结束。")
    except KeyboardInterrupt:
        print("\n程序被用户中断。")
    finally:
        # 关闭所有进程池
        if 'pool' in locals():
            pool.close()
            pool.join()
            print("已关闭所有进程池。")
