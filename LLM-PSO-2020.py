from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import levy
from scipy.stats import levy_stable
from opfunu.cec_based.cec2020 import *
from openai import OpenAI
import os
import re

# 配置DeepSeek客户端
client = OpenAI(
    api_key='Your API',
    base_url="deepseek.com")

PopSize = 100
DimSize = 50
LB = [-100] * DimSize
UB = [100] * DimSize
MaxFEs = DimSize * 1000

MaxIter = 500
curIter = 0

Pop = np.zeros((PopSize, DimSize))
Off = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)
FitOff = np.zeros(PopSize)
Trace = []

# PSO 特有参数
pbest = None  # 个体最优位置
gbest = None  # 全局最优位置
pbest_fit = None  # 个体最优适应度
gbest_fit = None  # 全局最优适应度
v = None  # 粒子速度
v_max = None  # 最大速度

memory_size = 5
F_memory = np.random.uniform(0.4, 0.9, memory_size)
Cr_memory = np.random.uniform(0.7, 0.95, memory_size)
memory_pos = 0
archive = []  # 外部存档SHADE策略参数

tmpFitPop = deepcopy(FitPop)
Trials = 3


def unified_initialization(func):
    """统一的种群初始化函数，确保所有算法使用相同的初始种群"""
    global Pop, FitPop, PopSize, DimSize, LB, UB
    Pop = np.zeros((PopSize, DimSize))
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + np.random.rand() * (UB[j] - LB[j])
        FitPop[i] = func.evaluate(Pop[i])
    return deepcopy(Pop), deepcopy(FitPop)


def random_initialization():
    """随机初始化种群 - 保持向后兼容"""
    global Pop, FitPop, PopSize, DimSize, LB, UB
    Pop = np.zeros((PopSize, DimSize))
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + np.random.rand() * (UB[j] - LB[j])


def Enhanced_CMA_ES_Operator(i):
    """增强的CMA-ES算子 - 通用版本"""
    global Pop, gbest, curIter, MaxIter, DimSize
    # 自适应学习率
    sigma = 0.2 * (1 - curIter/MaxIter)**1.5 + 0.05

    # 动态协方差策略
    if curIter < MaxIter * 0.4:
        # 早期：使用对角矩阵快速探索
        cov_matrix = np.eye(DimSize) * sigma
    else:
        # 中后期：使用种群协方差信息
        pop_cov = np.cov(Pop.T)
        # 正则化防止奇异
        eigenvals = np.linalg.eigvals(pop_cov)
        min_eigenval = np.min(np.abs(eigenvals))
        if min_eigenval < 1e-10:
            cov_matrix = np.eye(DimSize) * sigma
        else:
            cov_matrix = 0.6 * pop_cov + 0.4 * np.eye(DimSize) * sigma

    sample = np.random.multivariate_normal(np.zeros(DimSize), cov_matrix)

    # 动态混合策略
    exploration_weight = 0.3 * (1 - curIter/MaxIter) + 0.1
    exploitation_weight = 0.6 + 0.2 * np.sin(curIter/15)  # 震荡以保持多样性

    Off[i] = (gbest * exploitation_weight +
              Pop[i] * exploration_weight +
              0.1 * sample)
    Off[i] = np.clip(Off[i], LB, UB)

def Adaptive_Levy_Flight(i):
    """自适应Levy飞行算子"""
    global Pop, gbest, pbest, curIter, MaxIter, LB, UB
    # 自适应步长控制
    base_scale = 0.1 * (UB[0] - LB[0])
    scale = base_scale / ((curIter * 0.1 + 1)**0.5)

    # 动态beta参数（范围限制在[-1, 1]，符合levy_stable要求）
    beta = 0.3 + 0.4 * np.sin(curIter/20)  # 调整范围避免超出[-1,1]
    beta = np.clip(beta, -0.9, 0.9)  # 留一点余量

    # 多策略Levy飞行
    strategy = np.random.choice(['global', 'personal', 'diverse'])

    if strategy == 'global':
        # 围绕全局最优，使用levy_stable分布
        levy_step = levy_stable.rvs(alpha=1.0, beta=beta, size=DimSize) * scale
        Off[i] = gbest + levy_step
    elif strategy == 'personal':
        # 围绕个体历史最优
        levy_step = levy_stable.rvs(alpha=1.0, beta=beta, size=DimSize) * scale * 0.7
        Off[i] = pbest[i] + levy_step
    else:
        # 围绕随机个体保持多样性
        random_idx = np.random.randint(PopSize)
        levy_step = levy_stable.rvs(alpha=1.0, beta=beta, size=DimSize) * scale * 0.5
        Off[i] = Pop[random_idx] + levy_step

    Off[i] = np.clip(Off[i], LB, UB)

def JADE_style_DE(i):
    """JADE风格的自适应差分进化"""
    global Pop, pbest, pbest_fit, PopSize, gbest
    # 自适应参数生成
    mu_F = 0.5  # F的均值
    mu_Cr = 0.9  # Cr的均值

    F = np.random.normal(mu_F, 0.1)
    F = np.clip(F, 0.05, 0.95)
    Cr = np.random.normal(mu_Cr, 0.05)
    Cr = np.clip(Cr, 0.7, 0.99)

    # 当前到pbest/1策略
    pbest_size = max(2, PopSize // 4)
    pbest_indices = np.argsort(pbest_fit)[:pbest_size]
    pbest_idx = np.random.choice(pbest_indices)

    candidates = [x for x in range(PopSize) if x != i]
    r1, r2 = np.random.choice(candidates, 2, replace=False)

    # 变异操作
    mutant = Pop[i] + F * (pbest[pbest_idx] - Pop[i]) + F * (Pop[r1] - Pop[r2])

    # 交叉操作
    trial = np.copy(Pop[i])
    j_rand = np.random.randint(DimSize)
    cross_points = np.random.rand(DimSize) < Cr
    cross_points[j_rand] = True  # 确保至少一个维度交叉
    trial[cross_points] = mutant[cross_points]

    Off[i] = np.clip(trial, LB, UB)

def Comprehensive_Learning(i):
    """增强的综合学习策略"""
    global Pop, pbest, pbest_fit, PopSize
    # 动态学习概率
    Pc = 0.05 + 0.1 * (1 - curIter/MaxIter)  # 随迭代递减

    learning_exemplar = np.zeros(DimSize)

    for d in range(DimSize):
        if np.random.rand() < Pc:
            # 向其他粒子学习
            candidates = [j for j in range(PopSize) if j != i]
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            # 选择适应度更好的作为老师
            if pbest_fit[r1] < pbest_fit[r2]:
                teacher = r1
            else:
                teacher = r2
            learning_exemplar[d] = pbest[teacher, d]
        else:
            # 向自身历史学习
            learning_exemplar[d] = pbest[i, d]

    # 结合当前速度信息（如果有）
    if 'v' in globals():
        w = 0.9 - (0.9 - 0.4) * (curIter / MaxIter)
        Off[i] = Pop[i] + w * v[i] + 0.5 * (learning_exemplar - Pop[i])
    else:
        Off[i] = Pop[i] + 0.5 * (learning_exemplar - Pop[i])

    Off[i] = np.clip(Off[i], LB, UB)

def Hybrid_PSO_GA(i):
    """PSO-GA混合算子"""
    global Pop, pbest, gbest, v, curIter, MaxIter, PopSize

    # 动态操作选择
    operation = np.random.choice(['pso', 'crossover', 'mutation'],
                                p=[0.5, 0.35, 0.15])

    if operation == 'pso':
        # PSO核心更新
        w = 0.9 - (0.9 - 0.4) * (curIter / MaxIter)
        c1, c2 = 1.8 + 0.4 * np.sin(curIter/25), 1.8 + 0.4 * np.cos(curIter/25)
        r1, r2 = np.random.rand(DimSize), np.random.rand(DimSize)

        v[i] = w * v[i] + c1 * r1 * (pbest[i] - Pop[i]) + c2 * r2 * (gbest - Pop[i])
        v[i] = np.clip(v[i], -v_max, v_max)
        Off[i] = Pop[i] + v[i]

    elif operation == 'crossover':
        # 模拟二进制交叉
        candidates = [x for x in range(PopSize) if x != i]
        parent2_idx = np.random.choice(candidates)
        parent2 = Pop[parent2_idx]

        beta = np.random.rand(DimSize) * 2 - 1  # [-1, 1]
        beta = np.where(np.random.rand(DimSize) < 0.5,
                       (2 * np.random.rand(DimSize))**(1/3),
                       (1/(2 * (1 - np.random.rand(DimSize))))**(1/3))

        Off[i] = 0.5 * ((1 + beta) * Pop[i] + (1 - beta) * parent2)

    else:  # mutation
        # 自适应高斯变异
        mutation_strength = 0.1 * (UB[0] - LB[0]) * (1 - curIter/MaxIter)**2
        mutation = np.random.normal(0, mutation_strength, DimSize)
        Off[i] = Pop[i] + mutation

    Off[i] = np.clip(Off[i], LB, UB)

def Opposition_Learning_Enhanced(i):
    """增强的广义反向学习"""
    global Pop, LB, UB, gbest
    # 动态边界计算
    lb_array = np.array(LB)
    ub_array = np.array(UB)

    # 使用种群统计信息
    pop_min = np.min(Pop, axis=0)
    pop_max = np.max(Pop, axis=0)
    pop_mean = np.mean(Pop, axis=0)

    # 多策略反向学习
    strategy = np.random.choice(['classical', 'generalized', 'quasi'])

    if strategy == 'classical':
        # 经典反向学习
        opposite = lb_array + ub_array - Pop[i]
    elif strategy == 'generalized':
        # 广义反向学习
        k = np.random.rand()
        dynamic_lb = np.minimum(lb_array, gbest - k * (ub_array - lb_array))
        dynamic_ub = np.maximum(ub_array, gbest + k * (ub_array - lb_array))
        opposite = dynamic_lb + dynamic_ub - Pop[i]
    else:  # quasi-opposition
        # 准反向学习
        opposite = (lb_array + ub_array) / 2 + (lb_array + ub_array) / 2 - Pop[i]

    # 边界处理增强
    margin = 0.01 * (ub_array - lb_array)
    opposite = np.clip(opposite, lb_array + margin, ub_array - margin)
    Off[i] = opposite

def Local_Global_Mix(i):
    """局部-全局混合搜索算子"""
    global Pop, gbest, pbest, PopSize
    # 动态权重
    progress = curIter / MaxIter
    global_weight = 0.3 + 0.5 * progress  # 随迭代增加全局搜索权重
    local_weight = 0.5 - 0.3 * progress   # 随迭代减少局部搜索权重
    random_weight = 0.2                   # 保持随机探索

    # 全局搜索成分
    global_component = gbest - Pop[i]

    # 局部搜索成分（邻域搜索）
    neighbors = [j for j in range(PopSize) if j != i]
    neighbor_idx = np.random.choice(neighbors)
    local_component = Pop[neighbor_idx] - Pop[i]

    # 随机探索成分
    random_component = np.random.normal(0, 0.1 * (UB[0] - LB[0]), DimSize)

    # 混合搜索方向
    search_direction = (global_weight * global_component +
                       local_weight * local_component +
                       random_weight * random_component)

    # 自适应步长
    step_size = 0.5 * (1 - progress)**1.5 + 0.1

    Off[i] = Pop[i] + step_size * search_direction
    Off[i] = np.clip(Off[i], LB, UB)

def Self_Adaptive_Mutation(i):
    """自适应变异算子"""
    global Pop, curIter, MaxIter
    # 多变异策略
    strategies = ['gaussian', 'cauchy', 'polynomial']
    weights = [0.5, 0.3, 0.2]  # 策略权重

    chosen_strategy = np.random.choice(strategies, p=weights)
    base_scale = 0.1 * (UB[0] - LB[0])
    adaptive_scale = base_scale * (1 - curIter/MaxIter)**2

    if chosen_strategy == 'gaussian':
        # 高斯变异 - 局部精细搜索
        mutation = np.random.normal(0, adaptive_scale * 0.3, DimSize)
    elif chosen_strategy == 'cauchy':
        # 柯西变异 - 强探索能力
        mutation = np.random.standard_cauchy(DimSize) * adaptive_scale
    else:  # polynomial
        # 多项式变异
        mutation = np.random.rand(DimSize) * 2 - 1
        mutation = np.sign(mutation) * np.abs(mutation)**1.5 * adaptive_scale

    # 精英引导
    if np.random.rand() < 0.7:
        base_solution = gbest
    else:
        base_solution = Pop[i]

    Off[i] = base_solution + mutation
    Off[i] = np.clip(Off[i], LB, UB)

def DE_current_to_pbest(i):
    global Pop, pbest, PopSize
    candi = list(range(PopSize))
    candi.remove(i)

    # 选择pbest个体（前20%）
    pbest_size = max(1, PopSize // 5)
    pbest_indices = np.argsort(pbest_fit)[:pbest_size]
    pbest_idx = np.random.choice(pbest_indices)

    r1, r2 = np.random.choice(candi, 2, replace=False)
    F = 0.5
    Off[i] = Pop[i] + F * (pbest[pbest_idx] - Pop[i]) + F * (Pop[r1] - Pop[r2])
    Off[i] = np.clip(Off[i], LB, UB)
# 通用高性能算子集合

def CLPSO_learning(i):
    global Pop, pbest, v, gbest
    # 向其他粒子的历史最优学习
    teacher = np.random.choice(PopSize)
    phi = 0.1 + 0.4 * np.random.rand(DimSize)
    Off[i] = Pop[i] + v[i] + phi * (pbest[teacher] - Pop[i])
    Off[i] = np.clip(Off[i], LB, UB)

Operators = [
    Enhanced_CMA_ES_Operator,    # 0: 增强CMA-ES - 局部精细搜索
    Adaptive_Levy_Flight,        # 1: 自适应Levy飞行 - 强全局探索
    JADE_style_DE,               # 2: JADE差分进化 - 自适应参数
    Comprehensive_Learning,      # 3: 综合学习 - 多样性保持
    Hybrid_PSO_GA,               # 4: PSO-GA混合 - 平衡搜索
    Opposition_Learning_Enhanced, # 5: 增强反向学习 - 收敛加速
    Local_Global_Mix,            # 6: 局部-全局混合 - 自适应平衡
    Self_Adaptive_Mutation,      # 7: 自适应变异 - 多策略探索
    DE_current_to_pbest,         # 8: 当前到pbest - 稳健性能
    CLPSO_learning               # 9: 综合学习PSO - 防早熟
]
Sequence = np.zeros(PopSize)


def generate_adaptive_sequence():
    global PopSize, curIter, MaxIter
    progress = curIter / MaxIter
    max_op_index = 9  # 明确算子最大索引
    Sequence = np.zeros(PopSize, dtype=int)

    # 权重保持原逻辑
    if progress < 0.3:
        weights = [0.15, 0.05, 0.15, 0.1, 0.15, 0.05, 0.1, 0.05, 0.1, 0.1]
    elif progress < 0.7:
        weights = [0.1, 0.1, 0.1, 0.15, 0.1, 0.1, 0.15, 0.1, 0.05, 0.05]
    else:
        weights = [0.05, 0.15, 0.05, 0.1, 0.05, 0.15, 0.15, 0.15, 0.1, 0.05]

    # 归一化权重
    weights = np.array(weights) / np.sum(weights)

    # 严格从0~max_op_index选择
    for i in range(PopSize):
        Sequence[i] = np.random.choice(range(max_op_index + 1), p=weights)
    return Sequence.tolist()


def enhance_diversity(sequence, recommended_ops, max_op_index):
    """增强序列多样性（确保推荐算子被充分使用，且索引合法）"""
    sequence_array = np.array(sequence)

    # 核心修复1：过滤推荐算子，只保留0~max_op_index的有效值
    valid_recommended = [op for op in recommended_ops if 0 <= op <= max_op_index]

    # 核心修复2：如果推荐算子为空，用默认安全算子填充
    if not valid_recommended:
        valid_recommended = [1, 3, 5, 7]  # 通用安全算子

    unique_ops, counts = np.unique(sequence_array, return_counts=True)

    for op in valid_recommended[:6]:  # 只使用前6个有效推荐算子
        # 检查当前算子是否使用不足
        if op not in unique_ops or counts[list(unique_ops).index(op)] < len(sequence) // 15:
            # 随机替换部分位置为该算子（确保替换数量合理）
            replace_num = min(len(sequence) // 10, len(sequence) - len(unique_ops))
            if replace_num > 0:
                replace_indices = np.random.choice(len(sequence), size=replace_num, replace=False)
                sequence_array[replace_indices] = op  # 此时op已确保有效

    # 最终转换前再次校验
    return [max(0, min(int(x), max_op_index)) for x in sequence_array.tolist()]


def HighLevel(current_operators):
    global Pop, FitPop, PopSize, Sequence, tmpFitPop, curIter, MaxIter
    max_op_index = len(current_operators) - 1

    # 全面状态分析
    current_best = min(FitPop)
    fitness_std = np.std(FitPop)
    diversity_indicator = fitness_std / (np.max(FitPop) - current_best + 1e-12)

    # 收敛分析
    convergence_rate = 0
    if len(Trace) > 30:
        convergence_rate = (Trace[-30] - current_best) / 30

    # 种群质量分析
    top_10_percent = np.partition(FitPop, PopSize // 10)[:PopSize // 10]
    bottom_10_percent = np.partition(FitPop, -PopSize // 10)[-PopSize // 10:]
    population_quality = np.mean(top_10_percent) / (np.mean(bottom_10_percent) + 1e-12)

    # 智能阶段判断
    progress = curIter / MaxIter
    if progress < 0.2:
        phase = "AGGRESSIVE_EXPLORATION"
    elif progress < 0.5:
        phase = "BALANCED_SEARCH"
    elif progress < 0.8:
        phase = "FOCUSED_REFINEMENT"
    else:
        phase = "FINAL_POLISHING"

    # 基于多指标的策略选择
    strategy_analysis = "UNIVERSAL OPTIMIZATION STRATEGY:\n"

    if diversity_indicator < 0.02:
        strategy_analysis += "CRITICAL: Very low diversity - BOOST exploration operators 1,7,3\n"
        primary_ops = [1, 7, 3, 1, 7, 3]  # 重复关键算子
    elif diversity_indicator < 0.05:
        strategy_analysis += "WARNING: Low diversity - Increase exploration 1,7\n"
        primary_ops = [1, 7, 2, 3]
    elif convergence_rate < 1e-13:
        strategy_analysis += "SLOW convergence - Enhance 5,0,8 for acceleration\n"
        primary_ops = [5, 0, 8, 2]
    elif population_quality > 0.8:
        strategy_analysis += "Good population quality - Balance all operators\n"
        primary_ops = [2, 6, 4, 0, 5, 1, 7, 3, 8, 9]
    else:
        strategy_analysis += "Normal optimization - Phase-based strategy\n"
        primary_ops = []

    # 阶段专用策略
    phase_strategy = f"PHASE: {phase}\n"
    if phase == "AGGRESSIVE_EXPLORATION":
        phase_strategy += "Goal: Maximize global search capability\n"
        phase_strategy += "Priority: 1(Levy-25%), 7(Mutation-20%), 3(Learning-15%), 2(JADE-15%)\n"
        base_ops = [1, 1, 1, 7, 7, 3, 2, 5, 6, 4]
    elif phase == "BALANCED_SEARCH":
        phase_strategy += "Goal: Balance exploration and exploitation\n"
        phase_strategy += "Priority: 2(JADE-18%), 6(Mix-16%), 1(Levy-14%), 5(Opposition-14%)\n"
        base_ops = [2, 6, 1, 5, 4, 7, 0, 3, 8, 9]
    elif phase == "FOCUSED_REFINEMENT":
        phase_strategy += "Goal: Focus on local refinement with diversity\n"
        phase_strategy += "Priority: 0(CMA_ES-20%), 8(DE_pbest-18%), 5(Opposition-15%), 2(JADE-12%)\n"
        base_ops = [0, 8, 5, 2, 4, 6, 1, 7, 3, 9]
    else:  # FINAL_POLISHING
        phase_strategy += "Goal: Final convergence with escape mechanisms\n"
        phase_strategy += "Priority: 0(CMA_ES-22%), 8(DE_pbest-20%), 4(PSO-GA-15%), 5(Opposition-13%)\n"
        base_ops = [0, 8, 4, 5, 2, 6, 7, 1, 3, 9]

    # 合并策略
    if primary_ops:  # 如果有紧急策略
        recommended_operators = primary_ops + base_ops
    else:
        recommended_operators = base_ops

    # 构建全面优化的提示
    prompt = f"""COMPREHENSIVE OPTIMIZATION STRATEGY FOR SUPERIOR PERFORMANCE

CURRENT STATUS:
- Iteration: {curIter}/{MaxIter} ({progress:.1%})
- Best Fitness: {current_best:.6e}
- Diversity: {diversity_indicator:.4f} (ideal: 0.05-0.15)
- Convergence Rate: {convergence_rate:.2e}
- Population Quality: {population_quality:.3f}
- Phase: {phase}

{strategy_analysis}
{phase_strategy}

OPERATOR CAPABILITIES:
0: CMA_ES - Precision local search
1: Levy_Flight - Powerful global exploration  
2: JADE_DE - Adaptive differential evolution
3: Comprehensive_Learning - Diversity maintenance
4: PSO_GA - Hybrid balance
5: Opposition_Learning - Convergence acceleration
6: Local_Global_Mix - Adaptive balance
7: Self_Adaptive_Mutation - Escape local optima
8: DE_pbest - Robust performance
9: CLPSO - Comprehensive learning

STRATEGIC GUIDANCE:
- Early phase (<30%): Focus on 1,7,3 for strong exploration
- Mid phase (30-70%): Balance with 2,6,4,5
- Late phase (>70%): Refine with 0,8 but keep 20% exploration
- Always maintain 6+ different operators
- If diversity < 0.03, use 40%+ exploration operators
- If slow convergence, boost 5,0,8

Generate {PopSize} integers (0-9) that will achieve SUPERIOR performance across ALL benchmark functions.
Focus on creating a SMART, ADAPTIVE sequence that balances exploration and exploitation.

Output ONLY the sequence of {PopSize} integers:"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": f"""You are an elite optimization expert. Generate the BEST sequence of {PopSize} integers (0-9) for superior performance on ALL benchmark functions.
                 Key principles:
                 1. Early iterations: Heavy on exploration (operators 1,7,3)
                 2. Middle iterations: Balanced mix (2,6,4,5,0)
                 3. Late iterations: Focus on refinement but keep exploration (0,8,4 with some 1,7)
                 4. Always maintain high diversity - use at least 7 different operators
                 5. Adapt to low diversity by increasing operators 1,7,3
                 6. Ensure smooth transition between phases

                 Current phase: {phase}
                 Output ONLY numbers, no explanations."""},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            timeout=30
        )
        response_text = response.choices[0].message.content
        numbers = re.findall(r'\d+', response_text)

        # 处理响应
        raw_sequence = [int(num) for num in numbers[:PopSize]]
        valid_sequence = [max(0, min(num, max_op_index)) for num in raw_sequence]

        # 智能补充策略
        if len(valid_sequence) < PopSize:
            needed = PopSize - len(valid_sequence)
            supplement = []

            # 基于当前状态智能补充
            for _ in range(needed):
                if diversity_indicator < 0.03:
                    # 低多样性时补充探索算子
                    supplement.append(np.random.choice([1, 7, 3]))
                elif phase == "AGGRESSIVE_EXPLORATION":
                    supplement.append(np.random.choice([1, 7, 2, 3]))
                elif phase == "BALANCED_SEARCH":
                    supplement.append(np.random.choice([2, 6, 4, 5]))
                elif phase == "FOCUSED_REFINEMENT":
                    supplement.append(np.random.choice([0, 8, 5, 2]))
                else:  # FINAL_POLISHING
                    supplement.append(np.random.choice([0, 8, 4, 5]))

            valid_sequence.extend(supplement)

        Sequence = valid_sequence[:PopSize]

        # 强制质量保证
        unique_ops = len(set(Sequence))
        if unique_ops < 7:
            print(f"Quality enhancement: Increasing diversity from {unique_ops} to 7+ operators")
            Sequence = enhance_diversity(Sequence, recommended_operators, max_op_index)

        # 最终校验
        Sequence = [max(0, min(int(op), max_op_index)) for op in Sequence]
        operator_distribution = np.bincount(np.array(Sequence), minlength=10)

        # 性能分析
        exploration_ratio = (operator_distribution[1] + operator_distribution[7] + operator_distribution[3]) / PopSize
        exploitation_ratio = (operator_distribution[0] + operator_distribution[8] + operator_distribution[4]) / PopSize
        balance_ratio = (operator_distribution[2] + operator_distribution[5] + operator_distribution[6]) / PopSize

        print(f"ELITE Sequence Distribution: {operator_distribution}")
        print(
            f"Strategy: {exploration_ratio:.1%} Explore | {balance_ratio:.1%} Balance | {exploitation_ratio:.1%} Exploit")
        print(f"Operator Diversity: {unique_ops}/10 operators")

    except Exception as e:
        print(f"LLM call failed: {e}, using elite fallback strategy")
        # 精英降级策略
        if phase == "AGGRESSIVE_EXPLORATION":
            weights = [0.02, 0.25, 0.15, 0.12, 0.08, 0.10, 0.10, 0.12, 0.04, 0.02]
        elif phase == "BALANCED_SEARCH":
            weights = [0.10, 0.15, 0.16, 0.08, 0.12, 0.12, 0.12, 0.08, 0.04, 0.03]
        elif phase == "FOCUSED_REFINEMENT":
            weights = [0.18, 0.10, 0.12, 0.06, 0.14, 0.12, 0.10, 0.08, 0.08, 0.02]
        else:  # FINAL_POLISHING
            weights = [0.22, 0.08, 0.10, 0.05, 0.15, 0.12, 0.08, 0.08, 0.10, 0.02]

        # 根据多样性调整
        if diversity_indicator < 0.03:
            weights[1] += 0.10  # 增加Levy
            weights[7] += 0.08  # 增加变异
            weights = np.array(weights) / np.sum(weights)

        weights = np.array(weights) / np.sum(weights)
        Sequence = []
        for i in range(PopSize):
            Sequence.append(np.random.choice(range(max_op_index + 1), p=weights))
        Sequence = [max(0, min(int(op), max_op_index)) for op in Sequence]

        print(f"Fallback strategy activated for {phase} phase")

    return


def Rand():
    global Pop, FitPop, PopSize, Sequence, tmpFitPop
    Sequence = np.random.randint(0, len(Operators), PopSize)
    print(f"Random sequence: {Sequence[:10]}...")  # 显示前10个元素


def initialize_pso_parameters():
    """PSO参数初始化 - 单独的函数"""
    global PopSize, DimSize, Pop, FitPop, pbest, pbest_fit, gbest, gbest_fit, v, v_max, LB, UB

    # PSO 参数初始化
    v_max = 0.2 * (np.array(UB) - np.array(LB))  # 最大速度为搜索空间的20%
    v = np.random.uniform(-v_max, v_max, (PopSize, DimSize))  # 初始化速度

    pbest = deepcopy(Pop)  # 个体最优位置初始化为当前位置
    pbest_fit = deepcopy(FitPop)  # 个体最优适应度

    # 全局最优初始化
    best_idx = np.argmin(FitPop)
    gbest = deepcopy(Pop[best_idx])
    gbest_fit = FitPop[best_idx]


def LLMPSO(func):
    """
    LLM-PSO算法：使用LLM动态调整算子序列
    """
    global PopSize, DimSize, curIter, MaxIter, Pop, FitPop, Sequence, tmpFitPop, Off, FitOff
    global pbest, pbest_fit, gbest, gbest_fit, v, v_max, Trace

    # 使用统一的种群初始化
    Pop, FitPop = unified_initialization(func)

    # 初始化PSO特有参数
    initialize_pso_parameters()

    # 初始化算子序列
    for i in range(PopSize):
        Sequence[i] = np.random.randint(0, len(Operators))

    Trace = []
    Trace.append(gbest_fit)
    api_call_count = 0

    print("LLM-PSO: Using LLM for operator sequence selection")
    print(f"LLM-PSO initial best fitness: {gbest_fit:.6f}")

    for curIter in range(MaxIter):
        if curIter == 0:
            HighLevel(Operators)
        tmpFitPop = deepcopy(FitPop)

        for j in range(PopSize):
            # 1. PSO速度位置更新（标准PSO核心）
            w = 1.0  # 固定惯性权重，根据表中PSO参数
            c1, c2 = 2.05, 2.05  # 学习因子，根据表中PSO参数

            r1, r2 = np.random.rand(DimSize), np.random.rand(DimSize)

            # 速度更新公式
            v[j] = w * v[j] + c1 * r1 * (pbest[j] - Pop[j]) + c2 * r2 * (gbest - Pop[j])
            v[j] = np.clip(v[j], -2, 2)  # 速度边界限制，根据表中PSO参数

            # 位置更新
            Pop[j] = Pop[j] + v[j]
            Pop[j] = np.clip(Pop[j], LB, UB)

            # 评估新位置
            current_fit = func.evaluate(Pop[j])

            # 更新个体最优
            if current_fit < pbest_fit[j]:
                pbest[j] = deepcopy(Pop[j])
                pbest_fit[j] = current_fit

            # 更新全局最优
            if current_fit < gbest_fit:
                gbest = deepcopy(Pop[j])
                gbest_fit = current_fit

            # 2. 应用搜索算子进行额外探索
            Operator = Operators[int(Sequence[j])]
            Operator(j)
            FitOff[j] = func.evaluate(Off[j])

            # 如果算子生成的结果更好，则更新
            if FitOff[j] < FitPop[j]:
                FitPop[j] = FitOff[j]
                Pop[j] = deepcopy(Off[j])

                # 更新个体最优
                if FitOff[j] < pbest_fit[j]:
                    pbest[j] = deepcopy(Off[j])
                    pbest_fit[j] = FitOff[j]

                # 更新全局最优
                if FitOff[j] < gbest_fit:
                    gbest = deepcopy(Off[j])
                    gbest_fit = FitOff[j]

        # 记录当前最优适应度
        Trace.append(gbest_fit)

        # LLM-PSO: 定期调用HighLevel更新算子序列
        api_call_count += 1
        if api_call_count >= 5:
            HighLevel(Operators)
            api_call_count = 0
            print(f"LLM-PSO: Iteration {curIter}, Updated sequence")

        if curIter % 50 == 0:
            print(f"LLM-PSO: Iteration {curIter}, Best Fitness: {gbest_fit:.6f}")

    return Trace


def PSO(func):
    """
    基础PSO算法：使用随机算子序列，不调用LLM
    """
    global PopSize, DimSize, curIter, MaxIter, Pop, FitPop, Sequence, tmpFitPop, Off, FitOff
    global pbest, pbest_fit, gbest, gbest_fit, v, v_max, Trace

    # 使用统一的种群初始化
    Pop, FitPop = unified_initialization(func)

    # 初始化PSO特有参数
    initialize_pso_parameters()

    # 初始化随机算子序列
    for i in range(PopSize):
        Sequence[i] = np.random.randint(0, len(Operators))

    Trace = []
    Trace.append(gbest_fit)

    print("PSO: Using random operator sequence")
    print(f"PSO initial best fitness: {gbest_fit:.6f}")

    for curIter in range(MaxIter):
        tmpFitPop = deepcopy(FitPop)

        for j in range(PopSize):
            # 1. PSO速度位置更新（标准PSO核心）
            w = 1.0  # 固定惯性权重，根据表中PSO参数
            c1, c2 = 2.05, 2.05  # 学习因子，根据表中PSO参数

            r1, r2 = np.random.rand(DimSize), np.random.rand(DimSize)

            # 速度更新公式
            v[j] = w * v[j] + c1 * r1 * (pbest[j] - Pop[j]) + c2 * r2 * (gbest - Pop[j])
            v[j] = np.clip(v[j], -2, 2)  # 速度边界限制，根据表中PSO参数

            # 位置更新
            Pop[j] = Pop[j] + v[j]
            Pop[j] = np.clip(Pop[j], LB, UB)

            # 评估新位置
            current_fit = func.evaluate(Pop[j])

            # 更新个体最优
            if current_fit < pbest_fit[j]:
                pbest[j] = deepcopy(Pop[j])
                pbest_fit[j] = current_fit

            # 更新全局最优
            if current_fit < gbest_fit:
                gbest = deepcopy(Pop[j])
                gbest_fit = current_fit

        Trace.append(gbest_fit)

        if curIter % 50 == 0:
            print(f"PSO: Iteration {curIter}, Best Fitness: {gbest_fit:.6f}")

    return Trace


def PPSO(func):
    """
    并行粒子群优化算法 (Parallel PSO)
    """
    global PopSize, DimSize, MaxIter, LB, UB, Pop, FitPop, Trace

    # 使用统一的种群初始化
    Pop, FitPop = unified_initialization(func)

    # PPSO参数
    num_groups = 4  # 分组数量
    group_size = PopSize // num_groups
    v_max = 0.2 * (np.array(UB) - np.array(LB))

    # 初始化每个组的参数
    groups_v = [np.random.uniform(-v_max, v_max, (group_size, DimSize)) for _ in range(num_groups)]
    groups_pbest = [deepcopy(Pop[i * group_size:(i + 1) * group_size]) for i in range(num_groups)]
    groups_pbest_fit = [deepcopy(FitPop[i * group_size:(i + 1) * group_size]) for i in range(num_groups)]
    groups_gbest = [deepcopy(groups_pbest[i][np.argmin(groups_pbest_fit[i])]) for i in range(num_groups)]
    groups_gbest_fit = [np.min(groups_pbest_fit[i]) for i in range(num_groups)]

    # 全局最优
    global_best_idx = np.argmin(groups_gbest_fit)
    global_best = deepcopy(groups_gbest[global_best_idx])
    global_best_fit = groups_gbest_fit[global_best_idx]

    Trace = []
    Trace.append(global_best_fit)

    print("PPSO: Using parallel PSO with multiple groups")
    print(f"PPSO initial best fitness: {global_best_fit:.6f}")

    for iter_count in range(MaxIter):
        # 每个组独立进化
        for group_idx in range(num_groups):
            start_idx = group_idx * group_size
            end_idx = (group_idx + 1) * group_size

            w = 0.9 - (0.9 - 0.4) * (iter_count / MaxIter)
            c1, c2 = 2.0, 2.0

            for j in range(group_size):
                idx = start_idx + j
                r1, r2 = np.random.rand(DimSize), np.random.rand(DimSize)

                # 速度更新
                groups_v[group_idx][j] = (w * groups_v[group_idx][j] +
                                          c1 * r1 * (groups_pbest[group_idx][j] - Pop[idx]) +
                                          c2 * r2 * (groups_gbest[group_idx] - Pop[idx]))
                groups_v[group_idx][j] = np.clip(groups_v[group_idx][j], -v_max, v_max)

                # 位置更新
                Pop[idx] = Pop[idx] + groups_v[group_idx][j]
                Pop[idx] = np.clip(Pop[idx], LB, UB)

                # 评估
                current_fit = func.evaluate(Pop[idx])
                FitPop[idx] = current_fit

                # 更新个体最优
                if current_fit < groups_pbest_fit[group_idx][j]:
                    groups_pbest[group_idx][j] = deepcopy(Pop[idx])
                    groups_pbest_fit[group_idx][j] = current_fit

                # 更新组最优
                if current_fit < groups_gbest_fit[group_idx]:
                    groups_gbest[group_idx] = deepcopy(Pop[idx])
                    groups_gbest_fit[group_idx] = current_fit

                # 更新全局最优
                if current_fit < global_best_fit:
                    global_best = deepcopy(Pop[idx])
                    global_best_fit = current_fit

        # 组间信息交换（每10代）
        if iter_count % 10 == 0:
            # 随机选择两个组交换最优个体
            group1, group2 = np.random.choice(num_groups, 2, replace=False)
            swap_idx = np.random.randint(group_size)
            groups_pbest[group1][swap_idx] = deepcopy(groups_gbest[group2])
            groups_pbest_fit[group1][swap_idx] = groups_gbest_fit[group2]

        Trace.append(global_best_fit)

        if iter_count % 50 == 0:
            print(f"PPSO: Iteration {iter_count}, Best Fitness: {global_best_fit:.6f}")

    return Trace


def CLPSO(func):
    """
    综合学习粒子群优化算法 (Comprehensive Learning PSO)
    """
    global PopSize, DimSize, MaxIter, LB, UB, Pop, FitPop, Trace

    # 使用统一的种群初始化
    Pop, FitPop = unified_initialization(func)

    # CLPSO参数 - 根据表中参数
    c_local = 1.2  # 局部系数
    w_max, w_min = 0.9, 0.4  # 最大和最小权重

    # 初始化速度
    v_max = 0.2 * (np.array(UB) - np.array(LB))
    v = np.random.uniform(-v_max, v_max, (PopSize, DimSize))

    # 初始化个体最优
    pbest = deepcopy(Pop)
    pbest_fit = deepcopy(FitPop)

    # 全局最优
    best_idx = np.argmin(FitPop)
    gbest = deepcopy(Pop[best_idx])
    gbest_fit = FitPop[best_idx]

    # 学习范例矩阵
    learning_exemplars = np.zeros((PopSize, DimSize), dtype=int)
    refresh_count = 0
    refresh_gap = 7  # 刷新间隔

    Trace = []
    Trace.append(gbest_fit)

    print("CL-PSO: Using Comprehensive Learning PSO")
    print(f"CL-PSO initial best fitness: {gbest_fit:.6f}")

    def update_learning_exemplars():
        """更新学习范例"""
        nonlocal learning_exemplars

        for i in range(PopSize):
            for d in range(DimSize):
                if np.random.rand() < 0.3:  # 学习概率
                    # 随机选择两个不同的个体
                    candidates = [j for j in range(PopSize) if j != i]
                    r1, r2 = np.random.choice(candidates, 2, replace=False)

                    # 选择适应度更好的个体作为学习对象
                    if pbest_fit[r1] < pbest_fit[r2]:
                        learning_exemplars[i, d] = r1
                    else:
                        learning_exemplars[i, d] = r2
                else:
                    # 向自身历史最优学习
                    learning_exemplars[i, d] = i

    # 初始更新学习范例
    update_learning_exemplars()

    for iter_count in range(MaxIter):
        # 周期性更新学习范例
        refresh_count += 1
        if refresh_count >= refresh_gap:
            update_learning_exemplars()
            refresh_count = 0

        # 自适应惯性权重
        w_current = w_max - (w_max - w_min) * (iter_count / MaxIter)

        for i in range(PopSize):
            # 构建学习目标
            learning_target = np.zeros(DimSize)
            for d in range(DimSize):
                teacher_idx = learning_exemplars[i, d]
                learning_target[d] = pbest[teacher_idx, d]

            # 速度更新
            r = np.random.rand(DimSize)
            v[i] = (w_current * v[i] +
                    c_local * r * (learning_target - Pop[i]))
            v[i] = np.clip(v[i], -v_max, v_max)

            # 位置更新
            Pop[i] = Pop[i] + v[i]
            Pop[i] = np.clip(Pop[i], LB, UB)

            # 评估新位置
            current_fit = func.evaluate(Pop[i])

            # 更新个体最优
            if current_fit < pbest_fit[i]:
                pbest[i] = deepcopy(Pop[i])
                pbest_fit[i] = current_fit

                # 更新全局最优
                if current_fit < gbest_fit:
                    gbest = deepcopy(Pop[i])
                    gbest_fit = current_fit

        Trace.append(gbest_fit)

        if iter_count % 50 == 0:
            print(f"CL-PSO: Iteration {iter_count}, Best Fitness: {gbest_fit:.6f}")

    return Trace


def DMS_PSO(func):
    """
    动态多群粒子群优化算法 (Dynamic Multi-Swarm PSO)
    """
    global PopSize, DimSize, MaxIter, LB, UB, Pop, FitPop, Trace

    # 使用统一的种群初始化
    Pop, FitPop = unified_initialization(func)

    # DMS-PSO参数
    num_subswarms = 5  # 子群数量
    subswarm_size = PopSize // num_subswarms
    regrouping_period = 10  # 重组周期
    v_max = 0.2 * (np.array(UB) - np.array(LB))

    # 初始化子群
    subswarms = []
    subswarms_pbest = []
    subswarms_pbest_fit = []
    subswarms_gbest = []
    subswarms_gbest_fit = []
    subswarms_v = []

    # 随机分配粒子到子群
    indices = np.random.permutation(PopSize)
    for i in range(num_subswarms):
        start_idx = i * subswarm_size
        end_idx = start_idx + subswarm_size

        # 最后一个子群可能包含剩余的粒子
        if i == num_subswarms - 1:
            end_idx = PopSize

        subswarm_indices = indices[start_idx:end_idx]
        subswarm = Pop[subswarm_indices]
        subswarm_fit = FitPop[subswarm_indices]

        # 初始化子群的速度
        subswarm_v = np.random.uniform(-v_max, v_max, (len(subswarm_indices), DimSize))

        # 初始化子群的个体最优
        subswarm_pbest = deepcopy(subswarm)
        subswarm_pbest_fit = deepcopy(subswarm_fit)

        # 初始化子群的全局最优
        best_idx = np.argmin(subswarm_fit)
        subswarm_gbest = deepcopy(subswarm[best_idx])
        subswarm_gbest_fit = subswarm_fit[best_idx]

        subswarms.append(subswarm)
        subswarms_pbest.append(subswarm_pbest)
        subswarms_pbest_fit.append(subswarm_pbest_fit)
        subswarms_gbest.append(subswarm_gbest)
        subswarms_gbest_fit.append(subswarm_gbest_fit)
        subswarms_v.append(subswarm_v)

    # 全局最优
    global_best_idx = np.argmin(subswarms_gbest_fit)
    global_best = deepcopy(subswarms_gbest[global_best_idx])
    global_best_fit = subswarms_gbest_fit[global_best_idx]

    Trace = []
    Trace.append(global_best_fit)

    print("DMS-PSO: Using Dynamic Multi-Swarm PSO")
    print(f"DMS-PSO initial best fitness: {global_best_fit:.6f}")
    print(f"Subswarms: {num_subswarms}, Size: {subswarm_size}, Regrouping period: {regrouping_period}")

    for iter_count in range(MaxIter):
        # 每个子群独立进化
        for swarm_idx in range(num_subswarms):
            current_swarm = subswarms[swarm_idx]
            current_pbest = subswarms_pbest[swarm_idx]
            current_pbest_fit = subswarms_pbest_fit[swarm_idx]
            current_gbest = subswarms_gbest[swarm_idx]
            current_v = subswarms_v[swarm_idx]

            swarm_size = len(current_swarm)

            # 自适应惯性权重
            w = 0.9 - (0.9 - 0.4) * (iter_count / MaxIter)
            c1, c2 = 2.0, 2.0

            for j in range(swarm_size):
                r1, r2 = np.random.rand(DimSize), np.random.rand(DimSize)

                # 速度更新
                current_v[j] = (w * current_v[j] +
                                c1 * r1 * (current_pbest[j] - current_swarm[j]) +
                                c2 * r2 * (current_gbest - current_swarm[j]))
                current_v[j] = np.clip(current_v[j], -v_max, v_max)

                # 位置更新
                current_swarm[j] = current_swarm[j] + current_v[j]
                current_swarm[j] = np.clip(current_swarm[j], LB, UB)

                # 评估新位置
                current_fit = func.evaluate(current_swarm[j])

                # 更新个体最优
                if current_fit < current_pbest_fit[j]:
                    current_pbest[j] = deepcopy(current_swarm[j])
                    current_pbest_fit[j] = current_fit

                # 更新子群最优
                if current_fit < subswarms_gbest_fit[swarm_idx]:
                    subswarms_gbest[swarm_idx] = deepcopy(current_swarm[j])
                    subswarms_gbest_fit[swarm_idx] = current_fit

                # 更新全局最优
                if current_fit < global_best_fit:
                    global_best = deepcopy(current_swarm[j])
                    global_best_fit = current_fit

            # 更新子群数据
            subswarms[swarm_idx] = current_swarm
            subswarms_pbest[swarm_idx] = current_pbest
            subswarms_pbest_fit[swarm_idx] = current_pbest_fit
            subswarms_v[swarm_idx] = current_v

        # 周期性重组（动态特性）
        if iter_count % regrouping_period == 0:
            # 收集所有粒子
            all_particles = np.vstack(subswarms)
            all_fitness = np.hstack([subswarm_fit for subswarm_fit in subswarms_pbest_fit])

            # 按适应度排序
            sorted_indices = np.argsort(all_fitness)
            all_particles_sorted = all_particles[sorted_indices]
            all_fitness_sorted = all_fitness[sorted_indices]

            # 重新分配到子群
            new_subswarms = []
            new_subswarms_pbest = []
            new_subswarms_pbest_fit = []
            new_subswarms_gbest = []
            new_subswarms_gbest_fit = []
            new_subswarms_v = []

            for i in range(num_subswarms):
                start_idx = i * subswarm_size
                end_idx = start_idx + subswarm_size

                if i == num_subswarms - 1:
                    end_idx = PopSize

                new_subswarm = all_particles_sorted[start_idx:end_idx]
                new_subswarm_fit = all_fitness_sorted[start_idx:end_idx]

                # 重新初始化速度
                new_subswarm_v = np.random.uniform(-v_max, v_max, (len(new_subswarm), DimSize))

                # 更新个体最优（保持历史最优）
                new_subswarm_pbest = deepcopy(new_subswarm)
                new_subswarm_pbest_fit = deepcopy(new_subswarm_fit)

                # 更新子群最优
                best_idx = np.argmin(new_subswarm_fit)
                new_subswarm_gbest = deepcopy(new_subswarm[best_idx])
                new_subswarm_gbest_fit = new_subswarm_fit[best_idx]

                new_subswarms.append(new_subswarm)
                new_subswarms_pbest.append(new_subswarm_pbest)
                new_subswarms_pbest_fit.append(new_subswarm_pbest_fit)
                new_subswarms_gbest.append(new_subswarm_gbest)
                new_subswarms_gbest_fit.append(new_subswarm_gbest_fit)
                new_subswarms_v.append(new_subswarm_v)

            # 更新子群
            subswarms = new_subswarms
            subswarms_pbest = new_subswarms_pbest
            subswarms_pbest_fit = new_subswarms_pbest_fit
            subswarms_gbest = new_subswarms_gbest
            subswarms_gbest_fit = new_subswarms_gbest_fit
            subswarms_v = new_subswarms_v

            # 更新全局最优
            global_best_idx = np.argmin(subswarms_gbest_fit)
            global_best = deepcopy(subswarms_gbest[global_best_idx])
            global_best_fit = subswarms_gbest_fit[global_best_idx]

            if iter_count % 50 == 0:  # 减少输出频率
                print(f"DMS-PSO: Iteration {iter_count}, Regrouping completed")

        # 子群间信息交流（随机迁移）
        if iter_count % 7 == 0:  # 每7代进行一次迁移
            # 随机选择两个子群
            swarm1, swarm2 = np.random.choice(num_subswarms, 2, replace=False)

            # 随机选择要交换的粒子
            idx1 = np.random.randint(len(subswarms[swarm1]))
            idx2 = np.random.randint(len(subswarms[swarm2]))

            # 交换粒子
            subswarms[swarm1][idx1], subswarms[swarm2][idx2] = \
                deepcopy(subswarms[swarm2][idx2]), deepcopy(subswarms[swarm1][idx1])

            # 交换个体最优
            subswarms_pbest[swarm1][idx1], subswarms_pbest[swarm2][idx2] = \
                deepcopy(subswarms_pbest[swarm2][idx2]), deepcopy(subswarms_pbest[swarm1][idx1])

            subswarms_pbest_fit[swarm1][idx1], subswarms_pbest_fit[swarm2][idx2] = \
                subswarms_pbest_fit[swarm2][idx2], subswarms_pbest_fit[swarm1][idx1]

        Trace.append(global_best_fit)

        if iter_count % 50 == 0:
            # 计算子群多样性
            diversity_scores = []
            for swarm in subswarms:
                if len(swarm) > 1:
                    diversity = np.mean(np.std(swarm, axis=0))
                    diversity_scores.append(diversity)

            avg_diversity = np.mean(diversity_scores) if diversity_scores else 0
            print(f"DMS-PSO: Iteration {iter_count}, Best Fitness: {global_best_fit:.6f}, "
                  f"Avg Diversity: {avg_diversity:.4f}")

    return Trace


def PSO_GA(func):

    global PopSize, DimSize, curIter, MaxIter, Pop, FitPop, Sequence, tmpFitPop, Off, FitOff
    global pbest, pbest_fit, gbest, gbest_fit, v, v_max, Trace

    # 使用统一的种群初始化
    Pop, FitPop = unified_initialization(func)

    # 初始化PSO特有参数
    initialize_pso_parameters()

    # GA参数
    crossover_rate = 0.8  # 交叉概率
    mutation_rate = 0.1  # 变异概率
    tournament_size = 3  # 锦标赛选择的大小

    Trace = []
    Trace.append(gbest_fit)

    print("PSO-GA: Using hybrid PSO with GA operators")
    print(f"PSO-GA initial best fitness: {gbest_fit:.6f}")

    for iter_count in range(MaxIter):
        # 1. PSO更新阶段
        w = 0.9 - (0.9 - 0.4) * (iter_count / MaxIter)
        c1, c2 = 2.0, 2.0

        for j in range(PopSize):
            r1, r2 = np.random.rand(DimSize), np.random.rand(DimSize)

            # 速度更新
            v[j] = w * v[j] + c1 * r1 * (pbest[j] - Pop[j]) + c2 * r2 * (gbest - Pop[j])
            v[j] = np.clip(v[j], -v_max, v_max)

            # 位置更新
            Pop[j] = Pop[j] + v[j]
            Pop[j] = np.clip(Pop[j], LB, UB)

            # 评估新位置
            current_fit = func.evaluate(Pop[j])

            # 更新个体最优
            if current_fit < pbest_fit[j]:
                pbest[j] = deepcopy(Pop[j])
                pbest_fit[j] = current_fit

            # 更新全局最优
            if current_fit < gbest_fit:
                gbest = deepcopy(Pop[j])
                gbest_fit = current_fit

        # 2. GA操作阶段（每5代执行一次）
        if iter_count % 5 == 0:
            # 创建新种群用于GA操作
            new_pop = np.zeros_like(Pop)
            new_fit = np.zeros(PopSize)

            # 锦标赛选择
            for i in range(PopSize):
                # 随机选择tournament_size个个体
                candidates = np.random.choice(PopSize, tournament_size, replace=False)
                # 选择适应度最好的
                best_candidate = candidates[np.argmin(FitPop[candidates])]
                new_pop[i] = deepcopy(Pop[best_candidate])
                new_fit[i] = FitPop[best_candidate]

            # 交叉操作
            for i in range(0, PopSize - 1, 2):
                if np.random.rand() < crossover_rate:
                    parent1 = new_pop[i]
                    parent2 = new_pop[i + 1]

                    # 模拟二进制交叉
                    beta = np.random.rand(DimSize)
                    child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
                    child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)

                    child1 = np.clip(child1, LB, UB)
                    child2 = np.clip(child2, LB, UB)

                    new_pop[i] = child1
                    new_pop[i + 1] = child2

            # 变异操作
            for i in range(PopSize):
                if np.random.rand() < mutation_rate:
                    # 高斯变异
                    mutation_strength = 0.1 * (UB[0] - LB[0]) * (1 - iter_count / MaxIter)
                    mutation = np.random.normal(0, mutation_strength, DimSize)
                    new_pop[i] = new_pop[i] + mutation
                    new_pop[i] = np.clip(new_pop[i], LB, UB)

            # 评估新种群
            for i in range(PopSize):
                new_fit[i] = func.evaluate(new_pop[i])

            # 精英保留：用GA新种群替换当前种群，但保留全局最优
            best_ga_idx = np.argmin(new_fit)
            if new_fit[best_ga_idx] < gbest_fit:
                gbest = deepcopy(new_pop[best_ga_idx])
                gbest_fit = new_fit[best_ga_idx]

            Pop = deepcopy(new_pop)
            FitPop = deepcopy(new_fit)

            # 更新个体最优
            for j in range(PopSize):
                if FitPop[j] < pbest_fit[j]:
                    pbest[j] = deepcopy(Pop[j])
                    pbest_fit[j] = FitPop[j]

        Trace.append(gbest_fit)

        if iter_count % 50 == 0:
            print(f"PSO-GA: Iteration {iter_count}, Best Fitness: {gbest_fit:.6f}")

    return Trace


def SADE(func):
    """
    自适应差分进化算法 (Self-adaptive Differential Evolution)
    """
    global PopSize, DimSize, MaxIter, LB, UB, Pop, FitPop, Trace

    # 使用统一的种群初始化
    Pop, FitPop = unified_initialization(func)

    # SADE参数 - 根据表中参数
    μ_F, σ_F = 0.5, 0.3
    μ_Cr, σ_Cr = 0.5, 0.1

    # 初始化F和Cr
    F = np.random.normal(μ_F, σ_F, PopSize)
    Cr = np.random.normal(μ_Cr, σ_Cr, PopSize)

    # 限制参数范围
    F = np.clip(F, 0.1, 1.0)
    Cr = np.clip(Cr, 0.0, 1.0)

    # 记录最优适应度
    best_idx = np.argmin(FitPop)
    best_fit = FitPop[best_idx]
    best_solution = deepcopy(Pop[best_idx])

    Trace = []
    Trace.append(best_fit)

    print("SaDE: Using self-adaptive DE")
    print(f"SaDE initial best fitness: {best_fit:.6f}")

    # 成功参数记忆
    success_F = []
    success_Cr = []

    for iter_count in range(MaxIter):
        for i in range(PopSize):
            # 随机选择变异策略
            if np.random.rand() < 0.5:
                # DE/rand/1
                candidates = [x for x in range(PopSize) if x != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                mutant = Pop[r1] + F[i] * (Pop[r2] - Pop[r3])
            else:
                # DE/cur-to-best/1
                candidates = [x for x in range(PopSize) if x != i]
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                mutant = Pop[i] + F[i] * (best_solution - Pop[i]) + F[i] * (Pop[r1] - Pop[r2])

            mutant = np.clip(mutant, LB, UB)

            # 交叉操作
            trial = np.copy(Pop[i])
            j_rand = np.random.randint(DimSize)

            for j in range(DimSize):
                if np.random.rand() < Cr[i] or j == j_rand:
                    trial[j] = mutant[j]

            # 选择操作
            trial_fit = func.evaluate(trial)
            if trial_fit < FitPop[i]:
                # 记录成功的参数
                success_F.append(F[i])
                success_Cr.append(Cr[i])

                Pop[i] = trial
                FitPop[i] = trial_fit

                if trial_fit < best_fit:
                    best_fit = trial_fit
                    best_solution = deepcopy(trial)

        # 自适应更新F和Cr
        if len(success_F) > 0:
            # 更新F的均值和标准差
            μ_F = np.mean(success_F)
            σ_F = np.std(success_F)
            if σ_F < 0.1:  # 防止标准差过小
                σ_F = 0.1

            # 更新Cr的均值和标准差
            μ_Cr = np.mean(success_Cr)
            σ_Cr = np.std(success_Cr)
            if σ_Cr < 0.05:  # 防止标准差过小
                σ_Cr = 0.05

            # 生成新的参数
            F = np.random.normal(μ_F, σ_F, PopSize)
            Cr = np.random.normal(μ_Cr, σ_Cr, PopSize)

            # 限制参数范围
            F = np.clip(F, 0.1, 1.0)
            Cr = np.clip(Cr, 0.0, 1.0)

        Trace.append(best_fit)

        if iter_count % 50 == 0:
            print(f"SaDE: Iteration {iter_count}, Best Fitness: {best_fit:.6f}")

    return Trace


import numpy as np
import os
from copy import deepcopy


def unified_initialization(func):
    global PopSize, DimSize, LB, UB
    Pop = np.random.uniform(LB, UB, (PopSize, DimSize))
    FitPop = np.array([func.evaluate(ind) for ind in Pop])
    return Pop, FitPop


def WDE(func):
    """
    加权差分进化算法 (WDE) - 参数自由版本
    """
    global PopSize, DimSize, MaxIter, LB, UB, Pop, FitPop, Trace

    # 统一初始化种群
    Pop, FitPop = unified_initialization(func)

    # WDE是参数自由算法，不需要设置F和Cr
    # 自适应参数机制
    F_mean = 0.5
    Cr_mean = 0.5

    # 初始化最优解
    best_idx = np.argmin(FitPop)
    best_fit = FitPop[best_idx]
    best_solution = deepcopy(Pop[best_idx])

    # 初始化轨迹记录
    Trace = []
    Trace.append(best_fit)

    print("WDE: Using Weighted Differential Evolution (Parameter-free)")
    print(f"WDE initial best fitness: {best_fit:.6f}")

    # 成功参数记录
    success_F = []
    success_Cr = []

    for iter_count in range(MaxIter):
        # 自适应调整参数
        if len(success_F) > 0:
            F_mean = np.mean(success_F[-10:]) if len(success_F) >= 10 else np.mean(success_F)
            Cr_mean = np.mean(success_Cr[-10:]) if len(success_Cr) >= 10 else np.mean(success_Cr)

        # 为每个个体生成参数
        F = np.random.normal(F_mean, 0.1, PopSize)
        Cr = np.random.normal(Cr_mean, 0.1, PopSize)
        F = np.clip(F, 0.1, 1.0)
        Cr = np.clip(Cr, 0.0, 1.0)

        # 计算个体权重（适应度越好，权重越高）
        fit_min = np.min(FitPop)
        fit_max = np.max(FitPop)
        if fit_max - fit_min < 1e-8:
            weights = np.ones(PopSize) / PopSize
        else:
            norm_fit = (fit_max - FitPop) / (fit_max - fit_min)
            weights = norm_fit / np.sum(norm_fit)

        for i in range(PopSize):
            # 加权选择3个不同的变异个体
            candidates = np.delete(np.arange(PopSize), i)
            r1 = np.random.choice(candidates, p=weights[candidates] / np.sum(weights[candidates]))
            candidates_r2 = np.delete(candidates, np.where(candidates == r1))
            r2 = np.random.choice(candidates_r2, p=weights[candidates_r2] / np.sum(weights[candidates_r2]))
            candidates_r3 = np.delete(candidates_r2, np.where(candidates_r2 == r2))
            r3 = np.random.choice(candidates_r3, p=weights[candidates_r3] / np.sum(weights[candidates_r3]))

            # 变异操作
            mutant = Pop[r1] + F[i] * (Pop[r2] - Pop[r3])
            mutant = np.clip(mutant, LB, UB)

            # 交叉操作
            trial = np.copy(Pop[i])
            j_rand = np.random.randint(DimSize)
            for j in range(DimSize):
                if np.random.rand() < Cr[i] or j == j_rand:
                    trial[j] = mutant[j]

            # 选择操作
            trial_fit = func.evaluate(trial)
            if trial_fit < FitPop[i]:
                # 记录成功参数
                success_F.append(F[i])
                success_Cr.append(Cr[i])

                Pop[i] = trial
                FitPop[i] = trial_fit
                if trial_fit < best_fit:
                    best_fit = trial_fit
                    best_solution = deepcopy(trial)

        Trace.append(best_fit)

        if iter_count % 50 == 0:
            print(f"WDE: Iteration {iter_count}, Best Fitness: {best_fit:.6f}")

    return Trace


def DE(func):
    """
    标准差分进化算法 (DE/cur-to-rand/1)
    """
    global PopSize, DimSize, MaxIter, LB, UB, Pop, FitPop, Trace

    # DE算法参数 - 根据表中参数
    F = 0.8  # 缩放因子
    Cr = 0.9  # 交叉概率

    # 使用统一的初始化函数
    Pop, FitPop = unified_initialization(func)

    # 记录最优适应度
    best_idx = np.argmin(FitPop)
    best_fit = FitPop[best_idx]
    best_solution = deepcopy(Pop[best_idx])

    Trace = []
    Trace.append(best_fit)

    print("DE: Using DE/cur-to-rand/1 strategy")
    print(f"DE initial best fitness: {best_fit:.6f}")

    for iter_count in range(MaxIter):
        for i in range(PopSize):
            # 选择三个不同的个体
            candidates = [x for x in range(PopSize) if x != i]
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

            # 变异操作 - DE/cur-to-rand/1策略
            mutant = Pop[i] + F * (Pop[r1] - Pop[i]) + F * (Pop[r2] - Pop[r3])
            mutant = np.clip(mutant, LB, UB)

            # 交叉操作
            trial = np.copy(Pop[i])
            j_rand = np.random.randint(DimSize)

            for j in range(DimSize):
                if np.random.rand() < Cr or j == j_rand:
                    trial[j] = mutant[j]

            # 选择操作
            trial_fit = func.evaluate(trial)
            if trial_fit < FitPop[i]:
                Pop[i] = trial
                FitPop[i] = trial_fit

                if trial_fit < best_fit:
                    best_fit = trial_fit
                    best_solution = deepcopy(trial)

        Trace.append(best_fit)

        if iter_count % 50 == 0:
            print(f"DE: Iteration {iter_count}, Best Fitness: {best_fit:.6f}")

    return Trace

def main(dim):
    global DimSize, LB, UB, MaxFEs, MaxIter, Trials, PopSize, Pop, Off
    DimSize = dim
    LB = [-100] * dim
    UB = [100] * dim

    Pop = np.zeros((PopSize, DimSize))
    Off = np.zeros((PopSize, DimSize))
    PopSize = 100
    MaxFEs = 1000 * dim
    MaxIter = 500
    Trials = 3

    CEC2020 = [F82020(dim)]  # 测试CEC2020函数

    for i in range(len(CEC2020)):
        # 1. 初始化存储10次实验轨迹的列表（仅将shade改为wde）
        llm_pso_all_trials = []
        pso_all_trials = []
        ppso_all_trials = []
        sade_all_trials = []
        wde_all_trials = []  # 替换shade_all_trials为wde_all_trials
        de_all_trials = []
        dms_pso_all_trials = []
        clpso_all_trials = []
        pso_ga_all_trials = []

        print(f"Testing F{i + 1}_{dim}D...")

        for j in range(Trials):  # Trials=10，循环10次
            print(f"\n--- Trial {j + 1} ---")


            np.random.seed(2025 + 7 * j)
            llm_pso_trace = LLMPSO(CEC2020[i])
            llm_pso_all_trials.append(llm_pso_trace)

            np.random.seed(2025 + 7 * j)
            pso_trace = PSO(CEC2020[i])
            pso_all_trials.append(pso_trace)

            np.random.seed(2025 + 7 * j)
            ppso_trace = PPSO(CEC2020[i])
            ppso_all_trials.append(ppso_trace)

            np.random.seed(2025 + 7 * j)
            sade_trace = SADE(CEC2020[i])
            sade_all_trials.append(sade_trace)

            np.random.seed(2025 + 7 * j)
            wde_trace = WDE(CEC2020[i])  # 替换SHADE为WDE
            wde_all_trials.append(wde_trace)

            np.random.seed(2025 + 7 * j)
            de_trace = DE(CEC2020[i])
            de_all_trials.append(de_trace)

            np.random.seed(2025 + 7 * j)
            dms_pso_trace = DMS_PSO(CEC2020[i])
            dms_pso_all_trials.append(dms_pso_trace)

            np.random.seed(2025 + 7 * j)
            clpso_trace = CLPSO(CEC2020[i])
            clpso_all_trials.append(clpso_trace)

            np.random.seed(2025 + 7 * j)
            pso_ga_trace = PSO_GA(CEC2020[i])
            pso_ga_all_trials.append(pso_ga_trace)

        def calculate_average_trace(trials_traces):
            """
            输入：10次实验的适应度轨迹列表
            输出：10次实验的平均适应度轨迹（长度=最大迭代步）
            """
            max_len = max(len(trace) for trace in trials_traces)
            padded_traces = []
            for trace in trials_traces:
                if len(trace) < max_len:
                    padded = trace + [trace[-1]] * (max_len - len(trace))
                else:
                    padded = trace[:max_len]
                padded_traces.append(padded)
            padded_array = np.array(padded_traces)  # shape: (10, max_len)
            avg_trace = np.mean(padded_array, axis=0)  # shape: (max_len,)
            return avg_trace

        llm_pso_avg = calculate_average_trace(llm_pso_all_trials)
        pso_avg = calculate_average_trace(pso_all_trials)
        ppso_avg = calculate_average_trace(ppso_all_trials)
        sade_avg = calculate_average_trace(sade_all_trials)
        wde_avg = calculate_average_trace(wde_all_trials)
        de_avg = calculate_average_trace(de_all_trials)
        dms_pso_avg = calculate_average_trace(dms_pso_all_trials)
        clpso_avg = calculate_average_trace(clpso_all_trials)
        pso_ga_avg = calculate_average_trace(pso_ga_all_trials)

        average_results = np.vstack([
            llm_pso_avg,
            pso_avg,
            ppso_avg,
            sade_avg,
            wde_avg,
            de_avg,
            dms_pso_avg,
            clpso_avg,
            pso_ga_avg
        ])

        save_path = f"./LLMPSO_Data/CEC2020_50D/F{i + 8}_{dim}D_Average.csv"
        np.savetxt(save_path, average_results, delimiter=",", fmt="%.6f")  # 不添加.T

        print(f"已保存F{i + 3}_{dim}D 10次实验平均值数据，形状: {average_results.shape}")
        print(f"平均值数据维度说明：行=算法数(9)，列=迭代步数({average_results.shape[1]})")

if __name__ == "__main__":
    if os.path.exists('LLMPSO_Data/CEC2020_50D') == False:
        os.makedirs('LLMPSO_Data/CEC2020_50D')
    Dims = [50]
    for dim in Dims:
        main(dim)


