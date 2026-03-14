from copy import deepcopy
from scipy.stats import levy_stable
from enoppy.paper_based.pdo_2022 import *
from openai import OpenAI
import os
import re
import numpy as np

client = OpenAI(
    api_key='Your API',
    base_url="deepseek.com")

# ========== 全局变量初始化 ==========
PopSize = 100
DimSize = 4  # 会在运行时动态更新为当前问题维度
LB = [-100] * DimSize
UB = [100] * DimSize
MaxFEs = 10000
MaxIter = int(MaxFEs / PopSize)
curIter = 0
Trials = 10  # 实验次数

# 种群/适应度数组（动态初始化）
Pop = np.zeros((PopSize, DimSize))
Off = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)
FitOff = np.zeros(PopSize)
Trace = []

# PSO 特有参数
pbest = None
gbest = None
pbest_fit = None
gbest_fit = None
v = None
v_max = None

memory_size = 5
F_memory = np.random.uniform(0.4, 0.9, memory_size)
Cr_memory = np.random.uniform(0.7, 0.95, memory_size)
memory_pos = 0
archive = []
tmpFitPop = deepcopy(FitPop)


# ========== 通用工具函数 ==========
def safe_evaluate(func, ind):
    """安全评估函数：提取标量，消除NumPy弃用警告"""
    fit_val = func.evaluate(ind)
    if isinstance(fit_val, np.ndarray):
        return fit_val.item()
    elif isinstance(fit_val, list):
        return fit_val[0] if len(fit_val) > 0 else float('inf')
    else:
        return float(fit_val)


def unified_initialization(func):
    """统一的种群初始化函数，动态适配问题维度"""
    global Pop, FitPop, PopSize, DimSize, LB, UB

    # 同步当前问题的维度和边界
    DimSize = func._n_dims
    LB = func.lb
    UB = func.ub

    # 重新初始化种群数组（适配新维度）
    Pop = np.zeros((PopSize, DimSize))
    FitPop = np.zeros(PopSize)

    # 随机初始化种群
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + np.random.rand() * (UB[j] - LB[j])
        FitPop[i] = safe_evaluate(func, Pop[i])

    return deepcopy(Pop), deepcopy(FitPop)


# ========== 算子定义 ==========
def Enhanced_CMA_ES_Operator(i):
    """增强的CMA-ES算子"""
    global Pop, gbest, curIter, MaxIter, DimSize, LB, UB
    sigma = 0.2 * (1 - curIter / MaxIter) ** 1.5 + 0.05

    if curIter < MaxIter * 0.4:
        cov_matrix = np.eye(DimSize) * sigma
    else:
        pop_cov = np.cov(Pop.T)
        eigenvals = np.linalg.eigvals(pop_cov)
        min_eigenval = np.min(np.abs(eigenvals))
        if min_eigenval < 1e-10:
            cov_matrix = np.eye(DimSize) * sigma
        else:
            cov_matrix = 0.6 * pop_cov + 0.4 * np.eye(DimSize) * sigma

    sample = np.random.multivariate_normal(np.zeros(DimSize), cov_matrix)
    exploration_weight = 0.3 * (1 - curIter / MaxIter) + 0.1
    exploitation_weight = 0.6 + 0.2 * np.sin(curIter / 15)

    Off[i] = (gbest * exploitation_weight +
              Pop[i] * exploration_weight +
              0.1 * sample)
    Off[i] = np.clip(Off[i], LB, UB)


def Adaptive_Levy_Flight(i):
    """自适应Levy飞行算子"""
    global Pop, gbest, pbest, curIter, MaxIter, LB, UB
    base_scale = 0.1 * (UB[0] - LB[0])
    scale = base_scale / ((curIter * 0.1 + 1) ** 0.5)

    beta = 0.3 + 0.4 * np.sin(curIter / 20)
    beta = np.clip(beta, -0.9, 0.9)

    strategy = np.random.choice(['global', 'personal', 'diverse'])
    if strategy == 'global':
        levy_step = levy_stable.rvs(alpha=1.0, beta=beta, size=DimSize) * scale
        Off[i] = gbest + levy_step
    elif strategy == 'personal':
        levy_step = levy_stable.rvs(alpha=1.0, beta=beta, size=DimSize) * scale * 0.7
        Off[i] = pbest[i] + levy_step
    else:
        random_idx = np.random.randint(PopSize)
        levy_step = levy_stable.rvs(alpha=1.0, beta=beta, size=DimSize) * scale * 0.5
        Off[i] = Pop[random_idx] + levy_step

    Off[i] = np.clip(Off[i], LB, UB)


def JADE_style_DE(i):
    """JADE风格的自适应差分进化"""
    global Pop, pbest, pbest_fit, PopSize, gbest
    mu_F = 0.5
    mu_Cr = 0.9

    F = np.random.normal(mu_F, 0.1)
    F = np.clip(F, 0.05, 0.95)
    Cr = np.random.normal(mu_Cr, 0.05)
    Cr = np.clip(Cr, 0.7, 0.99)

    pbest_size = max(2, PopSize // 4)
    pbest_indices = np.argsort(pbest_fit)[:pbest_size]
    pbest_idx = np.random.choice(pbest_indices)

    candidates = [x for x in range(PopSize) if x != i]
    r1, r2 = np.random.choice(candidates, 2, replace=False)

    mutant = Pop[i] + F * (pbest[pbest_idx] - Pop[i]) + F * (Pop[r1] - Pop[r2])
    trial = np.copy(Pop[i])
    j_rand = np.random.randint(DimSize)
    cross_points = np.random.rand(DimSize) < Cr
    cross_points[j_rand] = True
    trial[cross_points] = mutant[cross_points]

    Off[i] = np.clip(trial, LB, UB)


def Comprehensive_Learning(i):
    """增强的综合学习策略"""
    global Pop, pbest, pbest_fit, PopSize, curIter, MaxIter, LB, UB
    Pc = 0.05 + 0.1 * (1 - curIter / MaxIter)
    learning_exemplar = np.zeros(DimSize)

    for d in range(DimSize):
        if np.random.rand() < Pc:
            candidates = [j for j in range(PopSize) if j != i]
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            if pbest_fit[r1] < pbest_fit[r2]:
                teacher = r1
            else:
                teacher = r2
            learning_exemplar[d] = pbest[teacher, d]
        else:
            learning_exemplar[d] = pbest[i, d]

    if 'v' in globals():
        w = 0.9 - (0.9 - 0.4) * (curIter / MaxIter)
        Off[i] = Pop[i] + w * v[i] + 0.5 * (learning_exemplar - Pop[i])
    else:
        Off[i] = Pop[i] + 0.5 * (learning_exemplar - Pop[i])

    Off[i] = np.clip(Off[i], LB, UB)


def Hybrid_PSO_GA(i):
    """PSO-GA混合算子"""
    global Pop, pbest, gbest, v, curIter, MaxIter, PopSize, LB, UB
    operation = np.random.choice(['pso', 'crossover', 'mutation'],
                                 p=[0.5, 0.35, 0.15])

    if operation == 'pso':
        w = 0.9 - (0.9 - 0.4) * (curIter / MaxIter)
        c1, c2 = 1.8 + 0.4 * np.sin(curIter / 25), 1.8 + 0.4 * np.cos(curIter / 25)
        r1, r2 = np.random.rand(DimSize), np.random.rand(DimSize)

        v[i] = w * v[i] + c1 * r1 * (pbest[i] - Pop[i]) + c2 * r2 * (gbest - Pop[i])
        v[i] = np.clip(v[i], -v_max, v_max)
        Off[i] = Pop[i] + v[i]

    elif operation == 'crossover':
        candidates = [x for x in range(PopSize) if x != i]
        parent2_idx = np.random.choice(candidates)
        parent2 = Pop[parent2_idx]

        beta = np.random.rand(DimSize) * 2 - 1
        beta = np.where(np.random.rand(DimSize) < 0.5,
                        (2 * np.random.rand(DimSize)) ** (1 / 3),
                        (1 / (2 * (1 - np.random.rand(DimSize)))) ** (1 / 3))

        Off[i] = 0.5 * ((1 + beta) * Pop[i] + (1 - beta) * parent2)

    else:
        mutation_strength = 0.1 * (UB[0] - LB[0]) * (1 - curIter / MaxIter) ** 2
        mutation = np.random.normal(0, mutation_strength, DimSize)
        Off[i] = Pop[i] + mutation

    Off[i] = np.clip(Off[i], LB, UB)


def Opposition_Learning_Enhanced(i):
    """增强的广义反向学习"""
    global Pop, LB, UB, gbest
    lb_array = np.array(LB)
    ub_array = np.array(UB)

    pop_min = np.min(Pop, axis=0)
    pop_max = np.max(Pop, axis=0)
    pop_mean = np.mean(Pop, axis=0)

    strategy = np.random.choice(['classical', 'generalized'])
    if strategy == 'classical':
        opposite = lb_array + ub_array - Pop[i]
    else:
        k = np.random.rand()
        dynamic_lb = np.minimum(lb_array, gbest - k * (ub_array - lb_array))
        dynamic_ub = np.maximum(ub_array, gbest + k * (ub_array - lb_array))
        opposite = dynamic_lb + dynamic_ub - Pop[i]

    margin = 0.01 * (ub_array - lb_array)
    opposite = np.clip(opposite, lb_array + margin, ub_array - margin)
    Off[i] = opposite


def Local_Global_Mix(i):
    """局部-全局混合搜索算子"""
    global Pop, gbest, pbest, PopSize, LB, UB
    progress = curIter / MaxIter
    global_weight = 0.3 + 0.5 * progress
    local_weight = 0.5 - 0.3 * progress
    random_weight = 0.2

    global_component = gbest - Pop[i]
    neighbors = [j for j in range(PopSize) if j != i]
    neighbor_idx = np.random.choice(neighbors)
    local_component = Pop[neighbor_idx] - Pop[i]
    random_component = np.random.normal(0, 0.1 * (UB[0] - LB[0]), DimSize)

    search_direction = (global_weight * global_component +
                        local_weight * local_component +
                        random_weight * random_component)
    step_size = 0.5 * (1 - progress) ** 1.5 + 0.1

    Off[i] = Pop[i] + step_size * search_direction
    Off[i] = np.clip(Off[i], LB, UB)


def Self_Adaptive_Mutation(i):
    """自适应变异算子"""
    global Pop, curIter, MaxIter, LB, UB
    strategies = ['gaussian', 'cauchy', 'polynomial']
    weights = [0.5, 0.3, 0.2]

    chosen_strategy = np.random.choice(strategies, p=weights)
    base_scale = 0.1 * (UB[0] - LB[0])
    adaptive_scale = base_scale * (1 - curIter / MaxIter) ** 2

    if chosen_strategy == 'gaussian':
        mutation = np.random.normal(0, adaptive_scale * 0.3, DimSize)
    elif chosen_strategy == 'cauchy':
        mutation = np.random.standard_cauchy(DimSize) * adaptive_scale
    else:
        mutation = np.random.rand(DimSize) * 2 - 1
        mutation = np.sign(mutation) * np.abs(mutation) ** 1.5 * adaptive_scale

    if np.random.rand() < 0.7:
        base_solution = gbest
    else:
        base_solution = Pop[i]

    Off[i] = base_solution + mutation
    Off[i] = np.clip(Off[i], LB, UB)


def DE_current_to_pbest(i):
    """DE/current-to-pbest/1策略"""
    global Pop, pbest, PopSize, LB, UB
    candi = list(range(PopSize))
    candi.remove(i)

    pbest_size = max(1, PopSize // 5)
    pbest_indices = np.argsort(pbest_fit)[:pbest_size]
    pbest_idx = np.random.choice(pbest_indices)

    r1, r2 = np.random.choice(candi, 2, replace=False)
    F = 0.5
    Off[i] = Pop[i] + F * (pbest[pbest_idx] - Pop[i]) + F * (Pop[r1] - Pop[r2])
    Off[i] = np.clip(Off[i], LB, UB)


def CLPSO_learning(i):
    """综合学习PSO算子"""
    global Pop, pbest, v, gbest, LB, UB
    teacher = np.random.choice(PopSize)
    phi = 0.1 + 0.4 * np.random.rand(DimSize)
    Off[i] = Pop[i] + v[i] + phi * (pbest[teacher] - Pop[i])
    Off[i] = np.clip(Off[i], LB, UB)


# 算子列表
Operators = [
    Enhanced_CMA_ES_Operator,
    Adaptive_Levy_Flight,
    JADE_style_DE,
    Comprehensive_Learning,
    Hybrid_PSO_GA,
    Opposition_Learning_Enhanced,
    Local_Global_Mix,
    Self_Adaptive_Mutation,
    DE_current_to_pbest,
    CLPSO_learning
]
Sequence = np.zeros(PopSize, dtype=int)


# ========== 高层策略函数 ==========
def generate_adaptive_sequence():
    global PopSize, curIter, MaxIter
    progress = curIter / MaxIter
    max_op_index = 9
    Sequence = np.zeros(PopSize, dtype=int)

    if progress < 0.3:
        weights = [0.15, 0.05, 0.15, 0.1, 0.15, 0.05, 0.1, 0.05, 0.1, 0.1]
    elif progress < 0.7:
        weights = [0.1, 0.1, 0.1, 0.15, 0.1, 0.1, 0.15, 0.1, 0.05, 0.05]
    else:
        weights = [0.05, 0.15, 0.05, 0.1, 0.05, 0.15, 0.15, 0.15, 0.1, 0.05]

    weights = np.array(weights) / np.sum(weights)
    for i in range(PopSize):
        Sequence[i] = np.random.choice(range(max_op_index + 1), p=weights)
    return Sequence.tolist()


def enhance_diversity(sequence, recommended_ops, max_op_index):
    """增强序列多样性"""
    sequence_array = np.array(sequence)
    valid_recommended = [op for op in recommended_ops if 0 <= op <= max_op_index]
    if not valid_recommended:
        valid_recommended = [1, 3, 5, 7]

    unique_ops, counts = np.unique(sequence_array, return_counts=True)
    for op in valid_recommended[:6]:
        if op not in unique_ops or counts[list(unique_ops).index(op)] < len(sequence) // 15:
            replace_num = min(len(sequence) // 10, len(sequence) - len(unique_ops))
            if replace_num > 0:
                replace_indices = np.random.choice(len(sequence), size=replace_num, replace=False)
                sequence_array[replace_indices] = op

    return [max(0, min(int(x), max_op_index)) for x in sequence_array.tolist()]


def HighLevel(current_operators):
    """高层策略（修复kth越界问题）"""
    global Pop, FitPop, PopSize, Sequence, tmpFitPop, curIter, MaxIter, Trace
    max_op_index = len(current_operators) - 1

    # 修复：动态计算top百分比（避免kth越界）
    top_percent_size = max(1, min(PopSize // 10, len(FitPop) - 1))  # 至少1，最多len(FitPop)-1

    current_best = min(FitPop)
    fitness_std = np.std(FitPop)
    diversity_indicator = fitness_std / (np.max(FitPop) - current_best + 1e-8)

    convergence_rate = 0
    if len(Trace) > 30:
        convergence_rate = (Trace[-30] - current_best) / 30

    # 修复：使用安全的top百分比计算
    top_10_percent = np.partition(FitPop, top_percent_size)[:top_percent_size]
    bottom_10_percent = np.partition(FitPop, -top_percent_size)[-top_percent_size:]
    population_quality = np.mean(top_10_percent) / (np.mean(bottom_10_percent) + 1e-8)

    progress = curIter / MaxIter
    if progress < 0.2:
        phase = "AGGRESSIVE_EXPLORATION"
    elif progress < 0.5:
        phase = "BALANCED_SEARCH"
    elif progress < 0.8:
        phase = "FOCUSED_REFINEMENT"
    else:
        phase = "FINAL_POLISHING"

    # 策略选择
    if diversity_indicator < 0.02:
        strategy_analysis = "CRITICAL: Very low diversity - BOOST exploration operators 1,7,3\n"
        primary_ops = [1, 7, 3, 1, 7, 3]
    elif diversity_indicator < 0.05:
        strategy_analysis = "WARNING: Low diversity - Increase exploration 1,7\n"
        primary_ops = [1, 7, 2, 3]
    elif convergence_rate < 1e-13:
        strategy_analysis = "SLOW convergence - Enhance 5,0,8 for acceleration\n"
        primary_ops = [5, 0, 8, 2]
    elif population_quality > 0.8:
        strategy_analysis = "Good population quality - Balance all operators\n"
        primary_ops = [2, 6, 4, 0, 5, 1, 7, 3, 8, 9]
    else:
        strategy_analysis = "Normal optimization - Phase-based strategy\n"
        primary_ops = []

    # 阶段策略
    if phase == "AGGRESSIVE_EXPLORATION":
        base_ops = [1, 1, 1, 7, 7, 3, 2, 5, 6, 4]
    elif phase == "BALANCED_SEARCH":
        base_ops = [2, 6, 1, 5, 4, 7, 0, 3, 8, 9]
    elif phase == "FOCUSED_REFINEMENT":
        base_ops = [0, 8, 5, 2, 4, 6, 1, 7, 3, 9]
    else:
        base_ops = [0, 8, 4, 5, 2, 6, 7, 1, 3, 9]

    if primary_ops:
        recommended_operators = primary_ops + base_ops
    else:
        recommended_operators = base_ops

    # 生成LLM提示（如果不需要LLM，可直接用随机序列）
    try:
        prompt = f"""Generate {PopSize} integers (0-9) for optimization.
        Phase: {phase}
        Diversity: {diversity_indicator:.4f}
        Output ONLY {PopSize} numbers:"""

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": f"You are an optimization expert. Generate {PopSize} integers (0-9) for {phase} phase. Output ONLY numbers."},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            timeout=35
        )
        response_text = response.choices[0].message.content
        numbers = re.findall(r'\d+', response_text)
        raw_sequence = [int(num) for num in numbers[:PopSize]]
        valid_sequence = [max(0, min(num, max_op_index)) for num in raw_sequence]

        # 补充序列长度
        if len(valid_sequence) < PopSize:
            needed = PopSize - len(valid_sequence)
            supplement = []
            for _ in range(needed):
                if diversity_indicator < 0.03:
                    supplement.append(np.random.choice([1, 7, 3]))
                elif phase == "AGGRESSIVE_EXPLORATION":
                    supplement.append(np.random.choice([1, 7, 2, 3]))
                else:
                    supplement.append(np.random.choice([0, 8, 4, 5]))
            valid_sequence.extend(supplement)

        Sequence = valid_sequence[:PopSize]
        unique_ops = len(set(Sequence))
        if unique_ops < 7:
            Sequence = enhance_diversity(Sequence, recommended_operators, max_op_index)

    except Exception as e:
        print(f"LLM call failed: {e}, using fallback strategy")
        # 降级策略
        if phase == "AGGRESSIVE_EXPLORATION":
            weights = [0.02, 0.25, 0.15, 0.12, 0.08, 0.10, 0.10, 0.12, 0.04, 0.02]
        elif phase == "BALANCED_SEARCH":
            weights = [0.10, 0.15, 0.16, 0.08, 0.12, 0.12, 0.12, 0.08, 0.04, 0.03]
        elif phase == "FOCUSED_REFINEMENT":
            weights = [0.18, 0.10, 0.12, 0.06, 0.14, 0.12, 0.10, 0.08, 0.08, 0.02]
        else:
            weights = [0.22, 0.08, 0.10, 0.05, 0.15, 0.12, 0.08, 0.08, 0.10, 0.02]

        weights = np.array(weights) / np.sum(weights)
        Sequence = [np.random.choice(range(max_op_index + 1), p=weights) for _ in range(PopSize)]

    Sequence = [max(0, min(int(op), max_op_index)) for op in Sequence]
    return


def initialize_pso_parameters():
    """初始化PSO参数"""
    global PopSize, DimSize, Pop, FitPop, pbest, pbest_fit, gbest, gbest_fit, v, v_max, LB, UB
    v_max = 0.2 * (np.array(UB) - np.array(LB))
    v = np.random.uniform(-v_max, v_max, (PopSize, DimSize))
    pbest = deepcopy(Pop)
    pbest_fit = deepcopy(FitPop)

    best_idx = np.argmin(FitPop)
    gbest = deepcopy(Pop[best_idx])
    gbest_fit = FitPop[best_idx]


# ========== 算法定义 ==========
def LLMPSO(func):
    """LLM-PSO主算法"""
    global PopSize, DimSize, curIter, MaxIter, Pop, FitPop, Sequence, tmpFitPop, Off, FitOff
    global pbest, pbest_fit, gbest, gbest_fit, v, v_max, Trace

    # 初始化种群（同步维度）
    Pop, FitPop = unified_initialization(func)
    # 重新初始化Off/FitOff（适配当前维度）
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)

    initialize_pso_parameters()
    Sequence = np.random.randint(0, len(Operators), PopSize).tolist()

    Trace = []
    Trace.append(gbest_fit)
    api_call_count = 0

    for curIter in range(MaxIter):
        if curIter == 0:
            HighLevel(Operators)
        tmpFitPop = deepcopy(FitPop)

        for j in range(PopSize):
            # PSO速度更新
            w = 1.0
            c1, c2 = 2.05, 2.05
            r1, r2 = np.random.rand(DimSize), np.random.rand(DimSize)

            v[j] = w * v[j] + c1 * r1 * (pbest[j] - Pop[j]) + c2 * r2 * (gbest - Pop[j])
            v[j] = np.clip(v[j], -2, 2)

            # 位置更新
            Pop[j] = Pop[j] + v[j]
            Pop[j] = np.clip(Pop[j], LB, UB)

            # 评估（安全提取标量）
            current_fit = safe_evaluate(func, Pop[j])

            # 更新个体最优
            if current_fit < pbest_fit[j]:
                pbest[j] = deepcopy(Pop[j])
                pbest_fit[j] = current_fit

            # 更新全局最优
            if current_fit < gbest_fit:
                gbest = deepcopy(Pop[j])
                gbest_fit = current_fit

            # 应用算子
            Operator = Operators[int(Sequence[j])]
            Operator(j)
            FitOff[j] = safe_evaluate(func, Off[j])

            # 选择更新
            if FitOff[j] < FitPop[j]:
                FitPop[j] = FitOff[j]
                Pop[j] = deepcopy(Off[j])

                if FitOff[j] < pbest_fit[j]:
                    pbest[j] = deepcopy(Off[j])
                    pbest_fit[j] = FitOff[j]

                if FitOff[j] < gbest_fit:
                    gbest = deepcopy(Off[j])
                    gbest_fit = FitOff[j]

        Trace.append(gbest_fit)
        api_call_count += 1
        if api_call_count >= 5:
            HighLevel(Operators)
            api_call_count = 0

    return Trace


def PSO(func):
    """基础PSO"""
    global PopSize, DimSize, curIter, MaxIter, Pop, FitPop, Sequence, tmpFitPop, Off, FitOff
    global pbest, pbest_fit, gbest, gbest_fit, v, v_max, Trace

    Pop, FitPop = unified_initialization(func)
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)

    initialize_pso_parameters()
    Sequence = np.random.randint(0, len(Operators), PopSize).tolist()

    Trace = []
    Trace.append(gbest_fit)

    for curIter in range(MaxIter):
        tmpFitPop = deepcopy(FitPop)

        for j in range(PopSize):
            w = 1.0
            c1, c2 = 2.05, 2.05
            r1, r2 = np.random.rand(DimSize), np.random.rand(DimSize)

            v[j] = w * v[j] + c1 * r1 * (pbest[j] - Pop[j]) + c2 * r2 * (gbest - Pop[j])
            v[j] = np.clip(v[j], -2, 2)

            Pop[j] = Pop[j] + v[j]
            Pop[j] = np.clip(Pop[j], LB, UB)

            current_fit = safe_evaluate(func, Pop[j])

            if current_fit < pbest_fit[j]:
                pbest[j] = deepcopy(Pop[j])
                pbest_fit[j] = current_fit

            if current_fit < gbest_fit:
                gbest = deepcopy(Pop[j])
                gbest_fit = current_fit

        Trace.append(gbest_fit)

    return Trace


def PPSO(func):
    """并行PSO"""
    global PopSize, DimSize, MaxIter, LB, UB, Pop, FitPop, Trace

    Pop, FitPop = unified_initialization(func)
    num_groups = 4
    group_size = PopSize // num_groups
    v_max = 0.2 * (np.array(UB) - np.array(LB))

    groups_v = [np.random.uniform(-v_max, v_max, (group_size, DimSize)) for _ in range(num_groups)]
    groups_pbest = [deepcopy(Pop[i * group_size:(i + 1) * group_size]) for i in range(num_groups)]
    groups_pbest_fit = [deepcopy(FitPop[i * group_size:(i + 1) * group_size]) for i in range(num_groups)]
    groups_gbest = [deepcopy(groups_pbest[i][np.argmin(groups_pbest_fit[i])]) for i in range(num_groups)]
    groups_gbest_fit = [np.min(groups_pbest_fit[i]) for i in range(num_groups)]

    global_best_idx = np.argmin(groups_gbest_fit)
    global_best = deepcopy(groups_gbest[global_best_idx])
    global_best_fit = groups_gbest_fit[global_best_idx]

    Trace = []
    Trace.append(global_best_fit)

    for iter_count in range(MaxIter):
        for group_idx in range(num_groups):
            start_idx = group_idx * group_size
            end_idx = (group_idx + 1) * group_size
            w = 0.9 - (0.9 - 0.4) * (iter_count / MaxIter)
            c1, c2 = 2.0, 2.0

            for j in range(group_size):
                idx = start_idx + j
                r1, r2 = np.random.rand(DimSize), np.random.rand(DimSize)

                groups_v[group_idx][j] = (w * groups_v[group_idx][j] +
                                          c1 * r1 * (groups_pbest[group_idx][j] - Pop[idx]) +
                                          c2 * r2 * (groups_gbest[group_idx] - Pop[idx]))
                groups_v[group_idx][j] = np.clip(groups_v[group_idx][j], -v_max, v_max)

                Pop[idx] = Pop[idx] + groups_v[group_idx][j]
                Pop[idx] = np.clip(Pop[idx], LB, UB)

                current_fit = safe_evaluate(func, Pop[idx])
                FitPop[idx] = current_fit

                if current_fit < groups_pbest_fit[group_idx][j]:
                    groups_pbest[group_idx][j] = deepcopy(Pop[idx])
                    groups_pbest_fit[group_idx][j] = current_fit

                if current_fit < groups_gbest_fit[group_idx]:
                    groups_gbest[group_idx] = deepcopy(Pop[idx])
                    groups_gbest_fit[group_idx] = current_fit

                if current_fit < global_best_fit:
                    global_best = deepcopy(Pop[idx])
                    global_best_fit = current_fit

        if iter_count % 10 == 0:
            group1, group2 = np.random.choice(num_groups, 2, replace=False)
            swap_idx = np.random.randint(group_size)
            groups_pbest[group1][swap_idx] = deepcopy(groups_gbest[group2])
            groups_pbest_fit[group1][swap_idx] = groups_gbest_fit[group2]

        Trace.append(global_best_fit)

    return Trace


def CLPSO(func):
    """综合学习PSO"""
    global PopSize, DimSize, MaxIter, LB, UB, Pop, FitPop, Trace

    Pop, FitPop = unified_initialization(func)
    c_local = 1.2
    w_max, w_min = 0.9, 0.4
    v_max = 0.2 * (np.array(UB) - np.array(LB))
    v = np.random.uniform(-v_max, v_max, (PopSize, DimSize))

    pbest = deepcopy(Pop)
    pbest_fit = deepcopy(FitPop)
    best_idx = np.argmin(FitPop)
    gbest = deepcopy(Pop[best_idx])
    gbest_fit = FitPop[best_idx]

    learning_exemplars = np.zeros((PopSize, DimSize), dtype=int)
    refresh_count = 0
    refresh_gap = 7

    Trace = []
    Trace.append(gbest_fit)

    def update_learning_exemplars():
        nonlocal learning_exemplars
        for i in range(PopSize):
            for d in range(DimSize):
                if np.random.rand() < 0.3:
                    candidates = [j for j in range(PopSize) if j != i]
                    r1, r2 = np.random.choice(candidates, 2, replace=False)
                    if pbest_fit[r1] < pbest_fit[r2]:
                        learning_exemplars[i, d] = r1
                    else:
                        learning_exemplars[i, d] = r2
                else:
                    learning_exemplars[i, d] = i

    update_learning_exemplars()

    for iter_count in range(MaxIter):
        refresh_count += 1
        if refresh_count >= refresh_gap:
            update_learning_exemplars()
            refresh_count = 0

        w_current = w_max - (w_max - w_min) * (iter_count / MaxIter)

        for i in range(PopSize):
            learning_target = np.zeros(DimSize)
            for d in range(DimSize):
                teacher_idx = learning_exemplars[i, d]
                learning_target[d] = pbest[teacher_idx, d]

            r = np.random.rand(DimSize)
            v[i] = (w_current * v[i] +
                    c_local * r * (learning_target - Pop[i]))
            v[i] = np.clip(v[i], -v_max, v_max)

            Pop[i] = Pop[i] + v[i]
            Pop[i] = np.clip(Pop[i], LB, UB)

            current_fit = safe_evaluate(func, Pop[i])

            if current_fit < pbest_fit[i]:
                pbest[i] = deepcopy(Pop[i])
                pbest_fit[i] = current_fit

                if current_fit < gbest_fit:
                    gbest = deepcopy(Pop[i])
                    gbest_fit = current_fit

        Trace.append(gbest_fit)

    return Trace


def DMS_PSO(func):
    """动态多群PSO"""
    global PopSize, DimSize, MaxIter, LB, UB, Pop, FitPop, Trace

    Pop, FitPop = unified_initialization(func)
    num_subswarms = 5
    subswarm_size = PopSize // num_subswarms
    regrouping_period = 10
    v_max = 0.2 * (np.array(UB) - np.array(LB))

    subswarms = []
    subswarms_pbest = []
    subswarms_pbest_fit = []
    subswarms_gbest = []
    subswarms_gbest_fit = []
    subswarms_v = []

    indices = np.random.permutation(PopSize)
    for i in range(num_subswarms):
        start_idx = i * subswarm_size
        end_idx = start_idx + subswarm_size if i < num_subswarms - 1 else PopSize
        subswarm_indices = indices[start_idx:end_idx]
        subswarm = Pop[subswarm_indices]
        subswarm_fit = FitPop[subswarm_indices]

        subswarm_v = np.random.uniform(-v_max, v_max, (len(subswarm_indices), DimSize))
        subswarm_pbest = deepcopy(subswarm)
        subswarm_pbest_fit = deepcopy(subswarm_fit)

        best_idx = np.argmin(subswarm_fit)
        subswarm_gbest = deepcopy(subswarm[best_idx])
        subswarm_gbest_fit = subswarm_fit[best_idx]

        subswarms.append(subswarm)
        subswarms_pbest.append(subswarm_pbest)
        subswarms_pbest_fit.append(subswarm_pbest_fit)
        subswarms_gbest.append(subswarm_gbest)
        subswarms_gbest_fit.append(subswarm_gbest_fit)
        subswarms_v.append(subswarm_v)

    global_best_idx = np.argmin(subswarms_gbest_fit)
    global_best = deepcopy(subswarms_gbest[global_best_idx])
    global_best_fit = subswarms_gbest_fit[global_best_idx]

    Trace = []
    Trace.append(global_best_fit)

    for iter_count in range(MaxIter):
        for swarm_idx in range(num_subswarms):
            current_swarm = subswarms[swarm_idx]
            current_pbest = subswarms_pbest[swarm_idx]
            current_pbest_fit = subswarms_pbest_fit[swarm_idx]
            current_gbest = subswarms_gbest[swarm_idx]
            current_v = subswarms_v[swarm_idx]
            swarm_size = len(current_swarm)

            w = 0.9 - (0.9 - 0.4) * (iter_count / MaxIter)
            c1, c2 = 2.0, 2.0

            for j in range(swarm_size):
                r1, r2 = np.random.rand(DimSize), np.random.rand(DimSize)

                current_v[j] = (w * current_v[j] +
                                c1 * r1 * (current_pbest[j] - current_swarm[j]) +
                                c2 * r2 * (current_gbest - current_swarm[j]))
                current_v[j] = np.clip(current_v[j], -v_max, v_max)

                current_swarm[j] = current_swarm[j] + current_v[j]
                current_swarm[j] = np.clip(current_swarm[j], LB, UB)

                current_fit = safe_evaluate(func, current_swarm[j])

                if current_fit < current_pbest_fit[j]:
                    current_pbest[j] = deepcopy(current_swarm[j])
                    current_pbest_fit[j] = current_fit

                if current_fit < subswarms_gbest_fit[swarm_idx]:
                    subswarms_gbest[swarm_idx] = deepcopy(current_swarm[j])
                    subswarms_gbest_fit[swarm_idx] = current_fit

                if current_fit < global_best_fit:
                    global_best = deepcopy(current_swarm[j])
                    global_best_fit = current_fit

            subswarms[swarm_idx] = current_swarm
            subswarms_pbest[swarm_idx] = current_pbest
            subswarms_pbest_fit[swarm_idx] = current_pbest_fit
            subswarms_v[swarm_idx] = current_v

        if iter_count % regrouping_period == 0:
            all_particles = np.vstack(subswarms)
            all_fitness = np.hstack([subswarm_fit for subswarm_fit in subswarms_pbest_fit])
            sorted_indices = np.argsort(all_fitness)
            all_particles_sorted = all_particles[sorted_indices]
            all_fitness_sorted = all_fitness[sorted_indices]

            new_subswarms = []
            new_subswarms_pbest = []
            new_subswarms_pbest_fit = []
            new_subswarms_gbest = []
            new_subswarms_gbest_fit = []
            new_subswarms_v = []

            for i in range(num_subswarms):
                start_idx = i * subswarm_size
                end_idx = start_idx + subswarm_size if i < num_subswarms - 1 else PopSize
                new_subswarm = all_particles_sorted[start_idx:end_idx]
                new_subswarm_fit = all_fitness_sorted[start_idx:end_idx]

                new_subswarm_v = np.random.uniform(-v_max, v_max, (len(new_subswarm), DimSize))
                new_subswarm_pbest = deepcopy(new_subswarm)
                new_subswarm_pbest_fit = deepcopy(new_subswarm_fit)

                best_idx = np.argmin(new_subswarm_fit)
                new_subswarm_gbest = deepcopy(new_subswarm[best_idx])
                new_subswarm_gbest_fit = new_subswarm_fit[best_idx]

                new_subswarms.append(new_subswarm)
                new_subswarms_pbest.append(new_subswarm_pbest)
                new_subswarms_pbest_fit.append(new_subswarm_pbest_fit)
                new_subswarms_gbest.append(new_subswarm_gbest)
                new_subswarms_gbest_fit.append(new_subswarm_gbest_fit)
                new_subswarms_v.append(new_subswarm_v)

            subswarms = new_subswarms
            subswarms_pbest = new_subswarms_pbest
            subswarms_pbest_fit = new_subswarms_pbest_fit
            subswarms_gbest = new_subswarms_gbest
            subswarms_gbest_fit = new_subswarms_gbest_fit
            subswarms_v = new_subswarms_v

            global_best_idx = np.argmin(subswarms_gbest_fit)
            global_best = deepcopy(subswarms_gbest[global_best_idx])
            global_best_fit = subswarms_gbest_fit[global_best_idx]

            if iter_count % 50 == 0:
                print(f"DMS-PSO: Iteration {iter_count}, Regrouping completed")

        if iter_count % 7 == 0:
            swarm1, swarm2 = np.random.choice(num_subswarms, 2, replace=False)
            idx1 = np.random.randint(len(subswarms[swarm1]))
            idx2 = np.random.randint(len(subswarms[swarm2]))

            subswarms[swarm1][idx1], subswarms[swarm2][idx2] = deepcopy(subswarms[swarm2][idx2]), deepcopy(
                subswarms[swarm1][idx1])
            subswarms_pbest[swarm1][idx1], subswarms_pbest[swarm2][idx2] = deepcopy(
                subswarms_pbest[swarm2][idx2]), deepcopy(subswarms_pbest[swarm1][idx1])
            subswarms_pbest_fit[swarm1][idx1], subswarms_pbest_fit[swarm2][idx2] = subswarms_pbest_fit[swarm2][idx2], \
            subswarms_pbest_fit[swarm1][idx1]

        Trace.append(global_best_fit)

    return Trace


def PSO_GA(func):
    """PSO-GA混合算法"""
    global PopSize, DimSize, curIter, MaxIter, Pop, FitPop, Sequence, tmpFitPop, Off, FitOff
    global pbest, pbest_fit, gbest, gbest_fit, v, v_max, Trace

    Pop, FitPop = unified_initialization(func)
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)

    initialize_pso_parameters()
    crossover_rate = 0.8
    mutation_rate = 0.1
    tournament_size = 3

    Trace = []
    Trace.append(gbest_fit)

    for iter_count in range(MaxIter):
        w = 0.9 - (0.9 - 0.4) * (iter_count / MaxIter)
        c1, c2 = 2.0, 2.0

        for j in range(PopSize):
            r1, r2 = np.random.rand(DimSize), np.random.rand(DimSize)

            v[j] = w * v[j] + c1 * r1 * (pbest[j] - Pop[j]) + c2 * r2 * (gbest - Pop[j])
            v[j] = np.clip(v[j], -v_max, v_max)

            Pop[j] = Pop[j] + v[j]
            Pop[j] = np.clip(Pop[j], LB, UB)

            current_fit = safe_evaluate(func, Pop[j])

            if current_fit < pbest_fit[j]:
                pbest[j] = deepcopy(Pop[j])
                pbest_fit[j] = current_fit

            if current_fit < gbest_fit:
                gbest = deepcopy(Pop[j])
                gbest_fit = current_fit

        if iter_count % 5 == 0:
            new_pop = np.zeros_like(Pop)
            new_fit = np.zeros(PopSize)

            for i in range(PopSize):
                candidates = np.random.choice(PopSize, tournament_size, replace=False)
                best_candidate = candidates[np.argmin(FitPop[candidates])]
                new_pop[i] = deepcopy(Pop[best_candidate])
                new_fit[i] = FitPop[best_candidate]

            for i in range(0, PopSize - 1, 2):
                if np.random.rand() < crossover_rate:
                    parent1 = new_pop[i]
                    parent2 = new_pop[i + 1]

                    beta = np.random.rand(DimSize)
                    child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
                    child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)

                    child1 = np.clip(child1, LB, UB)
                    child2 = np.clip(child2, LB, UB)

                    new_pop[i] = child1
                    new_pop[i + 1] = child2

            for i in range(PopSize):
                if np.random.rand() < mutation_rate:
                    mutation_strength = 0.1 * (UB[0] - LB[0]) * (1 - iter_count / MaxIter)
                    mutation = np.random.normal(0, mutation_strength, DimSize)
                    new_pop[i] = new_pop[i] + mutation
                    new_pop[i] = np.clip(new_pop[i], LB, UB)

            for i in range(PopSize):
                new_fit[i] = safe_evaluate(func, new_pop[i])

            best_ga_idx = np.argmin(new_fit)
            if new_fit[best_ga_idx] < gbest_fit:
                gbest = deepcopy(new_pop[best_ga_idx])
                gbest_fit = new_fit[best_ga_idx]

            Pop = deepcopy(new_pop)
            FitPop = deepcopy(new_fit)

            for j in range(PopSize):
                if FitPop[j] < pbest_fit[j]:
                    pbest[j] = deepcopy(Pop[j])
                    pbest_fit[j] = FitPop[j]

        Trace.append(gbest_fit)

    return Trace


def SADE(func):
    """自适应DE"""
    global PopSize, DimSize, MaxIter, LB, UB, Pop, FitPop, Trace

    Pop, FitPop = unified_initialization(func)
    μ_F, σ_F = 0.5, 0.3
    μ_Cr, σ_Cr = 0.5, 0.1

    F = np.random.normal(μ_F, σ_F, PopSize)
    Cr = np.random.normal(μ_Cr, σ_Cr, PopSize)
    F = np.clip(F, 0.1, 1.0)
    Cr = np.clip(Cr, 0.0, 1.0)

    best_idx = np.argmin(FitPop)
    best_fit = FitPop[best_idx]
    best_solution = deepcopy(Pop[best_idx])

    Trace = []
    Trace.append(best_fit)
    success_F = []
    success_Cr = []

    for iter_count in range(MaxIter):
        for i in range(PopSize):
            if np.random.rand() < 0.5:
                candidates = [x for x in range(PopSize) if x != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                mutant = Pop[r1] + F[i] * (Pop[r2] - Pop[r3])
            else:
                candidates = [x for x in range(PopSize) if x != i]
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                mutant = Pop[i] + F[i] * (best_solution - Pop[i]) + F[i] * (Pop[r1] - Pop[r2])

            mutant = np.clip(mutant, LB, UB)
            trial = np.copy(Pop[i])
            j_rand = np.random.randint(DimSize)

            for j in range(DimSize):
                if np.random.rand() < Cr[i] or j == j_rand:
                    trial[j] = mutant[j]

            trial_fit = safe_evaluate(func, trial)
            if trial_fit < FitPop[i]:
                success_F.append(F[i])
                success_Cr.append(Cr[i])

                Pop[i] = trial
                FitPop[i] = trial_fit

                if trial_fit < best_fit:
                    best_fit = trial_fit
                    best_solution = deepcopy(trial)

        if len(success_F) > 0:
            μ_F = np.mean(success_F)
            σ_F = np.std(success_F) if np.std(success_F) >= 0.1 else 0.1
            μ_Cr = np.mean(success_Cr)
            σ_Cr = np.std(success_Cr) if np.std(success_Cr) >= 0.05 else 0.05

            F = np.random.normal(μ_F, σ_F, PopSize)
            Cr = np.random.normal(μ_Cr, σ_Cr, PopSize)
            F = np.clip(F, 0.1, 1.0)
            Cr = np.clip(Cr, 0.0, 1.0)

        Trace.append(best_fit)

    return Trace


def WDE(func):
    """加权DE"""
    global PopSize, DimSize, MaxIter, LB, UB, Pop, FitPop, Trace

    Pop, FitPop = unified_initialization(func)
    F_mean = 0.5
    Cr_mean = 0.5

    best_idx = np.argmin(FitPop)
    best_fit = FitPop[best_idx]
    best_solution = deepcopy(Pop[best_idx])

    Trace = []
    Trace.append(best_fit)
    success_F = []
    success_Cr = []

    for iter_count in range(MaxIter):
        if len(success_F) > 0:
            F_mean = np.mean(success_F[-10:]) if len(success_F) >= 10 else np.mean(success_F)
            Cr_mean = np.mean(success_Cr[-10:]) if len(success_Cr) >= 10 else np.mean(success_Cr)

        F = np.random.normal(F_mean, 0.1, PopSize)
        Cr = np.random.normal(Cr_mean, 0.1, PopSize)
        F = np.clip(F, 0.1, 1.0)
        Cr = np.clip(Cr, 0.0, 1.0)

        fit_min = np.min(FitPop)
        fit_max = np.max(FitPop)
        if fit_max - fit_min < 1e-8:
            weights = np.ones(PopSize) / PopSize
        else:
            norm_fit = (fit_max - FitPop) / (fit_max - fit_min)
            weights = norm_fit / np.sum(norm_fit)

        for i in range(PopSize):
            candidates = np.delete(np.arange(PopSize), i)
            r1 = np.random.choice(candidates, p=weights[candidates] / np.sum(weights[candidates]))
            candidates_r2 = np.delete(candidates, np.where(candidates == r1))
            r2 = np.random.choice(candidates_r2, p=weights[candidates_r2] / np.sum(weights[candidates_r2]))
            candidates_r3 = np.delete(candidates_r2, np.where(candidates_r2 == r2))
            r3 = np.random.choice(candidates_r3, p=weights[candidates_r3] / np.sum(weights[candidates_r3]))

            mutant = Pop[r1] + F[i] * (Pop[r2] - Pop[r3])
            mutant = np.clip(mutant, LB, UB)

            trial = np.copy(Pop[i])
            j_rand = np.random.randint(DimSize)
            for j in range(DimSize):
                if np.random.rand() < Cr[i] or j == j_rand:
                    trial[j] = mutant[j]

            trial_fit = safe_evaluate(func, trial)
            if trial_fit < FitPop[i]:
                success_F.append(F[i])
                success_Cr.append(Cr[i])

                Pop[i] = trial
                FitPop[i] = trial_fit
                if trial_fit < best_fit:
                    best_fit = trial_fit
                    best_solution = deepcopy(trial)

        Trace.append(best_fit)

    return Trace


def DE(func):
    """标准DE"""
    global PopSize, DimSize, MaxIter, LB, UB, Pop, FitPop, Trace

    Pop, FitPop = unified_initialization(func)
    F = 0.8
    Cr = 0.9

    best_idx = np.argmin(FitPop)
    best_fit = FitPop[best_idx]
    best_solution = deepcopy(Pop[best_idx])

    Trace = []
    Trace.append(best_fit)

    for iter_count in range(MaxIter):
        for i in range(PopSize):
            candidates = [x for x in range(PopSize) if x != i]
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

            mutant = Pop[i] + F * (Pop[r1] - Pop[i]) + F * (Pop[r2] - Pop[r3])
            mutant = np.clip(mutant, LB, UB)

            trial = np.copy(Pop[i])
            j_rand = np.random.randint(DimSize)

            for j in range(DimSize):
                if np.random.rand() < Cr or j == j_rand:
                    trial[j] = mutant[j]

            trial_fit = safe_evaluate(func, trial)
            if trial_fit < FitPop[i]:
                Pop[i] = trial
                FitPop[i] = trial_fit

                if trial_fit < best_fit:
                    best_fit = trial_fit
                    best_solution = deepcopy(trial)

        Trace.append(best_fit)

    return Trace

def main():
    global DimSize, LB, UB, MaxFEs, MaxIter, Trials, PopSize, Pop, Off

    Probs = [WBP(), PVP(), CSP(), SRD(), TBTD(), GTD(), CBD(), IBD(), TCD(), PLD(), CBHD(), RCB()]
    Names = ["WBP", "PVP", "CSP", "SRD", "TBTD", "GTD", "CBD", "IBD", "TCD", "PLD", "CBHD", "RCB"]

    for i in range(len(Probs)):
        llm_pso_all_trials = []
        pso_all_trials = []
        ppso_all_trials = []
        sade_all_trials = []
        wde_all_trials = []
        de_all_trials = []
        dms_pso_all_trials = []
        clpso_all_trials = []
        pso_ga_all_trials = []

        for j in range(Trials):
            print(f"\n--- Trial {j + 1} - Problem {Names[i]} ---")

            np.random.seed(2025 + 7 * j)
            llm_pso_trace = LLMPSO(Probs[i])
            llm_pso_all_trials.append(llm_pso_trace)

            np.random.seed(2025 + 7 * j)
            pso_trace = PSO(Probs[i])
            pso_all_trials.append(pso_trace)

            np.random.seed(2025 + 7 * j)
            ppso_trace = PPSO(Probs[i])
            ppso_all_trials.append(ppso_trace)

            np.random.seed(2025 + 7 * j)
            sade_trace = SADE(Probs[i])
            sade_all_trials.append(sade_trace)

            np.random.seed(2025 + 7 * j)
            wde_trace = WDE(Probs[i])
            wde_all_trials.append(wde_trace)

            np.random.seed(2025 + 7 * j)
            de_trace = DE(Probs[i])
            de_all_trials.append(de_trace)

            np.random.seed(2025 + 7 * j)
            dms_pso_trace = DMS_PSO(Probs[i])
            dms_pso_all_trials.append(dms_pso_trace)

            np.random.seed(2025 + 7 * j)
            clpso_trace = CLPSO(Probs[i])
            clpso_all_trials.append(clpso_trace)

            np.random.seed(2025 + 7 * j)
            pso_ga_trace = PSO_GA(Probs[i])
            pso_ga_all_trials.append(pso_ga_trace)

        def calculate_average_trace(trials_traces):
            """计算平均轨迹（修复形状不均）"""
            max_len = max(len(trace) for trace in trials_traces)
            padded_traces = []
            for trace in trials_traces:
                if len(trace) < max_len:
                    padded = trace + [trace[-1]] * (max_len - len(trace))
                else:
                    padded = trace[:max_len]
                padded_traces.append(padded)
            padded_array = np.array(padded_traces, dtype=np.float64)
            avg_trace = np.mean(padded_array, axis=0)
            return avg_trace

        # 计算平均轨迹
        llm_pso_avg = calculate_average_trace(llm_pso_all_trials)
        pso_avg = calculate_average_trace(pso_all_trials)
        ppso_avg = calculate_average_trace(ppso_all_trials)
        sade_avg = calculate_average_trace(sade_all_trials)
        wde_avg = calculate_average_trace(wde_all_trials)
        de_avg = calculate_average_trace(de_all_trials)
        dms_pso_avg = calculate_average_trace(dms_pso_all_trials)
        clpso_avg = calculate_average_trace(clpso_all_trials)
        pso_ga_avg = calculate_average_trace(pso_ga_all_trials)

        # 保存结果
        average_results = np.vstack([
            llm_pso_avg, pso_avg, ppso_avg, sade_avg, wde_avg,
            de_avg, dms_pso_avg, clpso_avg, pso_ga_avg
        ])

        save_path = f"./LLMPSO_Data/LLMPSO_engineer_data/{Names[i]}_results.csv"
        np.savetxt(save_path, average_results, delimiter=",", fmt="%.6f")

        print(f"已保存 {Names[i]} 结果，形状: {average_results.shape}")


if __name__ == "__main__":
    if not os.path.exists('LLMPSO_Data/LLMPSO_engineer_data'):
        os.makedirs('LLMPSO_Data/LLMPSO_engineer_data')
    main()