import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns
import os
import re

# ================== 用户可修改参数 ==================
selected_n = 4  # ←←← 在这里修改你想要的 total_turns = n 值
show_individual = False  # 是否显示每个 case 的半透明线（True/False）
band_mode = "std"  # "std" 或 "sem"（仅用于双曲线图的色带计算）


# ====================================================

# ================== 提取 model 名称 ==================
def extract_model_name(filename):
    match = re.search(r'test_detail_(.+?)_with_refusal\.jsonl', filename)
    return match.group(1) if match else "unknown_model"


input_file = 'test_detail_llama3.1_with_refusal.jsonl'
model_name = extract_model_name(input_file)


# ================== 论文级可视化函数（双曲线 + 色带，图例仅均值曲线） ==================
def visualize_dual_scores(sim_matrix, ref_matrix, save_name="dual_scores",
                          title="Scores by Turn",
                          show_individual=False, alpha_band=0.25, alpha_lines=0.35,
                          band_mode="std"):
    """绘制 Similarity（蓝色）和 Refusal（黄色）的均值折线 + 范围色带，仅显示均值图例，置于右下角"""
    if sim_matrix is None or ref_matrix is None or sim_matrix.size == 0:
        return

    N, num_turns = sim_matrix.shape
    x = np.arange(1, num_turns + 1)

    # 使用 nanmean / nanstd，自动跳过 NaN（缺失值）
    mean_sim = np.nanmean(sim_matrix, axis=0)
    mean_ref = np.nanmean(ref_matrix, axis=0)

    if band_mode == "std":
        spread_sim = np.nanstd(sim_matrix, axis=0, ddof=1) if N > 1 else np.zeros(num_turns)
        spread_ref = np.nanstd(ref_matrix, axis=0, ddof=1) if N > 1 else np.zeros(num_turns)
        lower_sim, upper_sim = mean_sim - spread_sim, mean_sim + spread_sim
        lower_ref, upper_ref = mean_ref - spread_ref, mean_ref + spread_ref
    elif band_mode == "sem":
        std_sim = np.nanstd(sim_matrix, axis=0, ddof=1) if N > 1 else np.zeros(num_turns)
        std_ref = np.nanstd(ref_matrix, axis=0, ddof=1) if N > 1 else np.zeros(num_turns)
        sem_sim = std_sim / np.sqrt(np.sum(~np.isnan(sim_matrix), axis=0).clip(min=1))
        sem_ref = std_ref / np.sqrt(np.sum(~np.isnan(ref_matrix), axis=0).clip(min=1))
        lower_sim, upper_sim = mean_sim - sem_sim, mean_sim + sem_sim
        lower_ref, upper_ref = mean_ref - sem_ref, mean_ref + sem_ref
    else:
        raise ValueError("band_mode 必须是 'std' 或 'sem'")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    if show_individual:
        for i in range(N):
            ax.plot(x, sim_matrix[i], color="#1f77b4", alpha=alpha_lines, linewidth=1)
            ax.plot(x, ref_matrix[i], color="#ff7f0e", alpha=alpha_lines, linewidth=1)

    # Similarity（蓝色均值线 + 色带）
    ax.plot(x, mean_sim, color="#1f77b4", linewidth=2.8, marker="o", markersize=7,
            label="Similarity")
    ax.fill_between(x, lower_sim, upper_sim, color="#1f77b4", alpha=alpha_band)

    # Refusal（黄色均值线 + 色带）
    ax.plot(x, mean_ref, color="#ff7f0e", linewidth=2.8, marker="s", markersize=7,
            label="Refusal")
    ax.fill_between(x, lower_ref, upper_ref, color="#ff7f0e", alpha=alpha_band)

    ax.set_xlabel("Turn Number", fontsize=13, fontweight='bold')
    ax.set_ylabel("Average Score", fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    ax.set_xticks(x)
    ax.grid(True, linestyle="--", alpha=0.3)

    # 图例置于右下角，仅显示两条均值曲线
    ax.legend(fontsize=12, loc="lower right")

    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    os.makedirs('pic', exist_ok=True)
    base_path = f"pic/{save_name}"

    plt.tight_layout()
    plt.savefig(f"{base_path}.pdf", dpi=600, bbox_inches='tight')
    #plt.savefig(f"{base_path}.png", dpi=600, bbox_inches='tight')

    print(f"[*] Plot saved to {base_path}.pdf 和 {base_path}.png")
    plt.close(fig)


# ================== Success vs Failed Refusal 对比图（所有 cases，无 n 限制，仅曲线） ==================
def visualize_refusal_comparison_all(success_cases, failed_cases, save_name="refusal_comparison_all"):
    """计算所有 success 和 failed cases 的 refusal score 均值曲线（缺失值自动跳过）"""
    if not success_cases or not failed_cases:
        print("无法生成对比图：缺少 success 或 failed 数据")
        return

    success_turn_refs = defaultdict(list)
    failed_turn_refs = defaultdict(list)

    for case in success_cases:
        trajectory = case['trajectory']
        for i, turn in enumerate(trajectory):
            turn_idx = i + 1
            if 'refusal_score' in turn and turn['refusal_score'] is not None:
                success_turn_refs[turn_idx].append(turn['refusal_score'])

    for case in failed_cases:
        trajectory = case['trajectory']
        for i, turn in enumerate(trajectory):
            turn_idx = i + 1
            if 'refusal_score' in turn and turn['refusal_score'] is not None:
                failed_turn_refs[turn_idx].append(turn['refusal_score'])

    all_turns = sorted(set(success_turn_refs.keys()) | set(failed_turn_refs.keys()))
    if not all_turns:
        return

    max_turn = max(all_turns)
    success_matrix = np.full((len(success_cases), max_turn), np.nan)
    failed_matrix = np.full((len(failed_cases), max_turn), np.nan)

    for idx, case in enumerate(success_cases):
        trajectory = case['trajectory']
        for i, turn in enumerate(trajectory):
            turn_idx = i + 1
            if turn_idx <= max_turn and 'refusal_score' in turn and turn['refusal_score'] is not None:
                success_matrix[idx, turn_idx - 1] = turn['refusal_score']

    for idx, case in enumerate(failed_cases):
        trajectory = case['trajectory']
        for i, turn in enumerate(trajectory):
            turn_idx = i + 1
            if turn_idx <= max_turn and 'refusal_score' in turn and turn['refusal_score'] is not None:
                failed_matrix[idx, turn_idx - 1] = turn['refusal_score']

    # 计算均值（自动跳过 NaN，无需任何填充）
    mean_success = np.nanmean(success_matrix, axis=0)
    mean_failed = np.nanmean(failed_matrix, axis=0)
    x = np.arange(1, max_turn + 1)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 只在有有效数据的位置绘制点
    valid_success = ~np.isnan(mean_success)
    valid_failed = ~np.isnan(mean_failed)

    ax.plot(x[valid_success], mean_success[valid_success], color="#1f77b4", linewidth=2.8, marker="o", markersize=7,
            label="Success Cases - Refusal")
    ax.plot(x[valid_failed], mean_failed[valid_failed], color="#ff7f0e", linewidth=2.8, marker="s", markersize=7,
            linestyle="--",
            label="Failed Cases - Refusal")

    ax.set_xlabel("Turn Number", fontsize=13, fontweight='bold')
    ax.set_ylabel("Average Refusal Score", fontsize=13, fontweight='bold')
    ax.set_title(f'Refusal Score Comparison (All Cases, {model_name})\n'
                 f'Success vs Failed (no total_turns filter)',
                 fontsize=15, fontweight='bold', pad=20)

    ax.set_xticks(x)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=12, loc="lower right")

    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    os.makedirs('pic', exist_ok=True)
    base_path = f"pic/{save_name}"

    plt.tight_layout()
    plt.savefig(f"{base_path}.pdf", dpi=600, bbox_inches='tight')
    #plt.savefig(f"{base_path}.png", dpi=600, bbox_inches='tight')

    print(f"[*] All-cases Refusal comparison plot saved to {base_path}.pdf 和 {base_path}.png")
    plt.close(fig)


# 设置样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 读取数据
with open(input_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f.readlines()]

# 所有 success 和 failed（用于全量对比图）
all_success_cases = [case for case in data if case['status'] == 'success']
all_failed_cases = [case for case in data if case['status'] == 'failed']

# ================== 新逻辑：按 total_turns = n 定义的目标分组 ==================
target_success_cases = [case for case in data
                        if case['status'] == 'success' and case['metrics']['total_turns'] <= selected_n]

target_failed_cases = [case for case in data
                       if (case['metrics']['total_turns'] > selected_n) or
                       (case['metrics']['total_turns'] == selected_n and case['status'] == 'failed')]


def compute_per_turn_matrices(cases, max_turn_limit=None):
    """计算每个 turn 的矩阵，缺失值保留为 NaN（后续用 nanmean 跳过）"""
    if not cases:
        return None, None

    turn_sims = defaultdict(list)
    turn_refs = defaultdict(list)

    for case in cases:
        trajectory = case['trajectory']
        actual_max = min(len(trajectory), max_turn_limit) if max_turn_limit else len(trajectory)
        for i in range(actual_max):
            turn = trajectory[i]
            turn_idx = i + 1
            if 'similarity_to_origin' in turn and turn['similarity_to_origin'] is not None:
                turn_sims[turn_idx].append(turn['similarity_to_origin'])
            if 'refusal_score' in turn and turn['refusal_score'] is not None:
                turn_refs[turn_idx].append(turn['refusal_score'])

    max_turn = max(turn_sims.keys()) if turn_sims else 0
    if max_turn == 0:
        return None, None

    # 构建矩阵，缺失位置保持 NaN（不再填充）
    sim_matrix = np.full((len(cases), max_turn), np.nan)
    ref_matrix = np.full((len(cases), max_turn), np.nan)

    for idx, case in enumerate(cases):
        trajectory = case['trajectory']
        actual_max = min(len(trajectory), max_turn_limit) if max_turn_limit else len(trajectory)
        for i in range(actual_max):
            turn = trajectory[i]
            turn_idx = i + 1
            if turn_idx <= max_turn:
                if 'similarity_to_origin' in turn and turn['similarity_to_origin'] is not None:
                    sim_matrix[idx, turn_idx - 1] = turn['similarity_to_origin']
                if 'refusal_score' in turn and turn['refusal_score'] is not None:
                    ref_matrix[idx, turn_idx - 1] = turn['refusal_score']

    return sim_matrix, ref_matrix


# 计算目标分组的数据
success_sim_matrix, success_ref_matrix = compute_per_turn_matrices(target_success_cases)

# Failed cases 仅取前 selected_n 个 turn
failed_sim_matrix, failed_ref_matrix = compute_per_turn_matrices(target_failed_cases, max_turn_limit=selected_n)

# ================== 绘制 Success Cases（total_turns <= n） ==================
if target_success_cases and success_sim_matrix is not None:
    visualize_dual_scores(
        success_sim_matrix,
        success_ref_matrix,
        save_name=f"success_dual_turns_{selected_n}_{model_name}",
        title=f'Success Cases (total_turns <= {selected_n}, {model_name})\n'
              f'Average Similarity & Refusal Score per Turn',
        show_individual=show_individual,
        band_mode=band_mode
    )
else:
    print(f"警告：没有找到 total_turns <= {selected_n} 的 success cases")

# ================== 绘制 Failed Cases（仅前 n turns） ==================
if target_failed_cases and failed_sim_matrix is not None:
    visualize_dual_scores(
        failed_sim_matrix,
        failed_ref_matrix,
        save_name=f"failed_dual_turns_{selected_n}_{model_name}",
        title=f'Failed Cases (for total_turns = {selected_n}, {model_name})\n'
              f'Average Similarity & Refusal Score (first {selected_n} turns only)',
        show_individual=show_individual,
        band_mode=band_mode
    )
else:
    print(f"警告：没有找到符合条件的 failed cases")

# ================== 绘制所有 Cases 的 Refusal 对比图 ==================
visualize_refusal_comparison_all(
    all_success_cases,
    all_failed_cases,
    save_name=f"refusal_comparison_all_{model_name}"
)

print("\n绘图完成！所有图片已输出到 pic/ 目录，适合直接用于论文。")