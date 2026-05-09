import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ================== 提取 model 名称（可选） ==================
def extract_model_name(filename):
    match = re.search(r'test_detail_(.+?)_with_refusal\.jsonl', filename)
    return match.group(1) if match else "unknown_model"

input_file = 'test_detail_llama3.1_with_refusal.jsonl'
model_name = extract_model_name(input_file)

# ================== 应用参考代码的样式 ==================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# ================== 读取并解析数据 ==================
records = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            case = json.loads(line)
            case_id = case['case_id']
            is_success = (case['status'] == 'success')
            for turn in case['trajectory']:
                records.append({
                    'case_id': case_id,
                    'turn': turn['turn_id'],
                    'refusal_score': turn['refusal_score'],
                    'similarity_score': turn['similarity_to_origin'] + 0.1,  # 整体 +0.1
                    'is_success': is_success
                })

df = pd.DataFrame(records)

# ================== 参数设置 ==================
n = 9  # 前 n 个 turn，可调整
df = df[df['turn'] <= n].copy()

# 根据整体 status 划分 success / failed
case_success_status = df.groupby('case_id')['is_success'].first().to_dict()
success_cases = [cid for cid, status in case_success_status.items() if status]
failed_cases  = [cid for cid, status in case_success_status.items() if not status]

print(f"Success cases: {len(success_cases)}, Failed cases: {len(failed_cases)}")

# ================== 计算每个 turn 的均值（只使用实际存在的值） ==================
def compute_mean_by_turn(df, case_list, metric):
    sub = df[df['case_id'].isin(case_list)]
    result = sub.groupby('turn')[metric].mean().reset_index()
    result.rename(columns={metric: 'mean'}, inplace=True)
    return result

ref_success = compute_mean_by_turn(df, success_cases, 'refusal_score')
ref_failed  = compute_mean_by_turn(df, failed_cases,  'refusal_score')

sim_success = compute_mean_by_turn(df, success_cases, 'similarity_score')
sim_failed  = compute_mean_by_turn(df, failed_cases,  'similarity_score')

# ================== 绘图函数（参考 visualize_refusal_comparison_all 样式） ==================
def plot_comparison_curve(turns_success, values_success, label_success,
                          turns_failed, values_failed, label_failed,
                          ylabel, title, save_name):
    """绘制两条曲线的对比图，样式与参考代码一致"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Success 曲线（蓝色实线）
    ax.plot(turns_success, values_success,
            color="#1f77b4", linewidth=2.8, marker="o", markersize=7,
            label=label_success)

    # Failed 曲线（橙色虚线）
    ax.plot(turns_failed, values_failed,
            color="#ff7f0e", linewidth=2.8, marker="s", markersize=7,
            linestyle="--", label=label_failed)

    ax.set_xlabel("Turn Number", fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    ax.set_xticks(range(1, n+1))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=12, loc="lower right")

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{save_name}.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(f'{save_name}.png', dpi=600, bbox_inches='tight')
    print(f"[*] Saved: {save_name}.pdf / .png")
    plt.close(fig)

# ================== 分别绘制两张图 ==================

# 1. Refusal Score 对比图
plot_comparison_curve(
    turns_success=ref_success['turn'], values_success=ref_success['mean'],
    label_success="Success Cases",
    turns_failed=ref_failed['turn'], values_failed=ref_failed['mean'],
    label_failed="Failed Cases",
    ylabel="Average Refusal Score",
    title=f'Refusal Score Comparison (Top {n} Turns, {model_name})',
    save_name=f'refusal_comparison_top{n}_{model_name}'
)

# 2. Similarity Score 对比图
plot_comparison_curve(
    turns_success=sim_success['turn'], values_success=sim_success['mean'],
    label_success="Success Cases",
    turns_failed=sim_failed['turn'], values_failed=sim_failed['mean'],
    label_failed="Failed Cases",
    ylabel="Average Similarity Score",
    title=f'Similarity Score Comparison (Top {n} Turns, {model_name})',
    save_name=f'similarity_comparison_top{n}_{model_name}'
)

print("\n绘图完成！")