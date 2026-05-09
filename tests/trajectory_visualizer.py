#!/usr/bin/env python3
"""
visualize_results.py
读取 test_detail_{model}_with_refusal.jsonl，生成 10 张可视化图表。
不依赖模型推理，全部从文件读取。
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from collections import defaultdict

# ────────────────────────────────────────────────────────
# 命令行参数
# ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", "-i",
    default="test_detail_llama3.1_with_refusal.jsonl",
    help="with_refusal jsonl 文件"
)
parser.add_argument(
    "--output_dir", "-o",
    default="./figures",
    help="图表输出目录"
)
parser.add_argument(
    "--case_id", "-c",
    type=int, default=0,
    help="图1/图2 使用的单个 case_id"
)
args = parser.parse_args()

OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"]  = 11


# ────────────────────────────────────────────────────────
# 数据加载
# ────────────────────────────────────────────────────────
def load_with_refusal(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


print(f"[*] 读取 {args.input} ...")
all_details = load_with_refusal(args.input)

# 按 status 预分组，供 mode 参数使用
details_success = [d for d in all_details if d.get("status") == "success"]
details_fail    = [d for d in all_details if d.get("status") != "success"]
MODE_MAP        = {"all": all_details, "success": details_success, "fail": details_fail}

n_succ = len(details_success)
n_fail = len(details_fail)
print(f"[*] 共 {len(all_details)} 个 case（success={n_succ}, fail={n_fail}）")


# ────────────────────────────────────────────────────────
# 工具
# ────────────────────────────────────────────────────────
def save_fig(fig, name: str):
    path = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[✓] 已保存: {path}")


def get_traj_field(detail: dict, field: str) -> tuple[list, list]:
    """返回 (turn_ids, values)，跳过 None"""
    traj  = detail.get("trajectory", [])
    tids  = [t["turn_id"] for t in traj if t.get(field) is not None]
    vals  = [t[field]     for t in traj if t.get(field) is not None]
    return tids, vals


def avg_by_turn(details: list[dict], field: str) -> tuple[list, list, list]:
    """跨 case 汇总，返回 (sorted_turns, avg_values, counts)"""
    turn_vals = defaultdict(list)
    for d in details:
        for t in d.get("trajectory", []):
            tid = t.get("turn_id")
            val = t.get(field)
            if tid is not None and val is not None:
                turn_vals[tid].append(val)
    sorted_turns = sorted(turn_vals.keys())
    avg_vals = [np.mean(turn_vals[k]) for k in sorted_turns]
    counts   = [len(turn_vals[k])     for k in sorted_turns]
    return sorted_turns, avg_vals, counts


# ────────────────────────────────────────────────────────
# 图1: 单 case — similarity_score & refusal_score
# ────────────────────────────────────────────────────────
def plot_fig1(case_id: int):
    detail = next((d for d in all_details if d["case_id"] == case_id), None)
    if detail is None:
        print(f"[SKIP] 图1: case_id={case_id} 不存在"); return

    traj  = detail["trajectory"]
    turns = [t["turn_id"] for t in traj]
    sims  = [t.get("similarity_to_origin") for t in traj]
    refs  = [t.get("refusal_score")        for t in traj]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(turns, sims, "b-o", linewidth=2, label="Similarity Score")
    ax1.set_xlabel("Turn"); ax1.set_ylabel("Similarity Score (0–1)")
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"Case {case_id} — Similarity & Refusal Score per Turn")

    ax2 = ax1.twinx()
    ax2.plot(turns, refs, "r--s", linewidth=2, label="Refusal Score")
    ax2.set_ylabel("Refusal Score (cosine sim)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right")
    save_fig(fig, f"fig1_case{case_id}_similarity_refusal.png")


# ────────────────────────────────────────────────────────
# 图2: 单 case — harmfulness_score/10 & refusal_score
# ────────────────────────────────────────────────────────
def plot_fig2(case_id: int):
    detail = next((d for d in all_details if d["case_id"] == case_id), None)
    if detail is None:
        print(f"[SKIP] 图2: case_id={case_id} 不存在"); return

    traj  = detail["trajectory"]
    turns = [t["turn_id"] for t in traj]
    harms = [(t.get("harmfulness_score") or 0) / 10 for t in traj]
    refs  = [t.get("refusal_score") for t in traj]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(turns, harms, "g-o", linewidth=2, label="Harmfulness / 10")
    ax1.set_xlabel("Turn"); ax1.set_ylabel("Score (0–1)")
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"Case {case_id} — Harmfulness(/10) & Refusal Score per Turn")

    ax2 = ax1.twinx()
    ax2.plot(turns, refs, "r--s", linewidth=2, label="Refusal Score")
    ax2.set_ylabel("Refusal Score (cosine sim)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right")
    save_fig(fig, f"fig2_case{case_id}_harmfulness_refusal.png")


# ────────────────────────────────────────────────────────
# 图3: 所有 case 折线 — refusal_score（支持 mode）
# ────────────────────────────────────────────────────────
def plot_fig3(mode: str = "all"):
    src = MODE_MAP.get(mode, all_details)
    if not src:
        print(f"[SKIP] 图3 ({mode}): 无数据"); return

    fig, ax = plt.subplots(figsize=(14, 6))
    colors  = cm.tab20(np.linspace(0, 1, max(len(src), 1)))

    for i, detail in enumerate(src):
        tids, refs = get_traj_field(detail, "refusal_score")
        if tids:
            ax.plot(tids, refs, color=colors[i % len(colors)], alpha=0.4, linewidth=0.8)

    ax.set_xlabel("Turn"); ax.set_ylabel("Refusal Score (cosine sim)")
    ax.set_title(f"All Cases [{mode}] — Refusal Score per Turn")
    save_fig(fig, f"fig3_refusal_all_cases_{mode}.png")


# ────────────────────────────────────────────────────────
# 图4: 所有 case 折线 — harmfulness_score
# ────────────────────────────────────────────────────────
def plot_fig4():
    fig, ax = plt.subplots(figsize=(14, 6))
    colors  = cm.tab20(np.linspace(0, 1, max(len(all_details), 1)))

    for i, detail in enumerate(all_details):
        tids, harms = get_traj_field(detail, "harmfulness_score")
        if tids:
            ax.plot(tids, harms, color=colors[i], alpha=0.4, linewidth=0.8)

    ax.set_xlabel("Turn"); ax.set_ylabel("Harmfulness Score")
    ax.set_title("All Cases — Harmfulness Score per Turn")
    save_fig(fig, "fig4_harmfulness_all_cases.png")


# ────────────────────────────────────────────────────────
# 图5: 折线图 — total_turns 分布
# ────────────────────────────────────────────────────────
def plot_fig5():
    turn_counts = defaultdict(int)
    for d in all_details:
        total = d.get("metrics", {}).get("total_turns", 0)
        turn_counts[total] += 1

    xs = sorted(turn_counts.keys())
    ys = [turn_counts[x] for x in xs]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(xs, ys, "b-o", linewidth=2, markersize=5)
    ax.set_xlabel("Total Turns per Case"); ax.set_ylabel("Number of Cases")
    ax.set_title("Distribution of Total Turns per Case")
    ax.grid(True, alpha=0.3)
    save_fig(fig, "fig5_total_turns_distribution.png")


# ────────────────────────────────────────────────────────
# 图6: 平均 similarity_score 随 turn 变化
# ────────────────────────────────────────────────────────
def plot_fig6():
    sorted_turns, avg_sims, counts = avg_by_turn(all_details, "similarity_to_origin")

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(sorted_turns, avg_sims, "b-o", linewidth=2,
             markersize=5, label="Avg Similarity Score")
    ax1.set_xlabel("Turn"); ax1.set_ylabel("Average Similarity Score")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Average Similarity Score per Turn (All Cases)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(sorted_turns, counts, alpha=0.15, color="gray", label="# Cases")
    ax2.set_ylabel("Number of Cases at Turn", color="gray")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    save_fig(fig, "fig6_avg_similarity_per_turn.png")


# ────────────────────────────────────────────────────────
# 图7: 平均 refusal_score 随 turn 变化（支持 mode）
# ────────────────────────────────────────────────────────
def plot_fig7(mode: str = "all"):
    src = MODE_MAP.get(mode, all_details)
    sorted_turns, avg_refs, _ = avg_by_turn(src, "refusal_score")
    if not sorted_turns:
        print(f"[SKIP] 图7 ({mode}): 无数据"); return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sorted_turns, avg_refs, "r-o", linewidth=2, markersize=5)
    ax.set_xlabel("Turn"); ax.set_ylabel("Average Refusal Score (cosine sim)")
    ax.set_title(f"Average Refusal Score per Turn [{mode}]")
    ax.grid(True, alpha=0.3)
    save_fig(fig, f"fig7_avg_refusal_per_turn_{mode}.png")


# ────────────────────────────────────────────────────────
# 图8: 拒绝概率 vs similarity_score（分箱）
# ────────────────────────────────────────────────────────
def plot_fig8():
    sims_list, refused_list = [], []
    for d in all_details:
        for t in d.get("trajectory", []):
            sim      = t.get("similarity_to_origin")
            answered = t.get("is_answered")
            if sim is not None and answered is not None:
                sims_list.append(sim)
                refused_list.append(int(not answered))

    if not sims_list:
        print("[SKIP] 图8: 无数据"); return

    sims_arr    = np.array(sims_list)
    refused_arr = np.array(refused_list)
    bins        = np.linspace(0, 1, 21)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    refuse_rates, sample_counts = [], []

    for i in range(len(bins) - 1):
        mask = (sims_arr >= bins[i]) & (sims_arr < bins[i + 1])
        if mask.sum() > 0:
            refuse_rates.append(refused_arr[mask].mean())
            sample_counts.append(int(mask.sum()))
        else:
            refuse_rates.append(np.nan)
            sample_counts.append(0)

    valid = ~np.isnan(refuse_rates)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(bin_centers[valid], np.array(refuse_rates)[valid],
             "b-o", linewidth=2, markersize=5, label="Refusal Probability")
    ax1.set_xlabel("Similarity Score (binned)"); ax1.set_ylabel("Refusal Probability")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Refusal Probability vs. Similarity Score")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(bin_centers, sample_counts, width=0.045, alpha=0.2,
            color="gray", label="# Turns")
    ax2.set_ylabel("Number of Turns", color="gray")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    save_fig(fig, "fig8_refusal_prob_vs_similarity.png")


# ────────────────────────────────────────────────────────
# 图9: success vs fail — 平均 similarity_score 随 turn 变化
# ────────────────────────────────────────────────────────
def plot_fig9():
    turns_s, avg_s, _ = avg_by_turn(details_success, "similarity_to_origin")
    turns_f, avg_f, _ = avg_by_turn(details_fail,    "similarity_to_origin")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(turns_s, avg_s, "g-o",  linewidth=2, markersize=5, label=f"Success (n={n_succ})")
    ax.plot(turns_f, avg_f, "r--s", linewidth=2, markersize=5, label=f"Failed  (n={n_fail})")
    ax.set_xlabel("Turn"); ax.set_ylabel("Average Similarity Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Avg Similarity Score per Turn — Success vs. Failed Cases")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_fig(fig, "fig9_similarity_success_vs_failed.png")


# ────────────────────────────────────────────────────────
# 图10: success vs fail — 平均 refusal_score 随 turn 变化
# ────────────────────────────────────────────────────────
def plot_fig10():
    turns_s, avg_s, _ = avg_by_turn(details_success, "refusal_score")
    turns_f, avg_f, _ = avg_by_turn(details_fail,    "refusal_score")

    if not turns_s and not turns_f:
        print("[SKIP] 图10: 无 refusal_score 数据"); return

    fig, ax = plt.subplots(figsize=(12, 5))
    if turns_s:
        ax.plot(turns_s, avg_s, "g-o",  linewidth=2, markersize=5, label=f"Success (n={n_succ})")
    if turns_f:
        ax.plot(turns_f, avg_f, "r--s", linewidth=2, markersize=5, label=f"Failed  (n={n_fail})")
    ax.set_xlabel("Turn"); ax.set_ylabel("Average Refusal Score (cosine sim)")
    ax.set_title("Avg Refusal Score per Turn — Success vs. Failed Cases")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_fig(fig, "fig10_refusal_success_vs_failed.png")


# ────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_fig1(args.case_id)
    plot_fig2(args.case_id)

    for mode in ("all", "success", "fail"):
        plot_fig3(mode)
        plot_fig7(mode)

    plot_fig4()
    plot_fig5()
    plot_fig6()
    plot_fig8()
    plot_fig9()
    plot_fig10()

    print(f"\n✅ 所有图表已保存至 {OUTPUT_DIR}/")