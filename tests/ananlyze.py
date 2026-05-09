import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seabornundefined
import sns

sns
from typing import List, Dict
from pathlib import Path


class JailbreakTrajectoryAnalyzer:
    def __init__(self, results_path: str, output_dir: str = "analysis_results"):
        self.results_path = results_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.raw_results: List[Dict] = data.get("results", [])

        self.query_df = None  # query-level 底表
        self.sample_df = None  # sample-level 汇总表
        self._build_dataframes()
        print(
            f"✅ 加载完成 | 总样本数: {len(self.raw_results)} | 总查询次数: {len(self.query_df) if self.query_df is not None else 0}")

    def _build_dataframes(self):
        """构建三层数据表"""
        query_records = []
        sample_records = []

        for sample_idx, sample in enumerate(self.raw_results):
            original_q = sample.get("original_question", "")
            traj = sample.get("trajectory_time_order", [])
            success = sample.get("success", False)

            # query-level
            for turn_idx, turn in enumerate(traj):
                query_records.append({
                    "sample_id": sample_idx,
                    "turn_index": turn_idx,
                    "current_question": turn.get("question", ""),
                    "similarity": turn.get("similarity", 0.0),
                    "answered": int(turn.get("answered", False)),
                    "refused": int(turn.get("refused", True)),
                    "success": int(success),
                    "rps": turn.get("whitebox_refusal_score", 0.0),  # RPS = refusal propensity score
                    "step": turn.get("step", 0),
                    "regen_idx": turn.get("regen_idx", 0),
                    "response": turn.get("response", "")[:500] + "..." if len(
                        turn.get("response", "")) > 500 else turn.get("response", "")
                })

            # sample-level
            traj_df = pd.DataFrame(traj)
            answered_turns = traj_df[traj_df["answered"] == True]
            refused_turns = traj_df[traj_df["answered"] == False]

            last_answered_sim = answered_turns["similarity"].max() if not answered_turns.empty else np.nan
            first_refused_sim = refused_turns["similarity"].min() if not refused_turns.empty else np.nan
            boundary_width = first_refused_sim - last_answered_sim if pd.notna(first_refused_sim) and pd.notna(
                last_answered_sim) else np.nan

            sample_records.append({
                "sample_id": sample_idx,
                "total_turns": len(traj),
                "reached_step3": int(sample.get("final_answered_step", 0) >= 3),
                "jailbreak_success": int(success),
                "max_similarity": traj_df["similarity"].max() if not traj_df.empty else 0.0,
                "last_answered_similarity": last_answered_sim,
                "first_refused_similarity": first_refused_sim,
                "boundary_width": boundary_width,
                "final_answered_step": sample.get("final_answered_step", 0),
                "category": sample.get("category", "unknown")
            })

        self.query_df = pd.DataFrame(query_records)
        self.sample_df = pd.DataFrame(sample_records)

        # 保存三层表
        self.query_df.to_csv(self.output_dir / "query_level.csv", index=False, encoding='utf-8')
        self.sample_df.to_csv(self.output_dir / "sample_level.csv", index=False, encoding='utf-8')
        print("💾 已保存 query_level.csv 和 sample_level.csv")

    # ==================== 图 1：轨迹长度分布 ====================
    def plot_trajectory_length_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.sample_df, x="total_turns", bins=30, kde=True)
        plt.title("图 1：轨迹长度分布（每条样本实际执行的查询次数）")
        plt.xlabel("轨迹长度（查询次数）")
        plt.ylabel("样本数")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "fig1_trajectory_length.png", dpi=300, bbox_inches='tight')
        plt.close()

    # ==================== 图 2：生存曲线 ====================
    def plot_survival_curve(self):
        max_turns = self.query_df["turn_index"].max() + 1
        survival = []
        for t in range(max_turns):
            remaining = len(self.sample_df[self.sample_df["total_turns"] > t])
            survival.append(remaining / len(self.sample_df))

        plt.figure(figsize=(10, 6))
        plt.plot(range(max_turns), survival, marker='o', linewidth=2.5)
        plt.title("图 2：生存曲线（仍未终止的样本比例）")
        plt.xlabel("查询步数")
        plt.ylabel("仍未终止的样本比例")
        plt.grid(True, alpha=0.3)
        for i, v in enumerate(survival):
            if i % 5 == 0 or i == len(survival) - 1:
                plt.text(i, v, f"{v:.2%}", ha='center', va='bottom')
        plt.savefig(self.output_dir / "fig2_survival_curve.png", dpi=300, bbox_inches='tight')
        plt.close()

    # ==================== 图 3：平均相似度曲线 ====================
    def plot_avg_similarity_curve(self):
        grouped = self.query_df.groupby("turn_index").agg(
            mean_similarity=("similarity", "mean"),
            count=("sample_id", "count")
        ).reset_index()

        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax1.plot(grouped["turn_index"], grouped["mean_similarity"],
                 marker='o', linewidth=3, color='#1f77b4', label="平均语义相似度")
        ax1.set_xlabel("查询步数（原始执行顺序）")
        ax1.set_ylabel("平均语义相似度", color='#1f77b4')
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(grouped["turn_index"], grouped["count"],
                 marker='s', linestyle='--', color='#ff7f0e', label="参与样本数")
        ax2.set_ylabel("样本数", color='#ff7f0e')

        plt.title("图 3：原始执行顺序上的平均相似度曲线")
        fig.legend(loc="upper right")
        plt.savefig(self.output_dir / "fig3_avg_similarity_curve.png", dpi=300, bbox_inches='tight')
        plt.close()

    # ==================== 图 4：平均 RPS / 拒绝分数曲线 ====================
    def plot_avg_refusal_curve(self):
        grouped = self.query_df.groupby("turn_index")["rps"].mean().reset_index()

        plt.figure(figsize=(12, 7))
        plt.plot(grouped["turn_index"], grouped["rps"],
                 marker='o', linewidth=3, color='#d62728')
        plt.title("图 4：原始执行顺序上的平均白盒拒绝分数曲线")
        plt.xlabel("查询步数")
        plt.ylabel("平均 RPS (Refusal Propensity Score)")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "fig4_avg_refusal_curve.png", dpi=300, bbox_inches='tight')
        plt.close()

    # ==================== 图 5：相似度分桶下的拒绝率曲线 ====================
    def plot_refusal_by_similarity_bin(self):
        self.query_df["sim_bin"] = pd.cut(self.query_df["similarity"],
                                          bins=np.arange(0, 1.05, 0.05),
                                          labels=[f"{i / 20:.2f}-{(i + 1) / 20 :.2f}" for i in range(20)])

        bin_stats = self.query_df.groupby("sim_bin").agg(
            refusal_rate=("refused", "mean"),
            count=("sample_id", "count")
        ).reset_index()

        plt.figure(figsize=(12, 7))
        sns.lineplot(x=bin_stats["sim_bin"].astype(str), y=bin_stats["refusal_rate"],
                     marker='o', linewidth=3)
        plt.title("图 5：相似度分桶下的拒绝率曲线")
        plt.xlabel("语义相似度区间")
        plt.ylabel("拒绝率")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "fig5_refusal_by_similarity_bin.png", dpi=300, bbox_inches='tight')
        plt.close()

    # ==================== 图 6：成功组 vs 失败组对比 ====================
    def plot_success_vs_failure(self):
        self.query_df["normalized_position"] = self.query_df.groupby("sample_id")["turn_index"].transform(
            lambda x: x / (x.max() if x.max() > 0 else 1)
        )

        grouped = self.query_df.groupby(["jailbreak_success", "normalized_position"]).agg(
            mean_similarity=("similarity", "mean"),
            mean_rps=("rps", "mean")
        ).reset_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # 6a 相似度
        sns.lineplot(data=grouped, x="normalized_position", y="mean_similarity",
                     hue="jailbreak_success", ax=ax1, marker='o', linewidth=2.5)
        ax1.set_title("6a. 成功组 vs 失败组 - 归一化相似度轨迹")
        ax1.set_xlabel("归一化轨迹位置 (0~1)")
        ax1.set_ylabel("平均语义相似度")
        ax1.grid(True, alpha=0.3)

        # 6b RPS
        sns.lineplot(data=grouped, x="normalized_position", y="mean_rps",
                     hue="jailbreak_success", ax=ax2, marker='o', linewidth=2.5)
        ax2.set_title("6b. 成功组 vs 失败组 - 归一化 RPS 轨迹")
        ax2.set_xlabel("归一化轨迹位置 (0~1)")
        ax2.set_ylabel("平均白盒拒绝分数")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "fig6_success_vs_failure.png", dpi=300, bbox_inches='tight')
        plt.close()

    # ==================== 图 7：边界区间宽度分布 ====================
    def plot_boundary_width(self):
        valid = self.sample_df.dropna(subset=["boundary_width"])

        plt.figure(figsize=(10, 6))
        sns.histplot(valid["boundary_width"], bins=20, kde=True)
        plt.title("图 7：边界区间宽度分布")
        plt.xlabel("边界宽度 (first_refused_similarity - last_answered_similarity)")
        plt.ylabel("样本数")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "fig7_boundary_width.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"平均边界宽度: {valid['boundary_width'].mean():.4f} ± {valid['boundary_width'].std():.4f}")

    def run_all_analysis(self):
        """一键运行全部7张图"""
        print("🚀 开始生成7张分析图...")
        self.plot_trajectory_length_distribution()  # 图1
        self.plot_survival_curve()  # 图2
        self.plot_avg_similarity_curve()  # 图3
        self.plot_avg_refusal_curve()  # 图4
        self.plot_refusal_by_similarity_bin()  # 图5
        self.plot_success_vs_failure()  # 图6
        self.plot_boundary_width()  # 图7
        print(f"✅ 全部7张图已保存至: {self.output_dir}")


# ====================== 使用示例 ======================
if __name__ == "__main__":
    analyzer = JailbreakTrajectoryAnalyzer(
        results_path="experiment_results/enhanced_with_whitebox.json",  # 推荐使用白盒增强后的结果
        output_dir="analysis_results"
    )
    analyzer.run_all_analysis()