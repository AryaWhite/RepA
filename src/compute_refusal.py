#!/usr/bin/env python3
"""
compute_refusal.py

功能：
1. 计算 refusal_score（RepE）
2. 重算 similarity_to_origin（cosine）
3. 删除 question / response
4. 保留 fallback（防止异常 refusal_vector）
"""

import json
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from slope import *
from utils import *
from sentence_transformers import SentenceTransformer, util


# ────────────────────────────────────────────────────────
# 数据 I/O
# ────────────────────────────────────────────────────────
def load_detail(path: str):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ────────────────────────────────────────────────────────
# refusal_score
# ────────────────────────────────────────────────────────
def compute_refusal_score(question, model, tokenizer, refusal_vec_t):
    prompt = LLM.user_prompt(question)
    hidden = get_hidden_states(model, tokenizer, [prompt])
    hidden_t = torch.tensor(hidden, dtype=torch.float32)

    hidden_norm = torch.nn.functional.normalize(hidden_t, p=2, dim=1)
    ref_norm = torch.nn.functional.normalize(refusal_vec_t, p=2, dim=1)

    score = util.cos_sim(hidden_norm, ref_norm).item()
    return round(float(score), 6)


# ────────────────────────────────────────────────────────
# similarity
# ────────────────────────────────────────────────────────
def compute_similarity(origin_q, question, sim_model):
    emb1 = sim_model.encode(origin_q, convert_to_tensor=True, normalize_embeddings=True)
    emb2 = sim_model.encode(question, convert_to_tensor=True, normalize_embeddings=True)
    return round(util.cos_sim(emb1, emb2).item(), 6)


# ────────────────────────────────────────────────────────
# 精简 turn
# ────────────────────────────────────────────────────────
KEEP_FIELDS = {
    "turn_id",
    "step_label",
    "is_answered",
    "similarity_to_origin",
    "refusal_score",
    "harmfulness_score",
    "jailbreak_score"
}

def compress_turn(turn: dict) -> dict:
    return {k: turn.get(k) for k in KEEP_FIELDS}


# ────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="./Output/test_detail_llama3.1.jsonl")
    parser.add_argument("--model", "-m", default="llama3.1")
    parser.add_argument("--device", "-d", default="cuda:0")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = input_path.parent / (input_path.stem + "_with_refusal.jsonl")

    # ── 模型加载 ─────────────────────────────────────────
    print(f"[*] 加载 LLM: {args.model}")
    lm = llm_adapter(args.model, device=args.device)

    print("[*] 加载 similarity 模型")
    sim_model = SentenceTransformer("/home/breaker/public_models/all-MiniLM-L6-v2", device=args.device)

    print("[*] 计算 refusal_vector")
    refusal_vec = calc_refusal_vector2(lm)

    # ── 类型修复 ─────────────────────────────────────────
    if isinstance(refusal_vec, np.ndarray):
        refusal_vec = refusal_vec.astype(np.float32)
    else:
        refusal_vec = np.array(refusal_vec, dtype=np.float32)

    refusal_vec = refusal_vec.flatten()

    print(f"[DEBUG] refusal_vec shape: {refusal_vec.shape}, std={refusal_vec.std():.6f}")

    # ────────────────────────────────────────────────────
    # ✅ fallback（保留！关键部分）
    # ────────────────────────────────────────────────────
    if refusal_vec.shape[0] <= 1 or np.allclose(refusal_vec, refusal_vec[0], atol=1e-5):
        print("⚠️ 检测到异常 refusal_vector，执行 fallback")

        probe_dir = Path("data/probe_inst/bh_ques")
        benign_path = probe_dir / "benign_test.txt"
        harmful_path = probe_dir / "harmful_test.txt"

        with open(benign_path, encoding="utf-8") as f:
            benign_lines = [line.strip() for line in f if line.strip()]
        with open(harmful_path, encoding="utf-8") as f:
            harmful_lines = [line.strip() for line in f if line.strip()]

        min_len = min(len(benign_lines), len(harmful_lines))

        if min_len < 10:
            raise ValueError("❌ fallback失败：probe数据不足")

        print(f"[*] fallback使用 {min_len} 对样本")

        benign_repr = get_hidden_states(lm.model, lm.tokenizer, benign_lines[:min_len])
        evil_repr   = get_hidden_states(lm.model, lm.tokenizer, harmful_lines[:min_len])

        diff_repr = np.array(evil_repr) - np.array(benign_repr)
        refusal_vec = diff_repr.mean(axis=0).astype(np.float32)

        print(f"[DEBUG] fallback后 shape: {refusal_vec.shape}, std={refusal_vec.std():.6f}")

    # ── tensor化 ─────────────────────────────────────────
    refusal_vec_t = torch.tensor(
        np.expand_dims(refusal_vec, 0),
        dtype=torch.float32
    )

    print(f"[*] 最终 refusal_vector shape: {refusal_vec_t.shape}")

    # ── 读取数据 ─────────────────────────────────────────
    print(f"[*] 读取 {input_path}")
    all_details = load_detail(str(input_path))

    output_records = []

    for detail in tqdm(all_details, desc="Processing"):

        new_detail = {
            "case_id": detail.get("case_id"),
            "category": detail.get("category"),
            "status": detail.get("status"),
            "metrics": detail.get("metrics"),
        }

        origin_q = detail.get("origin_question", "")
        traj = detail.get("trajectory", [])

        new_traj = []

        for turn in traj:
            question = turn.get("question", "")

            new_turn = copy.deepcopy(turn)

            # ===== refusal =====
            if question:
                new_turn["refusal_score"] = compute_refusal_score(
                    question,
                    lm.model,
                    lm.tokenizer,
                    refusal_vec_t
                )
            else:
                new_turn["refusal_score"] = None

            # ===== similarity =====
            if question and origin_q:
                new_turn["similarity_to_origin"] = compute_similarity(
                    origin_q,
                    question,
                    sim_model
                )
            else:
                new_turn["similarity_to_origin"] = None

            # ===== 压缩 =====
            new_traj.append(compress_turn(new_turn))

        new_detail["trajectory"] = new_traj
        output_records.append(new_detail)

    # ── 保存 ─────────────────────────────────────────────
    save_jsonl(output_records, str(output_path))

    print(f"\n✅ 已生成: {output_path}")


if __name__ == "__main__":
    main()