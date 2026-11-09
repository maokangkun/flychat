"""data_prep.py — LCCC-Base 过滤 → SFT 多轮对话集（对话 LLM 微调用）.

过滤规则（与 corpus.py v2 一致）：每轮 4-30 字、2-8 轮、超半数轮次含中文。
采样：种子 0 流式抽样 N 段（默认 60k），按首 turn 哈希去重。
输出 JSONL：{"id", "messages":[{role,content}...], "user_turns":[...],
             "cum_user_char_pos":[...]}（脑状态采集/对位用）。
"""
import gzip, hashlib, json, os, random, sys

sys.path.insert(0, "/data/mkk/malecns/flychat")
from flychat.corpus import _lccc_filter, desegment_cn, normalize_turn

GZ = "/data/mkk/malecns/flychat/data/lccc/lccc_base_train.jsonl.gz"
OUT = "/data/mkk/malecns/flychat/data/sft/lccc_sft_50k.jsonl"
N_TARGET = int(os.environ.get("SFT_N", "60000"))

def build_turns(dialogue):
    turns = _lccc_filter(dialogue)
    return turns

def main():
    rng = random.Random(0)
    seen = set()
    cand = []
    n_kept, n_seen = 0, 0
    with gzip.open(GZ, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            n_seen += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            if not isinstance(d, list):
                continue
            t = build_turns(d)
            if t is None:
                continue
            # 偶数轮（user/assistant 交替）；奇数轮丢最后一个 user turn
            if len(t) % 2 == 1:
                t = t[:-1]
            if not (2 <= len(t) <= 8):
                continue
            h = hashlib.md5(t[0].encode()).digest()[:6]
            if h in seen:
                continue
            seen.add(h)
            n_kept += 1
            if len(cand) < N_TARGET:
                cand.append((rng.random(), t))
            else:
                j = rng.randint(0, n_kept - 1)
                if j < N_TARGET:
                    cand[j] = (rng.random(), t)
            if n_seen % 1000000 == 0:
                print(f"scanned {n_seen} lines, kept {n_kept}, pool {len(cand)}", flush=True)
            if n_kept >= N_TARGET * 4:  # 样本池已足够，提前停止省时
                break
    rows = []
    for rid, (_p, t) in enumerate(sorted(cand)[:N_TARGET]):
        messages = []
        for i, x in enumerate(t):
            messages.append({"role": "user" if i % 2 == 0 else "assistant",
                             "content": x})
        # user turn 边界 -> 字符位置（脑仿真按累计文本逐 turn 停窗）
        cum = []
        pos = 0
        for m in messages:
            pos += len(m["content"])
            if m["role"] == "user":
                cum.append(pos)
            pos += 2  # token 换行分隔近似
        rows.append({"id": rid, "messages": messages,
                     "user_turns": [m["content"] for m in messages if m["role"] == "user"],
                     "cum_user_char_pos": cum})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, OUT)
    print(f"DONE scanned={n_seen} unique={n_kept} wrote={len(rows)} -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
