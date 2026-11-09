"""collect_sft_states.py — 为 SFT 对话子集采集果蝇脑状态（每 user turn 一个向量）.

对每条 SFT 对话：把累计对话文本（turns 以 '\n' 相连，char token 化）一次
送入 FlyBrain（teacher forcing），在每个 user turn 末尾记录状态向量。
输出 data/sft/brainstates_shardXX.npz：ids/n_turns/states(fp16 拼接)。

用法：
  python flychat/llm/collect_sft_states.py worker <shard> <n_shards> <n_dialogues>
  python flychat/llm/collect_sft_states.py driver <n_per_gpu> <gpu...> <n_dialogues>
"""
import os
os.environ.setdefault("KMP_AFFINITY", "disabled")
import glob, json, random, sys, time
import numpy as np
import torch

sys.path.insert(0, "/data/mkk/malecns/flychat")
sys.path.insert(0, "/data/mkk/malecns/flybrain")
from flychat.interface import BrainSession, StimEncoder
from flychat.corpus import CharTokenizer

ROOT = "/data/mkk/malecns/flychat"
JSONL = f"{ROOT}/data/sft/lccc_sft_50k.jsonl"
OUT_DIR = f"{ROOT}/data/sft"
STATE_DIM = 8062
N_OLF, N_VIS, N_DN = 2639, 4109, 1314
W_MS = float(os.environ.get("BS_WINDOW_MS", "4.0"))   # 窗口（默认 4ms：匹配在线注入通道）
MAX_TOKENS_PER_EPISODE = 400

with open(f"{ROOT}/data/split_v2.json") as f:
    CHARS = json.load(f)["vocab"]

def make_tok():
    tok = CharTokenizer()
    tok.chars = CHARS
    tok.stoi = {c: i for i, c in enumerate(CHARS)}
    tok.itos = {i: c for i, c in enumerate(CHARS)}
    tok.vocab_size = len(CHARS)
    return tok

def load_dialogues(n):
    rows = []
    with open(JSONL) as f:
        for line in f:
            rows.append(json.loads(line))
    rows = rows[:]                     # 不复制大对象时保持原顺序
    random.Random(0).shuffle(rows)
    return rows[:n]


def worker(shard, n_shards, n_dialogues):
    t0 = time.time()
    torch.manual_seed(0)
    tok = make_tok()
    enc = StimEncoder(len(CHARS), 64, N_OLF, N_VIS, seed=0).cuda()
    ck = f"{ROOT}/checkpoints/encoder.pt"
    if os.path.exists(ck):
        sd = torch.load(ck, map_location="cpu")
        if sd["emb.weight"].shape[0] == len(CHARS):
            enc.load_state_dict(sd)
    sess = BrainSession(w_scale=0.275, device="cuda")
    rows = load_dialogues(n_dialogues)
    mine = [r for i, r in enumerate(rows) if i % n_shards == shard]
    out_path = f"{OUT_DIR}/brainstates_shard{shard:02d}.npz"
    n_d, all_states, all_ids, all_nt = 0, [], [], []
    for k, r in enumerate(mine):
        turns = [m["content"] for m in r["messages"]]
        user_end = [i for i, m in enumerate(r["messages"]) if m["role"] == "user"]
        cum_end, pos = [], 0
        for i in user_end:
            pos += len(r["messages"][i]["content"]) - 1
            cum_end.append(pos)
            pos += 2
        text = "\n".join(turns)
        ids = tok.encode(text)
        if len(ids) > MAX_TOKENS_PER_EPISODE:
            ids = ids[:MAX_TOKENS_PER_EPISODE]
        end_clip = [p for p in cum_end if p < len(ids)]
        if not end_clip:
            continue
        st = sess.run_episode(ids, enc, window_ms=W_MS)     # [T, 8062]
        sel = st[end_clip].half().cpu().numpy()
        all_states.append(sel)
        all_ids.append(r["id"]); all_nt.append(sel.shape[0])
        n_d += 1
        if (k + 1) % 20 == 0:
            el = time.time() - t0
            print(f"[shard {shard}] {k+1}/{len(mine)} done ({el:.0f}s, "
                  f"{el/(k+1):.1f}s/dialogue)", flush=True)
            if n_d > 0:
                np.savez(f"{OUT_DIR}/brainstates_shard{shard:02d}.tmp.npz",
                         ids=np.asarray(all_ids), n_turns=np.asarray(all_nt),
                         states=np.concatenate(all_states, axis=0),
                         offsets=np.cumsum([0]+[s.shape[0] for s in all_states]))
                os.replace(f"{OUT_DIR}/brainstates_shard{shard:02d}.tmp.npz",
                           out_path)
    if all_states:
        np.savez(out_path, ids=np.asarray(all_ids), n_turns=np.asarray(all_nt),
                 states=np.concatenate(all_states, axis=0))
    with open(f"{OUT_DIR}/brainstates_shard{shard:02d}_DONE", "w") as f:
        f.write(str(n_d))
    print(f"[shard {shard}] ALL DONE {n_d} dialogues", flush=True)

def driver(n_per_gpu, gpus, n_dialogues):
    """监督进程：为每个 shard 维持一个活跃 worker，结束后打 DONE 标记。"""
    plan = [g for g in gpus for _ in range(n_per_gpu)]
    n_shards = len(plan)
    print(f"driver: {n_shards} shards on gpus {gpus}, {n_dialogues} dialogues",
          flush=True)
    procs = {}
    while True:
        all_done = True
        for shard in range(n_shards):
            done_m = f"{OUT_DIR}/brainstates_shard{shard:02d}_DONE"
            if os.path.exists(done_m):
                continue
            all_done = False
            p = procs.get(shard)
            if p is not None and p.poll() is None:
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan[shard]))
            log = open(f"{ROOT}/logs/bs_shard{shard:02d}.log", "a")
            procs[shard] = os.spawnle(
                os.P_NOWAIT, sys.executable, sys.executable, __file__,
                "worker", str(shard), str(n_shards), str(n_dialogues), env)
            print(f"driver: (re)started shard {shard} on gpu {plan[shard]}",
                  flush=True)
        if all_done:
            print("ALL COMPLETE", flush=True)
            break
        time.sleep(60)

if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif sys.argv[1] == "driver":
        driver(int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")],
               int(sys.argv[4]))
