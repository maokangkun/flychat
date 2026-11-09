"""collect_states.py — 并行收集果蝉脑状态轨迹（支持断点续跑）。

每个 worker 进程一个 FlyBrain 实例，对分到的对话序列做全脑仿真，
定期写增量 checkpoint（partial/*.npz，只保留最近 2 个以控磁盘），结束后合并。

  python collect_states.py driver          # 启动全部 worker（setsid 调用）
  python collect_states.py worker <shard_id> <n_shards>
  python collect_states.py merge

FLYCHAT_LANG: en=英文原版, cn=双语v1, v2=扩充双语v2（16 worker + oldtok_test 对比采集）
"""
import os
os.environ.setdefault("KMP_AFFINITY", "disabled")

import glob
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/data/mkk/malecns/flychat")
from flychat.corpus import build_sequences
from flychat.interface import StimEncoder, BrainSession

ROOT = "/data/mkk/malecns/flychat"
# FLYCHAT_LANG: en=英文原版, cn=双语v1, v2=扩充双语v2
_LANG = os.environ.get("FLYCHAT_LANG", "en")
_DIRS = {
    "en":  ("shards",     "split.json"),
    "cn":  ("shards_cn",  "split_cn.json"),
    "v2":  ("shards_v2",  "split_v2.json"),
}
SHARD_NAME, SPLIT_NAME = _DIRS.get(_LANG, _DIRS["en"])
SHARD_DIR = os.path.join(ROOT, "data", SHARD_NAME)
PARTIAL_DIR = os.path.join(SHARD_DIR, "partial")
SPLIT_PATH = os.path.join(ROOT, "data", SPLIT_NAME)
OLD_SPLIT_PATH = os.path.join(ROOT, "data", "split_cn.json")   # v1 tokenizer（旧模型对比）
IS_V2 = _LANG == "v2"

N_OLF, N_VIS, N_DN = 2639, 4109, 1314
STATE_DIM = N_OLF + N_VIS + N_DN
W_MS = 16.0
D_EMB = 64
# v1: 1000/100/100（原划分）；v2: test/val 吂 500，其余为 train
SPLITS = {"train": 1000, "val": 100, "test": 100} if not IS_V2 else \
         {"test": 500, "val": 500, "train": None}
CKPT_EVERY = 10 if IS_V2 else 5
WORKERS_PER_GPU = 2 if IS_V2 else 1


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def make_split():
    """构建划分（种子 0）。v2 额外保存 test 原文，供旧 tokenizer 对比评估。"""
    if IS_V2:
        tok, seqs, texts = build_sequences(version="v2", return_texts=True)
    else:
        tok, seqs = build_sequences()
        texts = None
    rng = np.random.RandomState(0)
    order = rng.permutation(len(seqs))
    split = {}
    i = 0
    for name, n in SPLITS.items():
        if n is None:                      # train -> 其余全部
            split[name] = [int(j) for j in order[i:]]
        else:
            split[name] = [int(j) for j in order[i:i + n]]
            i += n
    meta = {"splits": split, "vocab": tok.chars}
    if texts is not None:
        meta["test_texts"] = [texts[j] for j in split["test"]]
    tmp = SPLIT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f)
    os.replace(tmp, SPLIT_PATH)          # 原子写：多 worker 并发不截断
    return tok, seqs, split


def _load_old_tokenizer():
    """v1 词表 -> tokenizer（旧 2.5M 模型的输入视图）. """
    from flychat.corpus import CharTokenizer
    with open(OLD_SPLIT_PATH) as f:
        chars = json.load(f)["vocab"]
    tok = CharTokenizer()
    tok.chars = chars
    tok.stoi = {c: i for i, c in enumerate(chars)}
    tok.itos = {i: c for i, c in enumerate(chars)}
    tok.vocab_size = len(chars)
    return tok


def worker(shard_id, n_shards):
    print(f"[shard {shard_id}] starting", flush=True)
    tok, seqs, split = make_split()
    os.makedirs(PARTIAL_DIR, exist_ok=True)
    torch.manual_seed(0)
    enc = StimEncoder(len(tok.chars), D_EMB, N_OLF, N_VIS, seed=0)

    old_enc, old_seqs = None, None
    if IS_V2:
        # 旧 tokenizer 视角的 test 序列（与 v2 test 对话逐条对齐）
        old_tok = _load_old_tokenizer()
        with open(SPLIT_PATH) as f:
            test_texts = json.load(f)["test_texts"]
        old_seqs = [old_tok.encode(t)[:128] for t in test_texts]
        old_seqs = [s if len(s) >= 16 else None for s in old_seqs]
        old_enc = StimEncoder(len(old_tok.chars), D_EMB, N_OLF, N_VIS, seed=0)
        old_enc.load_state_dict(
            torch.load(os.path.join(ROOT, "checkpoints", "encoder_small.pt"),
                       map_location="cpu"))

    for attempt in range(20):
        try:
            sess = BrainSession(w_scale=0.275, device="cuda:0")
            break
        except Exception as e:
            print(f"[shard {shard_id}] init failed ({e}); retry in 60s",
                  flush=True)
            time.sleep(60)
    else:
        raise RuntimeError("init failed 20 times")
    enc = enc.to(sess.brain.device)
    old_enc = old_enc.to(sess.brain.device) if old_enc is not None else None

    if IS_V2:
        # v2: 词表变了 -> 覆盖写 encoder.pt（原子替换，避免并发损坏）
        _atomic_save(enc.state_dict(),
                     os.path.join(ROOT, "checkpoints", "encoder.pt"))
    elif not os.path.exists(os.path.join(ROOT, "checkpoints", "encoder.pt")):
        torch.save(enc.state_dict(),
                   os.path.join(ROOT, "checkpoints", "encoder.pt"))

    all_splits = ("train", "val", "test", "oldtok_test") if IS_V2 \
        else ("train", "val", "test")
    for split_name in all_splits:
        enc_l, seqs_l = enc, seqs
        if split_name == "oldtok_test":
            ids = [k for k in range(len(split["test"]))
                   if old_seqs[k] is not None and k % n_shards == shard_id]
            enc_l, seqs_l = old_enc, old_seqs
        else:
            ids = split[split_name][shard_id::n_shards]
        prefix = os.path.join(PARTIAL_DIR, f"{split_name}_shard{shard_id:02d}")
        # ---- 断点续跑：找最大已完成 k ----
        done = -1
        for f in glob.glob(prefix + "_*.npz"):
            done = max(done, int(f.rsplit("_", 1)[1][:-4]))
        states_list, tokens_list, offsets = [], [], [0]
        if done >= 0:
            d = np.load(f"{prefix}_{done:05d}.npz")
            states_list.append(d["states"])
            tokens_list.append(d["tokens"])
            offsets = list(d["offsets"])
        t0 = time.time()
        for k in range(done + 1, len(ids)):
            st = sess.run_episode(seqs_l[ids[k]], enc_l, window_ms=W_MS)
            states_list.append(st.half().cpu().numpy())
            tokens_list.append(np.asarray(seqs_l[ids[k]], dtype=np.int16))
            offsets.append(offsets[-1] + len(seqs_l[ids[k]]))
            if (k + 1) % CKPT_EVERY == 0 or k == len(ids) - 1:
                np.savez(f"{prefix}_{k:05d}.npz",
                         states=np.concatenate(states_list, axis=0),
                         tokens=np.concatenate(tokens_list),
                         offsets=np.asarray(offsets, dtype=np.int32))
                # ---- 修剪旧 partial：只保留最近 2 个（磁盘可控）----
                ks = sorted(int(f.rsplit("_", 1)[1][:-4]) for f in
                            glob.glob(prefix + "_*.npz"))
                for old_k in ks[:-2]:
                    os.remove(f"{prefix}_{old_k:05d}.npz")
                print(f"[shard {shard_id}/{n_shards}] {split_name} {k+1}/{len(ids)}",
                      flush=True)
        el = time.time() - t0
        print(f"[shard {shard_id}] {split_name} done ({el:.0f}s elapsed)",
              flush=True)
    open(os.path.join(PARTIAL_DIR, f"done_shard{shard_id:02d}"), "w").close()
    print(f"[shard {shard_id}] ALL DONE", flush=True)


def gpu_assignments():
    """worker 的 GPU 分配表（v2 每 GPU 两个 worker）."""
    import subprocess
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]).decode()
    gpus = [int(l.strip()) for l in out.strip().split("\n") if l.strip()]
    if IS_V2:
        return [g for g in gpus for _ in range(WORKERS_PER_GPU)]
    return gpus


def progress_k(split_name, shard):
    """某 shard 某 split 已完成的最大序列下标（-1 表示无）。"""
    ks = sorted(int(f.rsplit("_", 1)[1][:-4]) for f in glob.glob(
        os.path.join(PARTIAL_DIR, f"{split_name}_shard{shard:02d}_*.npz")))
    return ks[-1] if ks else -1


def driver():
    """监督进程：为每个 shard 维持一个活跃 worker，停滞则重启（断点续跑）。"""
    import subprocess
    import time as _t
    plans = gpu_assignments()
    n_shards = len(plans)
    log_dir = os.path.join(ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(PARTIAL_DIR, exist_ok=True)

    pids = {}
    while True:
        all_done = True
        for shard, gpu in enumerate(plans):
            tag = f"shard{shard:02d}"
            done_marker = os.path.join(PARTIAL_DIR, f"done_{tag}")
            if os.path.exists(done_marker):
                continue
            all_done = False
            p = pids.get(shard)
            if p is not None and p.poll() is None:
                continue                             # 还活着就不动它
            log = open(os.path.join(log_dir, f"{tag}.log"), "a")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            pids[shard] = subprocess.Popen(
                [sys.executable, __file__, "worker", str(shard), str(n_shards)],
                env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        if all_done:
            print("ALL COMPLETE", flush=True)
            break
        _t.sleep(60)


def merge_shards():
    """合并 partial 文件为最终 shard 文件（取每 shard 每个 split 最大 k）。"""
    n_shards = len(gpu_assignments())
    for split_name in ("train", "val", "test", "oldtok_test"):
        for shard in range(n_shards):
            prefix = os.path.join(PARTIAL_DIR, f"{split_name}_shard{shard:02d}")
            ks = sorted(int(f.rsplit("_", 1)[1][:-4])
                        for f in glob.glob(prefix + "_*.npz"))
            if not ks:
                continue
            d = np.load(f"{prefix}_{ks[-1]:05d}.npz")
            out = os.path.join(SHARD_DIR, f"{split_name}_shard{shard:02d}.npz")
            if os.path.exists(out):
                continue
            np.savez_compressed(
                out, states=d["states"], tokens=d["tokens"],
                offsets=d["offsets"])
            print(f"merged {out}")


if __name__ == "__main__":
    if sys.argv[1] == "driver":
        driver()
    elif sys.argv[1] == "worker":
        worker(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "merge":
        merge_shards()
