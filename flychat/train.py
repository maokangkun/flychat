"""train.py — 离线训练读出 GPT（含基线对照）。

条件:
  brain   : 真实果蝇脑状态轨迹
  random  : 与真实状态逐维均值/方差匹配的 iid 随机向量（等参数量基线）
  shuffled: 真实状态在时间维上打乱（保留边缘分布，破坏 token 对齐）
  none    : 状态置零（纯 token LM，量化状态通道的贡献）
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, "/data/mkk/malecns/flychat")
from flychat.model import ReadoutGPT

ROOT = "/data/mkk/malecns/flychat"
# FLYCHAT_LANG: en=英文原版, cn=双语v1, v2=扩充双语v2
_LANG = os.environ.get("FLYCHAT_LANG", "en")
_DIRS = {
    "en":  ("shards",     "split.json"),
    "cn":  ("shards_cn",  "split_cn.json"),
    "v2":  ("shards_v2",  "split_v2.json"),
}
_SHARD_NAME, _SPLIT_NAME = _DIRS.get(_LANG, _DIRS["en"])
SHARD_DIR = os.path.join(ROOT, "data", _SHARD_NAME)
SPLIT_PATH = os.path.join(ROOT, "data", _SPLIT_NAME)
CKPT_DIR = os.path.join(ROOT, "checkpoints")

N_OLF, N_VIS, N_DN = 2639, 4109, 1314
STATE_DIM = N_OLF + N_VIS + N_DN
with open(SPLIT_PATH) as _f:
    VOCAB = len(json.load(_f)["vocab"])
# v2: d=384 / 6 层 / 6 头（约 30M 参数），经 FLYCHAT_* 环境变量注入
D_MODEL = int(os.environ.get("FLYCHAT_D", 128))
N_LAYERS = int(os.environ.get("FLYCHAT_LAYERS", 4))
N_HEAD = int(os.environ.get("FLYCHAT_HEADS", 4))


def save_model_config():
    """把读出 GPT 尺寸写入 checkpoints/model_config.json（server 部署时读取）."""
    cfg = {"d": D_MODEL, "n_layers": N_LAYERS, "n_head": N_HEAD,
           "vocab": VOCAB, "state_dim": STATE_DIM, "lang": _LANG}
    with open(os.path.join(CKPT_DIR, "model_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def n_params():
    m = ReadoutGPT(VOCAB, STATE_DIM, d=D_MODEL, n_layers=N_LAYERS, n_head=N_HEAD)
    return sum(p.numel() for p in m.parameters())


def load_split(split, n_shards):
    """返回 List[(tokens np.int64[T], states fp32[T,D])]"""
    seqs = []
    for shard in range(n_shards):
        path = os.path.join(SHARD_DIR, f"{split}_shard{shard:02d}.npz")
        d = np.load(path)
        toks, states, offs = d["tokens"], d["states"], d["offsets"]
        for i in range(len(offs) - 1):
            seqs.append((toks[offs[i]:offs[i + 1]].astype(np.int64),
                         states[offs[i]:offs[i + 1]].astype(np.float32)))
    return seqs


def batch_iter(data, batch_size, rng, pad_states=None, shuffle_time=False,
               device="cuda", indices=None):
    """按长度分桶产生 batch。indices 非空时按给定顺序（DDP 分片）。"""
    if indices is None:
        idx = np.arange(len(data))
        rng.shuffle(idx)
    else:
        idx = np.asarray(indices)
    for i in range(0, len(idx), batch_size):
        chunk = [data[j] for j in idx[i:i + batch_size]]
        chunk.sort(key=lambda x: -len(x[0]))
        T = max(len(t) for t, _ in chunk)
        B = len(chunk)
        toks = np.zeros((B, T), dtype=np.int64)
        stats = np.zeros((B, T, STATE_DIM), dtype=np.float32)
        lens = np.zeros(B, dtype=np.int64)
        for b, (t, s) in enumerate(chunk):
            toks[b, :len(t)] = t
            stats[b, :len(s)] = s
            lens[b] = len(t)
        # 有效位置掩码 [B,T]
        mask = np.arange(T)[None, :] < lens[:, None]
        yield (torch.from_numpy(toks).to(device),
               torch.from_numpy(stats).to(device),
               torch.from_numpy(mask).to(device))


def make_random_stats(data, seed=0):
    """与真实状态逐维均值/方差匹配的 iid 随机状态。"""
    all_states = np.concatenate([s for _, s in data], axis=0)
    mu = all_states.mean(axis=0, keepdims=True)
    sd = all_states.std(axis=0, keepdims=True) + 1e-6
    rng = np.random.RandomState(seed)
    out = []
    for t, s in data:
        noise = rng.standard_normal(s.shape).astype(np.float32)
        out.append((t, mu + noise * sd))
    return out


def make_shuffled_stats(data, seed=0):
    """真实状态在时间维打乱（保留逐步联合分布，破坏与 token 的对齐）。"""
    rng = np.random.RandomState(seed)
    out = []
    for t, s in data:
        perm = rng.permutation(len(s))
        out.append((t, s[perm]))
    return out


def make_zero_stats(data):
    return [(t, np.zeros_like(s)) for t, s in data]


@torch.no_grad()
def evaluate(model, data, device="cuda", indices=None):
    model.eval()
    tot_nll, tot_n = 0.0, 0
    for toks, stats, vmask in batch_iter(data, 32, np.random.RandomState(0),
                                         device=device, indices=indices):
        logits = model(toks[:, :-1], stats[:, :-1])
        tgt = toks[:, 1:]
        m = (vmask[:, 1:]).reshape(-1).float()
        nll = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1),
                              reduction="none")
        tot_nll += float((nll * m).sum())
        tot_n += float(m.sum())
    return tot_nll / tot_n


def train_condition(condition, train_data, val_data, epochs=15, seed=0,
                    device=None, lr=3e-4, batch_size=32, patience=None):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}"
    elif device is None:
        device = "cuda"
    torch.manual_seed(seed)
    raw_model = ReadoutGPT(VOCAB, STATE_DIM, d=D_MODEL, n_layers=N_LAYERS,
                           n_head=N_HEAD).to(device)
    model = raw_model
    if world > 1:
        model = DDP(raw_model, device_ids=[int(os.environ["LOCAL_RANK"])])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.RandomState(seed)
    curve = []
    t0 = time.time()
    best_val, best_state, since_best = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        ep_loss, ep_n = 0.0, 0
        ep_t = time.time()
        if world > 1:
            sampler = torch.utils.data.DistributedSampler(
                list(range(len(train_data))), num_replicas=world, rank=rank,
                shuffle=True, seed=seed)
            sampler.set_epoch(ep)
            indices = list(sampler)
        else:
            indices = None
        for toks, stats, vmask in batch_iter(train_data, batch_size, rng,
                                             device=device, indices=indices):
            logits = model(toks[:, :-1], stats[:, :-1])
            tgt = toks[:, 1:]
            mask = (vmask[:, 1:]).float()
            nll = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1),
                                  reduction="none")
            loss = (nll * mask.reshape(-1)).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss) * len(toks)
            ep_n += len(toks)
        ep_time = time.time() - ep_t
        # 各 rank 独立评估同一份 val（小数据，开销可忽略），决策完全一致
        val_nll = evaluate(model, val_data, device)
        curve.append({"epoch": ep, "train_nll": ep_loss / max(ep_n, 1),
                      "val_nll": val_nll,
                      "val_ppl": float(np.exp(val_nll)),
                      "epoch_time": ep_time,
                      "time": time.time() - t0})
        print(f"[{condition}] epoch {ep} train_nll={ep_loss/max(ep_n,1):.4f} "
              f"val_nll={val_nll:.4f} val_ppl={np.exp(val_nll):.2f} "
              f"epoch_time={ep_time:.1f}s", flush=True)
        if val_nll < best_val:
            best_val, since_best = val_nll, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in raw_model.state_dict().items()}
        else:
            since_best += 1
            if patience and since_best >= patience:
                print(f"[{condition}] early stop at epoch {ep} "
                      f"(best val_nll={best_val:.4f})", flush=True)
                break
    if best_state is not None:
        raw_model.load_state_dict(best_state)     # 回滚到 val 最优
    os.makedirs(CKPT_DIR, exist_ok=True)
    if rank == 0:
        torch.save(raw_model.state_dict(),
                   os.path.join(CKPT_DIR, f"readout_{condition}.pt"))
    return raw_model, curve


def run_one(cond, n_shards, epochs, out_suffix=""):
    if not out_suffix and _LANG == "cn":
        out_suffix = "_cn"
    if not out_suffix and _LANG == "v2":
        out_suffix = "_v2"
    train_data = load_split("train", n_shards)
    val_data = load_split("val", n_shards)
    test_data = load_split("test", n_shards)
    print(f"[{cond}] train {len(train_data)} val {len(val_data)} "
          f"test {len(test_data)} seqs")

    if cond == "brain":
        tr, va, te = train_data, val_data, test_data
    elif cond == "random":
        tr = make_random_stats(train_data, seed=0)
        va = make_random_stats(val_data, seed=0)
        te = make_random_stats(test_data, seed=0)
    elif cond == "shuffled":
        tr = make_shuffled_stats(train_data, seed=0)
        va = make_shuffled_stats(val_data, seed=0)
        te = make_shuffled_stats(test_data, seed=0)
    else:
        tr = make_zero_stats(train_data)
        va = make_zero_stats(val_data)
        te = make_zero_stats(test_data)

    lr = float(os.environ.get("FLYCHAT_LR", 3e-4))
    bs = int(os.environ.get("FLYCHAT_BS", 32))
    model, curve = train_condition(cond, tr, va, epochs=epochs, lr=lr,
                                   batch_size=bs,
                                   patience=int(os.environ.get("FLYCHAT_PATIENCE", 0)) or None)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        _dev = f"cuda:{os.environ.get('LOCAL_RANK', '0')}"
    else:
        _dev = "cuda"
    test_nll = evaluate(model, te, _dev)
    final = {"test_nll": test_nll,
             "test_ppl": float(np.exp(test_nll)),
             "best_val_ppl": min(r["val_ppl"] for r in curve),
             "n_params": n_params(),
             "mean_epoch_time": (sum(r.get("epoch_time", 0) for r in curve)
                                 / max(len(curve), 1))}
    print(f"[{cond}] test_ppl={final['test_ppl']:.2f}", flush=True)
    with open(os.path.join(ROOT, "data",
                           f"training_curve_{cond}{out_suffix}.json"), "w") as f:
        json.dump({"curve": curve, "final": final,
                   "model": {"d": D_MODEL, "n_layers": N_LAYERS,
                             "n_head": N_HEAD, "vocab": VOCAB}}, f, indent=2)


def _maybe_init_ddp():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")


if __name__ == "__main__":
    _maybe_init_ddp()
    rank = int(os.environ.get("RANK", "0"))
    if sys.argv[1] == "all":
        if rank == 0:
            os.makedirs(CKPT_DIR, exist_ok=True)
            save_model_config()
        for cond in ("brain", "random", "shuffled", "none"):
            run_one(cond, int(sys.argv[2]), int(sys.argv[3]))
    else:
        run_one(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    if dist.is_initialized():
        dist.destroy_process_group()
