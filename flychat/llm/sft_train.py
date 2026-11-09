"""sft_train.py — Qwen2.5-1.5B-Instruct LoRA SFT + 脑状态 prefix 投影（联合）.

设计（v3）：
  - base 冻结（fp32 master + bf16 autocast），LoRA（q/k/v/o/gate/up/down）
  - 调制通道：每条对话最多 4 个 user turn -> 4 个 prefix slot。
    slot_j = Proj(state_j)（有真实脑状态时）或 0 向量（无状态/置零 dropout），
    4 个 slot 的 embedding 拼到 chat 序列 inputs_embeds 最前端。
  - phase1：全部样本置零（学无条件对话）；phase2：真实脑状态子集
    （约 6k 段 LCCC）+ 随机置零，联合训练 LoRA + Proj。
  - DDP：torchrun --nproc_per_node=N（默认用空闲 GPU 1/4/6）。
"""
import os
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
os.environ["MKL_NUM_THREADS"] = "1"
import argparse, json, math, random, sys, time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, "/data/mkk/malecns/flychat")
STATE_DIM = 2639 + 4109 + 1314          # 8062
MAX_SLOTS = 4
ROOT = "/data/mkk/malecns/flychat"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=f"{ROOT}/checkpoints/llm/Qwen2.5-1.5B-Instruct")
    p.add_argument("--out", default=f"{ROOT}/checkpoints/llm")
    p.add_argument("--jsonl", default=f"{ROOT}/data/sft/lccc_sft_50k.jsonl")
    p.add_argument("--phase", default="phase1", choices=["phase1", "phase2"])
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lorar", type=int, default=16)
    p.add_argument("--init_adapter", default="")     # phase2: phase1 adapter 目录
    p.add_argument("--tag", default="v3phase1")
    return p.parse_args()


ASST_HEADER = None    # 惰性初始化："<|im_start|>assistant\n" 的 token 序列


def assistant_spans(ids, tok):
    """模板忠实掩码：每个 '<|im_start|>assistant\\n' 头之后到 '<|im_end|>' 的
    token（含 <|im_end|>）作为监督目标。返回与 ids 等长 bool 列表。"""
    global ASST_HEADER
    if ASST_HEADER is None:
        ASST_HEADER = tok.encode("<|im_start|>assistant\n",
                                 add_special_tokens=False)
    h = ASST_HEADER
    mask = [False] * len(ids)
    i = 0
    while i <= len(ids) - len(h):
        if ids[i:i + len(h)] == h:
            j = i + len(h)
            while j < len(ids):
                mask[j] = True
                if ids[j] == tok.convert_tokens_to_ids("<|im_end|>"):
                    break
                j += 1
            i = j + 1
        else:
            i += 1
    return mask


class SFTData(torch.utils.data.Dataset):
    def __init__(self, rows, tok, max_len=512):
        self.rows, self.tok, self.max_len = rows, tok, max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        out = self.tok.apply_chat_template(r["messages"], tokenize=True)
        ids = list(out.input_ids)[:self.max_len]
        mask = assistant_spans(ids, self.tok)
        ids_t = torch.tensor(ids, dtype=torch.long)
        m_t = torch.tensor(mask, dtype=torch.bool)
        labels = ids_t.masked_fill(~m_t, -100)
        st = r.get("brain_states")
        st = torch.tensor(st, dtype=torch.float32) if st else None
        return ids_t, labels, st


def collate(batch, zero_p=0.0):
    B = len(batch)
    L = max(ids.shape[0] for ids, _, _ in batch)
    input_ids = torch.zeros(B, L + MAX_SLOTS, dtype=torch.long)
    attent = torch.zeros(B, L + MAX_SLOTS, dtype=torch.long)
    labels = torch.full((B, L + MAX_SLOTS), -100, dtype=torch.long)
    slot_states = torch.zeros(B, MAX_SLOTS, STATE_DIM)
    slot_real = torch.zeros(B, MAX_SLOTS, dtype=torch.float32)
    n_user = []
    for b, (ids, lab, st) in enumerate(batch):
        input_ids[b, MAX_SLOTS:MAX_SLOTS + ids.shape[0]] = ids
        labels[b, MAX_SLOTS:MAX_SLOTS + lab.shape[0]] = lab
        attent[b, MAX_SLOTS:MAX_SLOTS + ids.shape[0]] = 1
        attent[b, :MAX_SLOTS] = 1
        if st is not None and st.numel():
            k = min(MAX_SLOTS, st.shape[0])
            keep = torch.ones(k)
            if zero_p > 0:
                g = torch.Generator().manual_seed(random.randrange(1 << 30))
                keep = (torch.rand(k, generator=g) > zero_p).float()
            slot_states[b, :k] = st[:k] * keep.unsqueeze(-1)
            slot_real[b, :k] = 1.0
    return {"input_ids": input_ids, "attention_mask": attent,
            "labels": labels, "slot_states": slot_states,
            "slot_real": slot_real}


def load_rows(args, rank=0, world=1):
    rows = []
    with open(args.jsonl) as f:
        for line in f:
            rows.append(json.loads(line))
    if args.phase == "phase2":
        # 真实脑状态子集 + phase1 数据回放（防遗忘），回放行全部零状态
        sel, mapping = [], {}
        for i, r in enumerate(rows):
            mapping[r["id"]] = i
        import glob
        for path in sorted(glob.glob(f"{ROOT}/data/sft/brainstates_*.npz")):
            d = np.load(path, allow_pickle=True)
            off = 0
            for k, (did, nt) in enumerate(zip(d["ids"], d["n_turns"])):
                st = d["states"][off:off + int(nt)]
                off += int(nt)
                if int(did) in mapping:
                    r = dict(rows[mapping[int(did)]])
                    r["brain_states"] = st.astype(np.float32).tolist()
                    sel.append(r)
        n_replay = min(len(rows), 2 * max(1, len(sel)))
        replay = random.Random(0).sample(rows, n_replay)
        rows = sel + replay                       # 状态绑定行 + 回放行
    random.Random(0).shuffle(rows)
    return rows


class PrefModel(nn.Module):
    """base(LoRA) + 脑状态 slot 投影。inputs_embeds = [slots] + embed(ids占位)."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.proj = nn.Sequential(
            nn.Linear(STATE_DIM, 2048), nn.GELU(),
            nn.Linear(2048, 2048), nn.GELU(),
            nn.Linear(2048, 2048))
        e = model.get_input_embeddings()
        self.out = nn.Linear(2048, e.weight.shape[1], bias=False)
        with torch.no_grad():
            nn.init.normal_(self.out.weight, std=0.02 / math.sqrt(2048))

    def forward(self, input_ids, attention_mask=None, labels=None,
                slot_states=None, slot_real=None):
        B, T = input_ids.shape
        embeds = self.model.get_input_embeddings()(input_ids)
        z = torch.where((slot_real > 0).unsqueeze(-1), slot_states,
                        torch.zeros_like(slot_states))
        slots = self.out(self.proj(z))                # [B, S, d] (0/真实混合)
        embeds = torch.cat([slots.to(embeds.dtype), embeds], dim=1)
        amend = torch.ones(B, MAX_SLOTS, device=embeds.device)
        am = attention_mask if attention_mask is not None else None
        if am is not None:
            am = torch.cat([amend, am], dim=1)
        lab = torch.cat([torch.full((B, MAX_SLOTS), -100, device=labels.device,
                                    dtype=torch.long), labels], dim=1) \
            if labels is not None else None
        return self.model(inputs_embeds=embeds, attention_mask=am, labels=lab)


def main():
    args = parse()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.distributed.init_process_group("nccl")
    torch.manual_seed(0)
    dev = f"cuda:{local}" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.base)
    assert tok.chat_template is not None
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float32, attn_implementation="sdpa")
    if args.phase == "phase1":
        lcfg = LoraConfig(r=args.lorar, lora_alpha=args.lorar * 2, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj",
                                          "o_proj", "gate_proj",
                                          "up_proj", "down_proj"],
                          task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg)
    else:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter,
                                          is_trainable=True)
    pm = PrefModel(model).to(dev)
    pm.model.print_trainable_parameters()

    rows = load_rows(args)
    if rank == 0:
        print(f"[data] phase={args.phase} rows={len(rows)}", flush=True)
    ds = SFTData(rows, tok)
    loader = None
    if world > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds, num_replicas=world, rank=rank, shuffle=True, seed=0)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch, sampler=sampler, num_workers=0,
            collate_fn=collate, pin_memory=True, drop_last=True)
    else:
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch, num_workers=0, shuffle=True,
            collate_fn=collate, pin_memory=True, drop_last=True)

    freeze = [p for n, p in pm.named_parameters()
              if not (p.requires_grad or n.startswith("proj") or n.startswith("out"))]
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))

    steps_per_epoch = len(loader) // args.accum
    total_steps = int(steps_per_epoch * args.epochs)
    warmup = max(10, total_steps // 20)
    def lr_at(s):
        if s < warmup: return args.lr * s / warmup
        t = (s - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.1 + 0.9 * args.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    for g in opt.param_groups: g["lr"] = 0.0

    pm.train()
    step, t0 = 0, time.time()
    tokens_done = 0
    loss_acc, loss_n = 0.0, 0
    done = False
    for epoch in range(math.ceil(args.epochs)):
        if world > 1: sampler.set_epoch(epoch)
        for it, batch in enumerate(loader):
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = pm(**batch)
                loss = out.loss / args.accum
            loss.backward()
            loss_acc += float(out.loss.detach()); loss_n += 1
            tokens_done += int((batch["labels"] != -100).sum())
            if (it + 1) % args.accum == 0:
                for g in opt.param_groups: g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 10 == 0 and rank == 0:
                    el = time.time() - t0
                    print(f"step {step}/{total_steps} loss={loss_acc/max(1,loss_n):.4f} "
                          f"tok={tokens_done} tok/s={tokens_done/max(1,el):.0f} "
                          f"lr={lr_at(step):.2e} mem={torch.cuda.max_memory_allocated()/2**30:.1f}GB",
                          flush=True)
                    loss_acc, loss_n = 0.0, 0
                if step >= total_steps:
                    done = True
            if done: break
        if done: break

    if rank == 0:
        out_dir = os.path.join(args.out, args.tag)
        os.makedirs(out_dir, exist_ok=True)
        pm.model.save_pretrained(out_dir)      # adapter
        torch.save({"proj": pm.proj.state_dict(), "out": pm.out.state_dict(),
                    "phase": args.phase}, os.path.join(out_dir, "brain_proj.pt"))
        with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
            json.dump({"base": args.base, "phase": args.phase,
                       "steps": step, "tokens": tokens_done,
                       "epochs": args.epochs, "lr": args.lr,
                       "r": args.lorar, "rows": len(rows),
                       "jsonl": args.jsonl}, f, ensure_ascii=False, indent=1)
        print(f"[done] adapter -> {out_dir} steps={step} tok={tokens_done} "
              f"({time.time()-t0:.0f}s)", flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
