"""generate.py — 果蝇脑条件下的增量式对话生成。

生成协议：脑状态逐 token 增量延展（无需重放），
每生成一个 token 先把它推入全脑仿真、读取新状态，再由读出 GPT 采样。
"""
import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("KMP_AFFINITY", "disabled")
sys.path.insert(0, "/data/mkk/malecns/flychat")

from flychat.corpus import build_sequences, CharTokenizer
from flychat.interface import StimEncoder, BrainSession
from flychat.model import ReadoutGPT

ROOT = "/data/mkk/malecns/flychat"
_LANG = os.environ.get("FLYCHAT_LANG", "en")
SPLIT_PATH = os.path.join(ROOT, "data", "split.json" if _LANG == "en" else "split_cn.json")
N_OLF, N_VIS, N_DN = 2639, 4109, 1314
STATE_DIM = N_OLF + N_VIS + N_DN
W_MS = 16.0
TURN_SEP = "\n"


def generate_dialogue(sess, enc, model, prompt_ids, turn_sep_id,
                      max_new=64, temperature=0.85, top_k=12, device="cuda"):
    """给定 prompt（上下文若干轮），增量生成直到回合结束或达到上限。"""
    tokens = list(prompt_ids)
    sess.start_episode()
    states = []
    with torch.no_grad():
        for t in tokens:
            states.append(sess.push_token(t, enc, window_ms=W_MS))
        for _ in range(max_new):
            tk = torch.tensor([tokens], device=device)
            st = torch.stack(states).unsqueeze(0).to(device)
            logits = model(tk, st)                      # [1,T,vocab]
            logits = logits[0, -1] / temperature
            if top_k:
                kth = torch.topk(logits, top_k).values[-1]
                logits[logits < kth] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
            tokens.append(nxt)
            if nxt == turn_sep_id and len(tokens) - len(prompt_ids) > 4:
                break
            states.append(sess.push_token(nxt, enc, window_ms=W_MS))
    return tokens



def main(n_samples=14, ckpt="readout_brain"):
    tok, seqs = build_sequences()
    split = json.load(open(SPLIT_PATH))
    test_ids = split["splits"]["test"]

    enc = StimEncoder(len(tok.chars), 64, N_OLF, N_VIS, seed=0)
    enc.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder.pt")))
    cfg_p = os.path.join(ROOT, "checkpoints", "model_config.json")
    _d = 128
    _l, _h = 4, 4
    if os.path.exists(cfg_p):
        with open(cfg_p) as f:
            _mc = json.load(f)
        _d, _l, _h = _mc["d"], _mc["n_layers"], _mc["n_head"]
    model = ReadoutGPT(len(tok.chars), STATE_DIM, d=_d, n_layers=_l, n_head=_h)
    model.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", f"{ckpt}.pt")))
    sess = BrainSession(w_scale=0.275, device="cuda:0")
    enc = enc.to(sess.brain.device)
    model = model.to(sess.brain.device).eval()

    turn_sep_id = tok.stoi["\n"]

    rng = np.random.RandomState(1)
    samples = []
    for i in range(n_samples):
        seq = seqs[test_ids[i]]
        # prompt = 第一轮（到第一个 '\n' 为止）
        first_nl = seq.index(tok.stoi["\n"]) if tok.stoi["\n"] in seq else len(seq) - 1
        prompt = seq[:first_nl + 1]
        out = generate_dialogue(sess, enc, model, prompt, turn_sep_id)
        prompt_text = tok.decode(prompt).strip()
        gen_text = tok.decode(out[len(prompt):]).strip()
        samples.append({
            "prompt": prompt_text,
            "generated": gen_text,
            "full": tok.decode(out).strip(),
        })
        print(f"--- sample {i+1}/{n_samples} ---")
        print("PROMPT  :", repr(prompt_text))
        print("GEN     :", repr(gen_text), flush=True)
    with open(os.path.join(ROOT, "data", "samples.json"), "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
