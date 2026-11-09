"""gen_engine.py — v3 对话引擎：冻结果蝇脑仿真器 + LLM(LoRA) + 脑状态注入.

架构：
  用户消息 -> 冻结 StimEncoder 感觉电流 -> 全脑 LIF 仿真（4 ms/token 窗口，
  与训练状态采集相同窗口）-> 本轮状态向量 s_t（8062 维：嗅觉/视觉/DN）
  -> BrainPrefixProjector 把最近 4 轮状态映射为 4 个 soft prefix token
  -> 拼到 chat 序列 inputs_embeds 最前端 -> Qwen2.5-1.5B-Instruct(LoRA) 生成。
  viewer 活动快照与 v2 相同（逐 token），空闲时后台线程低频仿真驱动 3D 查看器。
"""
import os
os.environ.setdefault("KMP_AFFINITY", "disabled")
import json, sys, threading, time
import torch
import torch.nn as nn

ROOT = os.environ.get("FLYCHAT_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MAX_SLOTS = 4
STATE_DIM = 8062
W_MS = 4.0
SIM_MAX_TOKENS = 40
MAX_NEW_TOKENS = 96
HISTORY_MAX_MSGS = 20
# 推理期系统提示（SFT 未训练该 system 文本；用于语言跟随，见训练报告 v3 部署节）
SYS_HINT = ("Always reply in the SAME LANGUAGE as the user's message: "
            "English user gets English reply, Chinese user gets Chinese reply.")


def _cjk_ratio(s):
    s = s or ""
    if not s:
        return 0.0
    return sum(1 for c in s if "\u4e00" <= c <= "\u9fff") / len(s)


def _lang_mismatch(user, reply):
    return _cjk_ratio(user) < 0.2 and _cjk_ratio(reply) > 0.3


class BrainPrefixProjector(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(STATE_DIM, 2048), nn.GELU(),
            nn.Linear(2048, 2048), nn.GELU(),
            nn.Linear(2048, 2048))
        self.out = nn.Linear(2048, d_model, bias=False)


def load_models(base, adapter_dir, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.float32, attn_implementation="sdpa")
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    d_model = model.get_input_embeddings().weight.shape[1]
    pm = BrainPrefixProjector(d_model)
    bp = torch.load(os.path.join(adapter_dir, "brain_proj.pt"), map_location="cpu")
    pm.proj.load_state_dict(bp["proj"])
    pm.out.load_state_dict(bp["out"])
    model = model.to(device).eval()
    pm = pm.to(device).eval()
    return tok, model, pm


class FlyChatLLM:
    def __init__(self, base=None, adapter_dir=None, device="cuda",
                 activity_hook=None):
        self.base = base or f"{ROOT}/checkpoints/llm/Qwen2.5-1.5B-Instruct"
        self.adapter_dir = adapter_dir or f"{ROOT}/checkpoints/llm/final_v3"
        self.device = device
        self.activity_hook = activity_hook or (lambda active, n: None)

        from flychat.interface import BrainSession, StimEncoder
        from flychat.corpus import CharTokenizer
        self.sess = BrainSession(w_scale=0.275, device=device)
        with open(f"{ROOT}/data/split_v2.json") as f:
            chars = json.load(f)["vocab"]
        tok = CharTokenizer()
        tok.chars = chars
        tok.stoi = {c: i for i, c in enumerate(chars)}
        tok.itos = {i: c for i, c in enumerate(chars)}
        tok.vocab_size = len(chars)
        self.ctok = tok
        self.enc = StimEncoder(len(chars), 64, 2639, 4109, seed=0)
        ck = f"{ROOT}/checkpoints/encoder.pt"
        if os.path.exists(ck):
            sd = torch.load(ck, map_location="cpu")
            if sd["emb.weight"].shape[0] == len(chars):
                self.enc.load_state_dict(sd)
        self.enc = self.enc.to(device).eval()

        self.tok, self.llm, self.proj = load_models(
            self.base, self.adapter_dir, device)

        # RLock：server 外层与 chat() 内层会重复获取同一把锁
        self.lock = threading.RLock()
        self.messages = []          # [{role, content}]
        self.round_states = []      # 每轮（user 之后）的状态向量
        self._idle = threading.Event(); self._idle.set()
        self._viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
        self._viewer_thread.start()

    # ------------------------------------------------------------ brain
    def _record_activity(self, window_ms):
        counts = self.sess.brain.last_counts.float()
        nz = torch.nonzero(counts).flatten()
        if nz.numel() == 0:
            return
        rates = counts[nz] / (window_ms / 1000.0)
        order = torch.argsort(rates, descending=True)[:3000]
        self.activity_hook([[int(i), float(r)] for i, r in zip(
            nz[order].tolist(), rates[order].tolist())], int(nz.numel()))

    def simulate_user(self, text, window_ms=W_MS):
        ids = [self.ctok.stoi.get(c, self.ctok.stoi[" "])
               for c in text[:SIM_MAX_TOKENS]]
        toc = torch.tensor(ids, dtype=torch.long, device=self.device)
        vec_olf, vec_vis = self.enc(toc)
        self.sess.reset()
        state = None
        b = self.sess.brain
        for t in range(toc.numel()):
            self.sess._set_tonic(vec_olf[t], vec_vis[t])
            b.run(window_ms)
            state = self.sess._read_state(window_ms)
            self._record_activity(window_ms)
        if state is None:
            state = torch.zeros(STATE_DIM, device=self.device)
        return state.detach().float().cpu(), int(toc.numel())

    def _viewer_loop(self):
        """空闲时的低频自发活动（viewer 实时模式用）。"""
        while True:
            if self._idle.is_set():
                try:
                    self.sess.brain.run(W_MS)
                    self._record_activity(W_MS)
                except Exception:
                    pass
            time.sleep(0.6)

    # ------------------------------------------------------------ llm
    def _slot_states(self):
        """最近 MAX_SLOTS 轮的状态（不足补零）：最后一位 = 本轮。"""
        recent = self.round_states[-MAX_SLOTS:]
        out = torch.zeros(MAX_SLOTS, STATE_DIM, device=self.device)
        k = len(recent)
        if k:
            out[MAX_SLOTS - k:] = torch.stack(
                [s.to(self.device) for s in recent])
        return out

    def _generate(self, messages, slots_states, temperature=0.85):
        msgs = ([{"role": "system", "content": SYS_HINT}] +
                [{k: m[k] for k in ("role", "content")} for m in messages
                 if m["role"] != "system"])
        enc_ids = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True)
        ids = torch.tensor([enc_ids.input_ids], dtype=torch.long,
                           device=self.device)
        embed = self.llm.get_input_embeddings()(ids)
        pref = self.proj.out(self.proj.proj(slots_states)).to(embed.dtype)
        embeds = torch.cat([pref.unsqueeze(0), embed], dim=1)
        attn = torch.ones(1, embeds.shape[1], dtype=torch.long,
                          device=self.device)
        out = self.llm.generate(
            inputs_embeds=embeds, attention_mask=attn,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True, temperature=temperature, top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=self.tok.convert_tokens_to_ids("<|im_end|>"),
            pad_token_id=self.tok.eos_token_id)
        # transformers: 只传 inputs_embeds 时返回值可能不含 prompt 段
        if out.shape[1] >= embeds.shape[1]:
            toks = out[0][embeds.shape[1]:]
        else:
            toks = out[0]
        r = self.tok.decode(toks, skip_special_tokens=True)
        return r

    def chat(self, user_text):
        t0 = time.time()
        text = (user_text or "").strip()[:500]
        with self.lock:
            self._idle.clear()
            try:
                state, n_sim = self.simulate_user(text)
                self.messages.append({"role": "user", "content": text})
                self.round_states.append(state)
                slots = self._slot_states()          # 含本轮（已 append）
                msgs = self.messages[-HISTORY_MAX_MSGS:]
                # 注意：slots 应只对应"在做回复前已知"的最近 4 轮状态（含本轮）。
                reply = self._generate(msgs, slots)
                # 语言跟随守卫：回复语言与用户消息不一致时，置零 slots 重试一次
                # （真实脑状态 prefix 偶发把英文输入拉回中文，见报告 v3 评估节）
                if _lang_mismatch(text, reply):
                    zero = torch.zeros(MAX_SLOTS, STATE_DIM, device=self.device)
                    reply2 = self._generate(msgs, zero)
                    if not _lang_mismatch(text, reply2):
                        reply = reply2
                if not reply.strip():
                    self.messages.pop(); self.round_states.pop()
                    raise RuntimeError("empty reply")
                self.messages.append({"role": "assistant", "content": reply})
                wall = time.time() - t0
                return {"reply": reply,
                        "n_simulated_tokens": n_sim,
                        "n_prefix_slots": MAX_SLOTS,
                        "n_context_msgs": len(msgs),
                        "turns": len(self.round_states),
                        "wall_s": round(wall, 2)}
            finally:
                self._idle.set()

    def reset(self):
        with self.lock:
            self.messages = []
            self.round_states = []
            self.sess.reset()

    def status(self):
        return {"ok": True, "engine": "v3-llm",
                "n_neurons": int(self.sess.brain.n),
                "llm_base": os.path.basename(self.base),
                "adapter": os.path.basename(self.adapter_dir),
                "turns": len(self.round_states)}
