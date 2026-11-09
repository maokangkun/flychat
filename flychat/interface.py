"""interface.py — 果蝇脑对话接口：编码器 + 读出层 + episode 运行器。

设计（见 TRAINING_REPORT.md）：
  编码器  token -> 感觉神经元群电流模式
    可训练嵌入 E: (vocab, d_emb)，经固定随机投影 R_olf / R_vis 分配到
    嗅觉群（superclass=cb_sensory, class=olfactory, 2,639 神经元）与
    视觉群（superclass=ol_sensory, 4,109 神经元），
    幅值 = a_lo + (a_hi-a_lo)·sigmoid(R·e)。
  读出    果蝇脑状态 -> token
    每个 token 占一个 W ms 的仿真子窗，读出 [嗅觉率; 视觉率; 下行率]
    (2,639+4,109+1,314 = 8,062 维) 作为状态向量 S_t。
  训练抽象
    全图 LIF 比实时慢 ~10x，逐 token 反向传播不可行。采用"每序列一次
    有限时窗全脑仿真"：对每条对话跑一次全脑（每 token W ms），缓存状态
    轨迹 {S_t}，再离线训练读出 GPT。生成时脑状态可增量延展（逐 token
    继续 run），无需重放。
"""
import os
os.environ.setdefault("KMP_AFFINITY", "disabled")

import math
import sys
import torch
import torch.nn as nn

sys.path.insert(0, "/data/mkk/malecns/flybrain")
from flybrain import FlyBrain  # noqa: E402

# 状态读出维度分段
STATE_GROUPS = ("olfactory", "visual", "dn")


class StimEncoder(nn.Module):
    """token -> 感觉神经元群电流模式（可训练嵌入 + 固定群分配）。"""

    def __init__(self, vocab: int, d_emb: int, n_olf: int, n_vis: int,
                 seed: int = 0, a_lo: float = 5.0, a_hi: float = 16.0):
        super().__init__()
        self.d_emb = d_emb
        self.a_lo, self.a_hi = a_lo, a_hi
        self.emb = nn.Embedding(vocab, d_emb)
        nn.init.normal_(self.emb.weight, std=1.0)
        g = torch.Generator().manual_seed(seed)
        scale = 1.0 / math.sqrt(d_emb)
        # 固定（不可训练）分配矩阵：嵌入 -> 每个感觉神经元的电流幅值
        self.register_buffer("R_olf", torch.randn(n_olf, d_emb, generator=g) * scale)
        self.register_buffer("R_vis", torch.randn(n_vis, d_emb, generator=g) * scale)

    def forward(self, token_ids):
        """(T,) token ids -> (T,n_olf), (T,n_vis) 两个电流幅值张量。"""
        e = self.emb(token_ids)                      # [T, d]
        u_olf = torch.sigmoid(e @ self.R_olf.T)      # [T, n_olf]
        u_vis = torch.sigmoid(e @ self.R_vis.T)
        return (self.a_lo + (self.a_hi - self.a_lo) * u_olf,
                self.a_lo + (self.a_hi - self.a_lo) * u_vis)


class BrainSession:
    """包装 FlyBrain：episode 内逐 token 注入刺激、收集状态；支持增量生成。"""

    def __init__(self, w_scale=0.275, device=None):
        self.brain = FlyBrain(w_scale=w_scale, device=device, verbose=False)
        b = self.brain
        self.olf_idx = b.olfactory
        self.vis_idx = b.idx_where(superclass="ol_sensory")
        self.dn_idx = b.dn
        self.state_dim = len(self.olf_idx) + len(self.vis_idx) + len(self.dn_idx)

    def reset(self):
        self.brain.reset(keep_memory=True)
        self.brain.tonic = None

    def _read_state(self, window_ms):
        return torch.cat([
            self.brain.readout("olfactory", window_ms=window_ms),
            self.brain.readout(self.vis_idx, window_ms=window_ms),
            self.brain.readout(self.dn_idx, window_ms=window_ms)])

    def _set_tonic(self, vec_olf, vec_vis):
        b = self.brain
        if b.tonic is None:
            b.tonic = torch.zeros(b.n, dtype=torch.float32, device=b.device)
        else:
            b.tonic[self.olf_idx] = 0.0
            b.tonic[self.vis_idx] = 0.0
        b.tonic[self.olf_idx] = vec_olf
        b.tonic[self.vis_idx] = vec_vis

    def run_episode(self, tokens, encoder, window_ms=12.0):
        """teacher-forcing：整段序列一次全脑仿真，返回状态轨迹 [T, state_dim]。"""
        tokens = torch.as_tensor(tokens, dtype=torch.long,
                                 device=self.brain.device)
        vec_olf, vec_vis = encoder(tokens)
        self.reset()
        states = torch.empty(len(tokens), self.state_dim,
                             dtype=torch.float32, device=self.brain.device)
        with torch.no_grad():
            for t in range(len(tokens)):
                self._set_tonic(vec_olf[t], vec_vis[t])
                self.brain.run(window_ms)
                states[t] = self._read_state(window_ms)
        self.reset()
        return states

    # ---- 生成时的增量接口：脑状态保持，逐 token 延展 ----
    def start_episode(self):
        self.reset()

    def push_token(self, token_id, encoder, window_ms=12.0):
        """注入一个 token 的刺激并仿真其子窗，返回新状态向量 [state_dim]。"""
        vec_olf, vec_vis = encoder(torch.tensor(
            [token_id], dtype=torch.long, device=self.brain.device))
        self._set_tonic(vec_olf[0], vec_vis[0])
        self.brain.run(window_ms)
        return self._read_state(window_ms)
