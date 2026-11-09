"""
flybrain.py — MaleCNS v1.0 全图 GPU LIF 仿真器（DOOMFLY 风格）
================================================================
在 MaleCNS v1.0 traced-only 神经元子图（164,587 神经元 / 25.56M 有向边）上
实现果蝇脑脉冲动力学 + 多巴胺门控蘑菇体可塑性。

动力学完全沿用 DOOMFLY（出处详见 DOOMFLY_NOTES.md）：
  - dt=0.1ms, 静息-52mV, 膜常数20ms, 突触常数5ms, 阈值-45mV,
    不应期2.2ms, 轴突延迟1.8ms（doom/engine.py:15-46）
  - 权重 = 突触计数 × 符号(突触前递质) × 0.275（doom/prepare.py:23）
  - ACh 兴奋；GABA/谷氨酸/组胺抑制（doom/transmitters.py:10-18）
  - KC→MBON 多巴胺门控乘性抑郁（doom_learning_v6/kernel.cpp:70-79）

与 DOOMFLY 的主要差异（详见 DOOMFLY_NOTES.md 第 5 节）：
  - GPU 向量化：每 tick 全图更新，不做惰性事件结算（语义等价）
  - 可塑回路推广到全部 KC→MBON 边；DAN 取全部多巴胺能神经元
  - 刺激接口泛化为任意神经元群电流注入（DOOMFLY 仅视网膜/糖感受器）
  - 多巴胺/章鱼胺/血清素能神经元只做调制、不做快传递（DOOMFLY v6 同）
"""
import os
os.environ["KMP_AFFINITY"] = "disabled"  # 容器内 OpenMP 绑定限制（须在 torch 前）

import math
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

DEFAULT_DATA_DIR = os.environ.get(
    "FLYCHAT_PROCESSED_DIR",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    ),
)
VARIANT = "minconf-0.5-traced-only"

# ---------------- DOOMFLY 动力学常数（engine.py:15-46） ----------------
DT_MS = 0.1
V_REST = -52.0          # mV
V_TH = -45.0            # mV, 发放阈值
TAU_V_MS = 20.0         # 膜时间常数
TAU_G_MS = 5.0          # 突触时间常数
REFRACTORY_MS = 2.2     # 不应期
DELAY_MS = 1.8          # 轴突传输延迟
W_SCALE = 0.275         # 突触计数 → mV 全局尺度（prepare.py:23）
COUPLING = 1.0 / 3.0    # g 耦合系数（engine.py:16）

# ------------- DOOMFLY v6 可塑性常数（brain.py:14-16, rule.py:7-11） ------
KC_REST_MV = -60.0      # KC 静息电位单独设定
KC_ADAPT_JUMP_MV = 8.0
KC_ADAPT_TAU_MS = 200.0
TAU_ELIG_MS = 1000.0    # KC 资格痕迹（brain.py 传入 trace_kc_seconds=1.）
ETA = 0.001             # 学习率
FLOOR_FRACTION = 0.1    # 权重下限 = baseline × 0.1（rule.py:7）
MODULATORY_NT = ("dopamine", "octopamine", "serotonin")  # v6 brain.py:53


class FlyBrain:
    """MaleCNS v1.0 全图 LIF 仿真器（GPU）。

    用法（对话接口层）：
        brain = FlyBrain()                      # 加载连接组 + 特征表
        ix = brain.idx_where(type="MBON11")     # 任意类型查询 -> LongTensor
        brain.set_tonic("type:ORN_DA1", 10.0)   # 持续背景电流 (mV)
        counts = brain.run(1000.0,              # 仿真 1 s，返回尖峰计数
                           stim={"kc": 15.0},   # 本次 run 的瞬时刺激
                           learning=True)       # 启用 DAN 门控可塑性
        rates = brain.readout("dn")             # 下行神经元放电率 (Hz)
        brain.reset(keep_memory=True)           # 清动力学、保留学习权重
    """

    # ------------------------------------------------------------------ init
    def __init__(self, data_dir=DEFAULT_DATA_DIR, variant=VARIANT,
                 device=None, w_scale=W_SCALE, verbose=True):
        """w_scale: 每突触 mV 全局尺度。DOOMFLY 取 0.275（prepare.py:23）。

        注意：在 traced-only 全图上，0.275 会产生有界的持续震荡吸引子
        （刺激后活动不回落，见 README §5）。若需要静息安静的网络，
        可取 w_scale=0.041（≈0.275×0.15，亚临界，但 KC→MBON 传导变弱）。
        """
        self.w_scale = float(w_scale)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._log = print if verbose else (lambda *a, **k: None)

        # ---- 连接组（CSR）：行=突触前，A[i,j]=i→j 突触计数 ----
        A = sp.load_npz(f"{data_dir}/adjacency_{variant}_neurons.npz").tocsr()
        ids = pd.read_parquet(
            f"{data_dir}/node_ids_{variant}_neurons.parquet").sort_values("row_index")
        self.n = A.shape[0]
        self.body_id = ids.bodyId.to_numpy(np.int64)
        self.indptr = A.indptr.astype(np.int64)
        self.post_np = A.indices.astype(np.int32)
        counts_np = A.data.astype(np.float32)

        # ---- 节点特征表（与连接组行对齐） ----
        nf = pd.read_parquet(f"{data_dir}/node_features.parquet")
        self.features = nf.set_index("bodyId").loc[self.body_id]
        nt = self.features.consensus_nt.fillna("unclear").astype(str).str.lower()
        self.nt = nt.to_numpy()

        # ---- 递质符号（transmitters.py:10-18）：ACh 兴奋，其余 fast 抑制 ----
        is_ach = nt.str.contains("acetylcholine").to_numpy()
        is_inh = nt.str.contains("gaba|glutamate|histamine").to_numpy()
        sign = np.ones(self.n, dtype=np.float32)     # ambiguous/缺失 -> +1
        sign[is_ach & ~is_inh] = 1.0
        sign[is_inh & ~is_ach] = -1.0
        self.nt_sign = sign

        # ---- 权重（prepare.py:23）----
        pre_flat = np.repeat(np.arange(self.n), np.diff(self.indptr))
        self.w0 = (counts_np * sign[pre_flat] * self.w_scale).astype(np.float32)

        # ---- 神经元分组 ----
        self.kc = self._where(lambda f: f.type.fillna("").str.startswith("KC"))
        self.mbon = self._where(lambda f: f.type.fillna("").str.startswith("MBON"))
        self.dan = self._where(
            lambda f: f.consensus_nt.fillna("").astype(str).str.lower()
                        .eq("dopamine"))
        self.mod_mask = torch.tensor(
            np.isin(nt, MODULATORY_NT), dtype=torch.bool, device=self.device)
        self.dn = self._where(lambda f: f.superclass.fillna("").eq("descending_neuron"))
        self.olfactory = self._where(lambda f: f["class"].fillna("").eq("olfactory"))
        self.sensory = self._where(lambda f: f.superclass.fillna("").str.contains("sensory"))
        self.orn = self._where(lambda f: f.type.fillna("").str.startswith("ORN"))

        # ---- 可塑回路：全部 KC→MBON 兴奋性边（circuit.py:13-15 的推广）----
        kc_set = np.zeros(self.n, dtype=bool); kc_set[self.kc.cpu().numpy()] = True
        mbon_set = np.zeros(self.n, dtype=bool); mbon_set[self.mbon.cpu().numpy()] = True
        self.plastic_edges = np.flatnonzero(
            kc_set[pre_flat] & mbon_set[self.post_np] & (self.w0 > 0)
        ).astype(np.int64)
        self.kc_pre = pre_flat[self.plastic_edges].astype(np.int64)
        self._plastic_edges_t = torch.tensor(self.plastic_edges, device=self.device)
        self._kc_pre_t = torch.tensor(self.kc_pre, device=self.device)
        self._w0_t = torch.tensor(self.w0, device=self.device)

        # ---- DAN→MBON 接触分数作 gain（circuit.py:21-24）----
        dan_np = self.dan.cpu().numpy()
        mbon_np = self.mbon.cpu().numpy()
        gain = np.zeros((len(dan_np), len(mbon_np)), dtype=np.float32)
        if len(dan_np) and len(mbon_np):
            sub = A[dan_np][:, mbon_np].tocoo()      # DAN→MBON 接触计数
            colsum = np.asarray(sub.sum(axis=0)).ravel()
            colsum[colsum == 0] = 1.0
            gain[sub.row, sub.col] = sub.data / colsum[sub.col]
        mbon_pos = np.searchsorted(mbon_np, self.post_np[self.plastic_edges])
        self.dan_gain = torch.tensor(gain[:, mbon_pos], device=self.device)
        self.dan_index = torch.full((self.n,), -1, dtype=torch.long,
                                    device=self.device)
        self.dan_index[self.dan] = torch.arange(len(dan_np), device=self.device)

        # ---- GPU 图张量 ----
        self._pre_ptr = torch.tensor(self.indptr, device=self.device)
        self._deg = torch.diff(self._pre_ptr)
        self._post = torch.tensor(self.post_np, device=self.device)
        self.w = torch.tensor(self.w0, device=self.device)

        self.delay_ticks = round(DELAY_MS / DT_MS)
        self.refr_ticks = round(REFRACTORY_MS / DT_MS)
        self.av = math.exp(-DT_MS / TAU_V_MS)
        self.ag = math.exp(-DT_MS / TAU_G_MS)
        self.aa = math.exp(-DT_MS / KC_ADAPT_TAU_MS)
        self._adapt_k = KC_ADAPT_TAU_MS / (KC_ADAPT_TAU_MS - TAU_V_MS)
        self.elig_decay = math.exp(-DT_MS / TAU_ELIG_MS)

        self.tonic = None
        self.reset(keep_memory=False)
        self._log(f"FlyBrain: {self.n} neurons, {len(self.w):,} edges, "
                  f"{len(self.plastic_edges):,} plastic KC→MBON edges, "
                  f"{len(dan_np)} DANs, device={self.device}")

    # ---------------------------------------------------------- 分组/查询
    def _where(self, fn) -> torch.LongTensor:
        sel = np.flatnonzero(fn(self.features).to_numpy())
        return torch.tensor(sel, dtype=torch.long, device=self.device)

    def idx_where(self, **kwargs) -> torch.LongTensor:
        """按 node_features 列查询神经元下标。

        支持 'type'、'class'、'superclass'、'instance'、'consensus_nt' 等；
        值以 '~' 结尾表示前缀匹配（如 type='MBON~' 匹配所有 MBON）。
        """
        f = self.features
        mask = np.ones(len(f), dtype=bool)
        for col, val in kwargs.items():
            if isinstance(val, str) and val.endswith("~"):
                mask &= f[col].fillna("").astype(str).str.startswith(val[:-1]).to_numpy()
            else:
                mask &= f[col].fillna("").astype(str).eq(str(val)).to_numpy()
        return torch.tensor(np.flatnonzero(mask), dtype=torch.long,
                            device=self.device)

    def resolve(self, name: str) -> torch.LongTensor:
        """组名 -> 下标。'type:X' / 'type:X~'（前缀）/ 预定义组名。"""
        if ":" in name:
            col, val = name.split(":", 1)
            return self.idx_where(**{col: val})
        if hasattr(self, name):
            return getattr(self, name)
        raise KeyError(f"未知组名: {name!r}（可用 'type:…' 查询或预定义组）")

    def _resolve_group(self, group):
        if isinstance(group, torch.Tensor):
            return group.to(self.device).long()
        if isinstance(group, str):
            return self.resolve(group)
        return torch.tensor(np.asarray(group, dtype=np.int64),
                            dtype=torch.long, device=self.device)

    # ---------------------------------------------------------- 状态管理
    def reset(self, keep_memory=True):
        """重置动力学。keep_memory=True 保留学习后的权重，False 恢复基线。"""
        dev, n = self.device, self.n
        self.rest = torch.full((n,), V_REST, dtype=torch.float32, device=dev)
        self.rest[self.kc] = KC_REST_MV
        self.v = torch.full((n,), V_REST, dtype=torch.float32, device=dev)
        self.g = torch.zeros(n, dtype=torch.float32, device=dev)
        self.refractory = torch.zeros(n, dtype=torch.int16, device=dev)
        self.adapt = torch.zeros(n, dtype=torch.float32, device=dev)
        self.elig = torch.zeros(n, dtype=torch.float32, device=dev)
        slots = self.delay_ticks + 1
        self.ring = torch.zeros((slots, n), dtype=torch.bool, device=dev)
        self.cursor = 0
        self.sim_ms = 0.0
        self.total_spikes = 0
        self.last_counts = torch.zeros(n, dtype=torch.int32, device=dev)
        self.last_window_ms = DT_MS
        if not keep_memory:
            self.w.copy_(self._w0_t)

    def set_tonic(self, group, current: float):
        """设置持续背景电流（mV，叠加到之后所有 run 的 drive 上）。"""
        if self.tonic is None:
            self.tonic = torch.zeros(self.n, dtype=torch.float32,
                                     device=self.device)
        self.tonic[self._resolve_group(group)] = current

    # ---------------------------------------------------------- 主循环
    def run(self, duration_ms, stim=None, learning=False):
        """仿真一段时间并返回每个神经元的尖峰计数 (IntTensor, on device)。

        duration_ms : 仿真时长（ms），四舍五入到 0.1 ms 步数
        stim        : {group: amplitude_mV}，本 run 期间的刺激电流。
                      group 为组名（'type:ORN_DA1'、'kc'…）或下标 LongTensor。
        learning    : 启用 DAN 门控 KC→MBON 可塑性
        """
        steps = int(round(duration_ms / DT_MS))
        if steps < 1:
            raise ValueError("duration 至少 1 个时间步 (0.1 ms)")
        n = self.n
        drive = torch.zeros(n, dtype=torch.float32, device=self.device)
        if self.tonic is not None:
            drive += self.tonic
        if stim:
            for group, amp in stim.items():
                ix = self._resolve_group(group)
                drive.scatter_add_(0, ix,
                                   torch.full((len(ix),), float(amp),
                                              device=self.device))

        v, g, refr, adapt, elig = (self.v, self.g, self.refractory,
                                   self.adapt, self.elig)
        counts = torch.zeros(n, dtype=torch.int32, device=self.device)
        pre_ptr, post, deg, w = self._pre_ptr, self._post, self._deg, self.w
        mod_mask = self.mod_mask
        av, ag, aa, ak = self.av, self.ag, self.aa, self._adapt_k
        ring, slots = self.ring, self.ring.shape[0]
        elig_decay = self.elig_decay
        kc_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        kc_mask[self.kc] = True

        t0 = time.perf_counter()
        for _ in range(steps):
            # ---- 1. 积分 + 阈值（engine.py:26-32）----
            active = refr == 0
            refr = torch.clamp(refr - 1, min=0)
            # 活跃神经元积分；不应期神经元 v/g 冻结（engine.py:22-24）
            v_new = (self.rest + (v - self.rest) * av
                     + drive * (1 - av) + g * (av - ag) * COUPLING)
            g_new = g * ag
            v = torch.where(active, v_new, v)
            g = torch.where(active, g_new, g)
            # KC 适应（v6 kernel.cpp:31-33）
            v = v - adapt * ak * (aa - av)
            adapt = adapt * aa
            spike = active & (v > V_TH)
            if spike.any():
                counts += spike.to(torch.int32)
                elig *= elig_decay
                if learning:
                    elig[spike & kc_mask] += 1.0     # KC 资格痕迹（v6:38-40）
                adapt[spike & kc_mask] += KC_ADAPT_JUMP_MV
                ring[(self.cursor + self.delay_ticks) % slots] = spike

                # ---- DAN 门控乘性抑郁（v6 kernel.cpp:70-79）----
                if learning:
                    dan_spiking = spike & (self.dan_index >= 0)
                    if dan_spiking.any():
                        di = self.dan_index[dan_spiking]
                        gain_vec = self.dan_gain[di].sum(dim=0)
                        new_w = (w[self._plastic_edges_t]
                                 * torch.exp(-ETA * gain_vec * elig[self._kc_pre_t]))
                        floor = self._w0_t[self._plastic_edges_t] * FLOOR_FRACTION
                        w[self._plastic_edges_t] = torch.clamp(new_w, min=floor)

            # ---- 2. 突触传递（engine.py:36-44；调制能神经元除外, v6:62-70）----
            slot = self.cursor % slots
            inflow = ring[slot]
            if inflow.any():
                src = torch.nonzero(inflow).squeeze(1)
                src = src[~mod_mask[src]]
                if len(src) > 0:
                    lens = deg[src]
                    total = int(lens.sum())
                    if total > 0:
                        starts = pre_ptr[src]
                        offs = (torch.arange(total, device=self.device)
                                - torch.repeat_interleave(
                                    torch.cumsum(lens, 0) - lens, lens))
                        idx = torch.repeat_interleave(starts, lens) + offs
                        tgt = post[idx]
                        m = refr[tgt] == 0
                        g.index_add_(0, tgt[m], w[idx][m])
                ring[slot].fill_(False)

            # ---- 3. 重置刚发放的神经元（engine.py:44-46）----
            v[spike] = self.rest[0]
            g[spike] = 0.0
            refr[spike] = self.refr_ticks
            self.cursor += 1

        self.v, self.g, self.refractory = v, g, refr
        self.adapt, self.elig = adapt, elig
        self.sim_ms += steps * DT_MS
        self.last_counts = counts
        self.last_window_ms = steps * DT_MS
        self.total_spikes += int(counts.sum().item())
        self.last_wall_s = time.perf_counter() - t0
        return counts

    # ---------------------------------------------------------- 读出
    def readout(self, group="dn", counts=None, window_ms=None):
        """读取某组神经元的放电率 (Hz)。默认对最近一次 run 的计数。"""
        ix = self._resolve_group(group)
        c = counts if counts is not None else self.last_counts
        window = window_ms if window_ms is not None else self.last_window_ms
        return c[ix].float() / (window / 1000.0)

    def weights_summary(self) -> dict:
        """可塑性权重相对基线的变化摘要。"""
        w = self.w[self._plastic_edges_t].cpu().numpy()
        b = self.w0[self.plastic_edges]
        frac = w / b
        return dict(plastic_edges=len(w),
                    changed=int((w != b).sum()),
                    mean_frac=float(frac.mean()),
                    min_frac=float(frac.min()),
                    max_frac=float(frac.max()))

    # ---------------------------------------------------------- 记忆存取
    def save_memory(self, path):
        """把当前（可能已被可塑性改变的）全部边权保存到 npz。"""
        np.savez(path, w=self.w.cpu().numpy(), body_id=self.body_id,
                 plastic_edges=self.plastic_edges)

    def load_memory(self, path):
        """恢复 save_memory 保存的权重（校验节点序）。"""
        d = np.load(path)
        if not np.array_equal(d["body_id"], self.body_id):
            raise ValueError("body_id 序列不匹配，拒绝加载")
        self.w.copy_(torch.tensor(d["w"], device=self.device))
