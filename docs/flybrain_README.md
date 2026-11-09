# flybrain — MaleCNS v1.0 全图 LIF 仿真器（对话系统阶段一交付）

在成年雄性果蝇连接组 MaleCNS v1.0 的 traced-only 神经元子图
（**164,587 神经元 / 25,563,197 有向突触连接**）上运行的 GPU 脉冲神经网络
仿真器，带多巴胺门控的蘑菇体 KC→MBON 可塑性。动力学参数全部沿用
DOOMFLY（见 `DOOMFLY_NOTES.md`）。

- 代码：`flybrain.py`（仿真器）· `validate.py`（验证实验）· `make_figures.py`（图表）
- 环境：conda `malecns`（torch 2.13 + CUDA），运行 torch 前必须
  `os.environ["KMP_AFFINITY"] = "disabled"`（模块已内置）
- 数据：`/data/mkk/malecns/processed/`（`adjacency_minconf-0.5-traced-only_neurons.npz`
  + `node_features.parquet`，数据字典见该目录 `REPORT.md`）

## 1. 快速上手（对话接口层最小示例）

```python
import torch
from flybrain import FlyBrain

brain = FlyBrain()                        # ~2 s；首次加载连接组 ~30 s

# —— 1. 查询神经元 ——
mbon11 = brain.idx_where(type="MBON11")          # 任意特征查询
all_mbon = brain.idx_where(type="MBON~")         # '~' = 前缀匹配
kc_gamma = brain.idx_where(type="KCg~")          # γ 分支 Kenyon 细胞
dn = brain.resolve("dn")                         # 预定义组

# —— 2. 注入感觉刺激 ——
#    stim = {组: 电流 mV}，仅在本次 run 期间生效
counts = brain.run(1000.0, stim={"kc": 20.0})    # 刺激全部 KC 1 秒
#    或直接给下标：
brain.run(500.0, stim={mbon11: 15.0})

# —— 3. 读出状态 ——
rates = brain.readout("dn")     # 下行神经元放电率 (Hz)，最近的 run
counts = brain.last_counts     # 最近 run 的每神经元尖峰计数 (Tensor[164587])
w = brain.weights_summary()    # 可塑性权重变化摘要

# —— 4. 学习（多巴胺门控 KC→MBON 抑郁）——
brain.run(500.0, stim={"kc": 20.0, "dan": 20.0}, learning=True)

# —— 5. 重置 ——
brain.reset(keep_memory=True)   # 清动力学状态，保留学习后的权重
brain.reset(keep_memory=False)  # 完全恢复基线（忘掉所学）
```

## 2. API 参考

### `FlyBrain(data_dir=..., device=None)`

初始化时加载连接组、按 `node_features.parquet` 的 `consensus_nt` 给突触
定符号（ACh 兴奋，GABA/谷氨酸/组胺抑制），权重 = 突触计数 × 符号 × 0.275。
GPU 占用 ~700 MB。

### 组（神经元群）解析

| 方式 | 示例 | 说明 |
|---|---|---|
| `brain.idx_where(col=val)` | `idx_where(type="MBON11")` | 等值查询 |
| `brain.idx_where(col="X~")` | `idx_where(type="MBON~")` | 前缀查询 |
| `brain.resolve(name)` | `resolve("type:MBON~")` / `resolve("kc")` | 组名解析 |
| 预定义组 | `brain.kc / mbon / dan / dn / olfactory / sensory / orn` | LongTensor 下标 |

`idx_where` 支持的列：`type`、`class`、`superclass`、`instance`、
`somaSide`、`consensus_nt`、`status` 等（见 `node_features.parquet`）。

### `run(duration_ms, stim=None, learning=False) -> IntTensor[n]`

仿真 `duration_ms` 毫秒，返回每个神经元的尖峰计数。

- **`stim: dict`** — `{组: 电流 mV}`。组可以是预定义组名（`"kc"`）、
  `"列:值"` 查询串（`"type:ORN_DA1"`）或 LongTensor 下标。
  电流语义：mV 渐近电位偏移；神经元发放需 drive 超过
  `V_th - V_rest`（普通神经元 7 mV，KC 15 mV）。
- **`learning=True`** — 启用多巴胺门控可塑性（见 §3）。

速度参考（A100）：活动状态 ~10× 实时（1 s 仿真 ≈ 10 s）。

### 读出接口

- `brain.readout(group)` — 组内放电率 (Hz)。
- `brain.last_counts` — 全图尖峰计数；`brain.v` / `brain.g` — 膜电位/突触电导。
- `brain.weights_summary()` — 可塑权重相对基线的比例（mean/min/max/changed）。
- 下行神经元：`brain.dn`（1,314 个），对话接口的"动作/状态"读出建议从
  `readout("dn")` 或 `readout("type:DNa02~")` 等更细的类型组取。

### 状态管理

- `brain.reset(keep_memory=True)` — 清动力学（v/g/不应期/痕迹/时钟），
  **保留**学习到的突触权重。对话轮次之间用这个。
- `brain.reset(keep_memory=False)` — 权重也恢复基线。
- `brain.save_memory(path)` / `brain.load_memory(path)` — 学习权重的持久化
  （npz，带 bodyId 序列校验）。

## 3. 可塑性模型（核内门控乘性抑郁；注意：DOOMFLY v6 自身实际运行的是其速率规则，核内 LTD 被禁用——见 DOOMFLY_NOTES.md §4.3/§4.5）

只有 **KC→MBON 的兴奋性边**（61,210 条）可塑，其余全部冻结：

```
KC 发放   →  资格痕迹 elig += 1（τ = 1 s 指数衰减）
DAN 发放  →  w[edge] *= exp(-η · gain · elig[pre])
            （η = 0.001，floor = 0.1 × 基线）
```

`gain` 为 DAN→MBON 解剖接触分数（circuit.py 归一化方式）。语义：
气味（KC 模式）与多巴胺（奖惩）**时间上配对**时，活跃 KC 的输出突触
被定向抑制——果蝇蘑菇体嗅觉回避/趋近学习的第一步。DAN 刺激即可当作
"奖惩信号"通道：`brain.run(..., stim={"dan": 20.0}, learning=True)`。

## 4. 与 DOOMFLY 的差异（重要）

1. **GPU 全图向量化**替代 CPU 惰性事件结算，tick 调度语义一致。
2. 可塑回路从 KC→MBON11 推广到**全部** KC→MBON 边（38 种 MBON 类型）；
   DAN 从 2 个 PPL101 细胞推广到全部 392 个多巴胺能神经元。
3. 感觉刺激泛化为任意神经元群电流注入（DOOMFLY 只有视网膜/糖）。
4. 多巴胺/章鱼胺/血清素能神经元的突触不做快传递，只做调制（DOOMFLY v6 同）。
5. 未实现：DOOMFLY 的 KC→MBON11 之外的速率规则（rule.py）、视网膜分级电位模型。

## 5. 已知限制与网络动力学注意事项

**持续震荡吸引子（重要）**。在 traced-only 全图 + DOOMFLY 权重尺度
（0.275 mV/突触）下，网络是**超临界**的：任何足够强的脉冲刺激都会把网络
推入一个有界的、空间结构稳定的的高活动吸引子（主要由 `cb_intrinsic`
中枢神经元主导，群体率约 3.4 spikes/s，≥ 10 s 不衰减）。活动不会发散，
但也**不会自发回落到静息**。因此：

- 受控实验/对话轮次之间应调用 `brain.reset(keep_memory=True)`（清动力学、
  保留记忆），把每个 episode 从同一静息初态开始；
- 本文 C 组实验（联想条件反射）即采用该协议，配对/不配对对照不受污染；
- 若下游需要"无输入即安静"的网络，可用 `FlyBrain(w_scale=0.041)`
  （≈0.275×0.15，亚临界：有传播、撤刺激后归零），代价是 KC→MBON 等
  长程通路传导显著变弱。两档工作点的对比见验证图 Fig. 2。

**其他限制**：

- LIF 无噪声：无驱动时网络完全静默，无自发活动。
- 递质符号是粗粒度假设（无受体类型区分）；0.275 mV/突触未标定。
- 可塑性规则是 DOOMFLY 声明式候选模型，未拟合行为数据；
  配对窗口 ≳ 1 s（痕迹 τ）内 CS-US 才有效。
- DAN 门控强度随被激活的 DAN 数量线性增长（gain 列归一化）；
  全部 392 个 DAN 同时强刺激会很快把权重压到下限，建议用适中强度
  （US drive ≈ 14 mV）或更小的的 DAN 子集。
