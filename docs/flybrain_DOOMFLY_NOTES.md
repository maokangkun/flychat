# DOOMFLY 核心实现精读笔记（供全图 LIF 仿真器复用）

> 仓库：`/data/mkk/malecns/doomfly/`（MaleCNS v1.0 雄性果蝇连接组）
> 所有行号以 2026-10-03 检出版本为准。

## 1. 连接组导入（doom/connectome.py）

- **节点保留**：`normalize_nodes()`（connectome.py:71-104）只保留
  `superclass` 非空的神经元；`status == "Glia"` 的 body 显式剔除
  （connectome.py:82-83）。
- **ID 纪律**：`exact_ids()`（connectome.py:16-27）禁止 64 位 bodyId 经过
  浮点（防精度损失），全部用 `uint64`。
- **边索引**：`index_edges()`（connectome.py:30-42）用
  `np.searchsorted` 把 `body_pre/body_post` 映射到行下标，逐行核算被
  排除的边（"loss-accounted"），不做任何静默丢弃。
- **权重语义**：边权 = 突触接触计数（synapse count），不是估计的
  突触强度（connectome.py:172 `"synaptic_weights_are_contact_counts": True`）。

## 2. LIF 动力学参数（doom/engine.py + doom/kernel.cpp）

| 参数 | 值 | 出处 |
|---|---|---|
| 时间步长 dt | 0.1 ms（强制，engine.py:49） | engine.py:49 |
| 静息电位 V_rest | -52 mV | engine.py:26, 67 |
| 膜时间常数 τ_v | 20 ms | engine.py:15（`exp(-dt/20)`） |
| 突触时间常数 τ_g | 5 ms | engine.py:15（`exp(-dt/5)`） |
| 突触耦合系数 | g·(a_v − a_g)/3，即 1/3 耦合到膜 | engine.py:15-16 |
| 阈值 V_th | -45 mV（v > -45 即发放） | engine.py:28 |
| 重置 | v ← -52，g ← 0 | engine.py:46 |
| 不应期 | 2.2 ms（22 个 tick） | engine.py:46 |
| 轴突/传输延迟 | 1.8 ms（18 个 tick，环形延迟队列） | engine.py:30, 69 |
| 调度次序 | 先积分/阈值 → 到期尖峰投递 → 对刚发放神经元重置 | engine.py:31-46 |

**解析亚阈值积分**（engine.py:26，kernel.cpp:21）：

```
v(t+dt) = V_rest + (v - V_rest)·a_v + drive·(1 - a_v) + g·(a_v - a_g)/3
g(t+dt) = g · a_g
其中 a_v = exp(-dt/τ_v), a_g = exp(-dt/τ_g)
```

- `drive` 是外加电流（mV 渐近值量纲），外部刺激全部以 `drive` 注入。
- "只有入流电流能驱动神经元"（engine.py 文件头 docstring）。
- kernel.cpp 是惰性精确演化：静默神经元只在有事件/读出边界时结算
  （kernel.cpp:1-3, 60-62），活跃判定用凸松弛界
  `v > -45 || drive > 7 || drive + g > 7`（kernel.cpp:37）。**这是 CPU 优化，
  GPU 版每步对全图做向量化更新即可，语义等价。**

### 感觉驱动（engine.py:82-93）

- 视网膜：`drive = 30 · lum / (0.02 + lum)`，亮度先经 10 ms 一阶低通
  （engine.py:86-88, 92）。
- 薄板层（lamina）tonic 偏置：默认 12.0（engine.py:91）。
- 糖感受器：恒定 30（engine.py:93）。

## 3. 神经递质符号处理（doom/transmitters.py:5-21 + doom/prepare.py:22-23）

**符号规则**（transmitters.py:10-18）：

| 递质 | 符号 |
|---|---|
| acetylcholine（乙酰胆碱） | +1（兴奋） |
| gaba（GABA） | -1（抑制） |
| glutamate（谷氨酸） | -1（抑制）※果蝇谷氨酸门控氯通道 |
| histamine（组胺） | -1（抑制） |
| 冲突/缺失/modulator-only | 参数 `ambiguous_sign`，默认 +1（不删边） |

多递质：只有一个 fast 符号时取该符号；互相冲突时用 ambiguous_sign。

**权重合成**（prepare.py:23）：

```python
weight = synapse_count × sign(pre) × 0.275
```

- 符号取**突触前**神经元的递质；0.275 mV/突触是全局尺度因子
  （未标定，declared choice）。
- 源码明确声明这"NOT receptor physiology"——是粗粒度 fast-transmission 假设。

## 4. 多巴胺门控 KC→MBON 可塑性（doom_learning_v6/）

### 4.1 回路识别（doom_learning/circuit.py:8-19）

- KC：`type.str.startswith('KC')`；
- MBON：`type == 'MBON11'`（雌雄各 1 对，恰好 2 个）；
- DAN（多巴胺能神经元）：`type == 'PPL101'`（2 个）；
- 可塑边：**已存在的** KC→MBON11 全部正向边（circuit.py:13-15，
  "All existing KC-to-MBON11 edges; no graph pruning or added connections"）；
- DAN 增益 `gain`：PPL101→MBON11 直接接触权重按 MBON 列归一化的分数
  （circuit.py:21-24）——解剖学 proxy，不是测量到的多巴胺释放。

### 4.2 门控抑郁规则（doom_learning_v6/kernel.cpp:37-79）

每次 DAN 尖峰（`dan_index[i] >= 0` 且 learning on）：

```python
for p in plastic_edges:                    # v6 kernel.cpp:70-79
    trace  = eligibility[pre[p]] * exp(-dt·Δt / tau_elig)   # KC 适应性痕迹
    gain   = dan_gain[dan, p]
    weight[edge[p]] = weight[edge[p]] * exp(-eta * gain * trace)
    weight[edge[p]] = max(weight[edge[p]], baseline * floor_fraction)
```

- **KC 资格痕迹**（eligibility trace）：KC 每次发放 `eligibility += 1`，
  以 τ_elig = 1 s 指数衰减（v6 kernel.cpp:37-40；brain.py:14 传入
  `trace_kc_seconds=1.`）。
- **DAN 调制痕迹**：DAN 尖峰不参与快传递，而是向其全部突触后目标
  叠加 `|w|/0.275` 的调制痕迹，100 ms 衰减
  （v6 kernel.cpp:63-70；brain.py:14 `modulator_delivery_trace_ms=100.`）。
- **权重界**：floor = baseline × `minimum_fraction = 0.1`
  （rule.py:7）；纯乘性抑郁、保号。
- **学习率** η 默认 0.001（brain.py:35）。
- 语义：**KC 活动 + 多巴胺同时出现 → 该 KC→MBON 突触定向抑制**——
  简化 Odor-shock 条件反射（负价）模型。

### 4.3 速率规则（doom_learning_v6/rule.py:5-35）——v6 实际生效的规则

**注意：v6 中这是实际生效的学习规则**——brain.py 调用
`_neural_step(..., learning=False)`（brain.py:100-115 附近注释：
"The original LTD update is disabled. Only the centered rule below
writes candidate memory efficacies"），核内 LTD 被禁用，权重由本规则
以 `baseline*(1+memory_w)` 形式写入。

- 基线中心的反 Hebbian 规则，改编自 Huang, Luo et al. 2024 附录式
  3.2-3.5（Nature, doi:10.1038/s41586-024-07819-w）；**连接组本身不替换**，
  全图尖峰计数充当速率（rule.py:6-9）。
- 双态记忆 u/w 遗忘系统：τ_memory = 1800 s，权重低通 50 ms，
  上下界 [0.1, 2.0]×baseline（rule.py:7-10）。
- 更新（rule.py:29-32，闭合解）：
  `drive = η·(KC_rate·(gainᵀ·DAN_mid) − DAN_rate·KC_mid)`，
  `u ← u·e^{-h/τu} + drive·τu·(1-e^{-h/τu})`，`w` 同理由 u/w 两态卷积。
- **饱和显式上报不隐藏**（rule.py:33 注释）。

### 4.4 KC 适应（doom_learning_v6/brain.py:15-16, 44-47 + v6 kernel.cpp:29-33）

- KC 静息电位单独设为 -60 mV（其余 -52）；每次 KC 发放自适应电位
  +8 mV，τ = 200 ms 指数衰减——压低 KC 后续发放，防 KC 高频锁定。

### 4.5 其他工程要点

- `reset(keep_memory)`：重置动力学状态，可选保留记忆（brain.py:60-65）。
- 可塑性更新以 100 tick（10 ms）为 bin 由 Python 侧 `rule.advance` 结算
  （brain.py:100-115）；v6 中核内 LTD 被禁用（`learning=False`），
  4.2 的核内门控乘性抑郁代码存在但不生效，实际写权重的是 4.3 的速率规则。
- 检查点带全部配置哈希（graph/rule/tonic），恢复时逐项校验
  （brain.py:118-138）。

## 5. 与本仿真器（flybrain/）的差异决策

| 方面 | DOOMFLY | flybrain（本项目） |
|---|---|---|
| 平台 | CPU numba/clang，惰性事件驱动 | GPU (PyTorch)，每步全图向量化 |
| 图 | MaleCNS v1.0 保留神经元（~138k） | traced-only 子图 164,587 神经元 / 25.56M 边 |
| 递质来源 | annotations + consensus 预测 | `node_features.parquet` 的 `consensus_nt` |
| 可塑回路 | 仅 KC→MBON11（PPL101 门控） | 推广到全部 KC→MBON 边，DAN 取全部多巴胺能神经元（按 DAN→MBON 接触分数做 gain） |
| 学习规则 | 速率规则（主，核内 LTD 在 v6 中被禁用） | 采用核内门控乘性抑郁（v6 kernel.cpp 形式；该规则在 DOOMFLY v6 中不生效，是我们有意识的选择），速率规则未实现 |
| 感觉输入 | 视网膜亮度/糖感受器 | 泛化刺激接口：按类型/超类/名字注入电流（嗅觉用随机稀疏 KC 模式，DOOMFLY 无视网膜图时以 KC 直接刺激代替） |
| KC 适应 | 有（-60 mV 静息 + 8 mV 跳变） | 保留 |
| ambiguous 递质 | +1 | +1（同 DOOMFLY 默认） |

## 6. 本项目实测发现（DOOMFLY 文档之外）

- **超临界动力学**：在 traced-only 全图上以 DOOMFLY 原始权重尺度
  （0.275 mV/突触）运行时，任何足够强的脉冲刺激都会触发一个有界的
  持续活动吸引子（≥10 s 不衰减，主要由 cb_intrinsic 主导，群体率
  ≈3.4 spikes/s）。DOOMFLY 代码与文档未讨论此行为。权重尺度降到
  ≈0.041 mV/突触（×0.15）时网络进入亚临界：仍有局部传播，撤刺激后
  归零，但 KC→MBON 传导同时大幅减弱。本项目保留 0.275 为默认值
  （DOOMFLY 保真），受控实验以 episode 间 `reset(keep_memory=True)`
  协议保证各 episode 独立；两档工作点的对比见验证报告。
- **DAN 门控强度**：v6 的 gain 按 MBON 列归一化（每列和为 1），因此
  同时激活的 DAN 越多、发放率越高，抑郁速度线性越快。全量 392 个 DAN
  强刺激（20 mV）可在数个 trial 内把大部分可塑边压到 0.1 下限；
  配对实验应使用适中 US 强度（≈14 mV）或小的 DAN 子集。
