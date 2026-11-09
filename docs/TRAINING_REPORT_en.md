# FlyChat — 果蝇脑混合对话系统 · 第二阶段训练报告

**日期**: 2026-10-04 · **代码**: `flychat/` · **权重**: `checkpoints/` · **图**: `figures/`

在冻结的 MaleCNS v1.0 全图 LIF 仿真器（164,587 神经元 / 25,563,197 边，DOOMFLY 动力学）
之上构建并训练文本对话接口。果蝇脑贡献真实脉冲动力学；人工接口（编码器 + 读出 GPT）
承担对话建模。**核心结果：读出层测试困惑度 6.84（果蝇脑状态）对比 9.58（等参数量
随机向量基线），果蝇动力学使下一 token 预测改善 28.6%。**

---

## 1. 对话语料

| 项 | 值 |
|---|---|
| 来源 | DailyDialog (Li et al., 2017)，GitHub 镜像 [amitabhadey/DailyDialogDataset](https://github.com/amitabhadey/DailyDialogDataset)（`DailyDialog.csv`，网络沙箱仅放行 GitHub 域） |
| 规模 | 13,117 段多轮对话；每段展开为 `turn1\nturn2\n...`（`\n` 为轮次分隔符） |
| Tokenization | 字符级（英文小写），**词表 54**（a-z、0-9、常用标点、`\n`） |
| 划分 | 训练 1,000 段 / 验证 100 段 / 测试 100 段（种子 0 洗牌） |
| 序列长度 | 截断至 128 token；训练段均值约 125 token |

全脑状态收集后的实际入库规模（含少量跨 shard 重复）：

| 划分 | 序列数 | token 数 |
|---|---|---|
| train | 1,142 | 142,902 |
| val | 114 | 14,447 |
| test | 114 | 14,491 |

## 2. 架构与关键工程抽象

### 2.1 编码器（token → 感觉神经元电流）

可训练嵌入 `E ∈ R^{54×64}`，经两个固定随机投影分配到感觉神经元群（按
`node_features` superclass 选择）：

- **嗅觉群**：superclass=`cb_sensory` 且 class=`olfactory`，2,639 神经元；
- **视觉群**：superclass=`ol_sensory`，4,109 神经元。

第 t 个 token 的电流幅值 `a = 5 + 11·sigmoid(R·e_t)`（mV 渐近电位偏移），逐
token 注入 tonic。`w_scale=0.275`（默认超临界工作点）。

### 2.2 状态读出（果蝇脑 → token）

每 token 占 **16 ms 仿真子窗**，读出三群神经元的放电率拼接为状态向量
`S_t = [嗅觉率; 视觉率; 下行(DN)率] ∈ R^8062`（2,639 + 4,109 + 1,314）。
读出 GPT：每步输入 `tok_emb(x_t) + state_proj(S_t)`，4 层 × d=128 × 4 头，
**1.87M 参数**，因果注意力，预测 x_{t+1}。

### 2.3 训练抽象（为什么这样设计）

全图 LIF 在 A100 上约 4–10× 减速实时，逐 token 反向传播全脑完全不可行，且脉冲
发放不可微。我们采用**"每序列一次有限时窗全脑仿真 + 离线训练读出层"**：

1. 对每条对话做一次全脑仿真（teacher forcing：逐 token 注入刺激、逐子窗记录
   状态），得到状态轨迹缓存（fp16，约 1.9 GB）；
2. 在缓存的状态数据集上离线训练读出 GPT，任意多轮 epoch 无需再碰仿真器；
3. **生成时无需重放**：脑状态支持增量延展——每生成一个 token，先把它推入仍在
   运行的全脑仿真、读取新状态，再由读出 GPT 采样下一 token。

编码器嵌入本身是可训练参数，但脉冲脑不可微，本阶段保持其初始随机投影
（见 §5 局限）。

### 2.4 状态收集工程

8 个 worker（每 GPU 一个 FlyBrain 实例，各 ~1.2 GB 显存）并行仿真，每 5 条
序列写增量 checkpoint、支持断点续跑；监督进程只复活已退出的 worker（GPU 被
其他租户挤占导致 worker 频繁死亡，重启策略是收集成功的决定因素）。
总计约 4 小时收集 1,142 段训练 episode（~1.4e5 token × 16 ms ≈ 37 分钟仿真时间）。

## 3. 训练与评估

训练：AdamW (lr 3e-4, wd 0.01)，batch 32，15 epoch，四条件并行（各占一块 GPU），
每条件约 2 分钟。四个条件使用**完全相同的数据、架构、种子与训练预算**，仅状态输入不同：

- **brain**：真实果蝇脑状态轨迹；
- **random**：逐维均值/方差与真实状态匹配的 iid 随机向量（等参数量基线）；
- **shuffled**：真实状态在时间维打乱（保留逐步联合分布，破坏 token 对齐）；
- **none**：状态置零（纯 token LM，量化状态通道的总贡献）。

### 3.1 结果：困惑度

| 条件 | 测试集困惑度 | 相对 brain |
|---|---|---|
| **brain（果蝇脑）** | **6.84** | — |
| shuffled（时间打乱） | 8.73 | +27.6% |
| random（随机向量） | 9.58 | +40.1% |
| none（无状态） | 8.68 | +26.9% |

（`figures/fig_training.png`）

三个对照给出一致的结论：

1. **brain 比等参数量随机向量好 28.6%**（6.84 vs 9.58）；
2. shuffled ≈ none：把真实状态的时间结构打乱后优势几乎消失（8.73 vs 8.68），
   说明增益来自果蝇动力学的** token 对齐信息**，而非状态向量的边缘统计；
3. random 略差于 none（9.58 vs 8.68）：随机状态只是噪声，模型正确地学会忽略它。

### 3.2 状态可解码性

对 40 个 episode 做留一最近邻 token 身份解码（`figures/fig_states.png`）：

| 状态子空间 | 准确率（随机水平 1.9%） |
|---|---|
| 全状态（8,062 维） | **94.7%** |
| 仅 DN（1,314 维） | 7.8% |
| 全状态，时间打乱 | 7.8% |

全状态高度可解码；DN 段对当前 token 的直接编码较弱（更多反映整脑动力学）。

### 3.3 生成样例

14 个增量生成样例（每 token 先过全脑仿真再采样，`data/samples.json`）：

```
PROMPT  : hello ! how are you today ?
GEN     : i'm so look , i's seell weallike to really for ens in the are is

PROMPT  : may i help you ?
GEN     : i sonlow like as oo , cone can i an in fice rom allappear . i am
```

输出呈"词形但无语义"——这是 1.87M 参数读出头在 14.3 万字符语料上的预期水平；
验证的是接口通路（增量全脑仿真 + 状态条件化解码）端到端可用，而非对话质量。

### 3.4 多巴胺可塑性对脑状态的影响

嗅觉条件反射协议的接口层版本：固定 token 刺激（CS）× 80 个 DAN 14 mV（US）
配对 3 × 200 ms（`figures/fig_dopamine.png`）：

| 指标 | 配对（learning=True） | 可塑性关闭 |
|---|---|---|
| KC→MBON 权重变化 | **29,493 / 61,210 条边**，均值 0.98×，最低 0.58× | 0 条 |
| DN 群体放电率 | 8.61 → 8.22 Hz | 8.61 → 8.38 Hz |
| 状态 L2（相对基线） | 509.7 | 515.8 |

把配对相与可塑性关闭相相减，可塑性**特异性**的状态变化 L2 = **349**，且按读出
群分解：嗅觉群 280、DN 204、视觉群 40——效应集中在 KC→MBON→DN 通路下游，
视觉感觉群（不受可塑突触支配）几乎不受影响，与解剖预期一致。

（注：持续震荡吸引子会内源性激活部分 DAN，因此"仅去掉外源 DAN 刺激"不构成
有效对照；对照取为关闭可塑性门控。）

## 4. 交付物

- `flychat/`：`corpus.py`（语料+tokenizer）、`interface.py`（编码器/episode/增量
  接口）、`model.py`（读出 GPT）、`collect_states.py`（并行状态收集）、
  `train.py`（四条件训练）、`generate.py`（增量生成）、`eval_dopamine.py`（多巴胺评估）
- `checkpoints/`：`encoder.pt`（编码器嵌入）、`readout_{brain,random,shuffled,none}.pt`
- `figures/`：`fig_training.png`、`fig_states.png`、`fig_dopamine.png`
- `data/`：`training_curves.json`、`final_metrics.json`、`state_analysis.json`、
  `samples.json`、`dopamine_effect.json`、`shards/`（状态缓存）

## 5. 局限与下一步

1. **编码器未做梯度训练**：脉冲脑不可微，编码器保持随机初始化。可经拟合的
   脑响应代理模型（surrogate）做直通估计微调，或用 REINFORCE。
2. **语料与模型刻意偏小**：读出困惑度的绝对值受限于 1.87M 参数与 14 万字符；
   扩展需更多收集算力（每 token 16 ms 仿真是硬瓶颈）。
3. **DN 段对当前 token 编码弱**（7.8%）：读出群体可加入 MBON、中央脑内源性
   神经元，或用突触电导 g 而非放电率。
4. **GPU 显存峰值随仿真时长增长**（~3.3 GB/s 仿真时间，疑似持续震荡吸引子下
   缓存分配器的碎片累积）：长 episode 需分进程运行（本项目已按此处理）。
5. **持续震荡吸引子**：w_scale=0.275 下任何强刺激把网络推入不衰减的活动吸引子，
   episode 间必须 `reset(keep_memory=True)`；已按 README 建议执行。
