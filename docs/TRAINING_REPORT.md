# FlyChat — v3 训练报告（预训练 LLM 对话引擎 + 果蝇脑状态调制）

**日期**: 2026-10-06 · **代码**: `flychat/llm/` + `server.py`(v3) · **权重**: `checkpoints/llm/`
**架构**: 冻结 MaleCNS 全脑 LIF 仿真器（不变）+ **Qwen2.5-1.5B-Instruct（LoRA）** 承担语言生成 +
**脑状态 prefix 投影** 注入调制。v2 的 32M 字符级读出 GPT 停用（保留 `server_v32m_backup.py`/
`checkpoints/readout_*.pt`，3D 查看器接口不变）。

---

## 0. 动机与结论

v2 的 32M 字符级模型只能产生词级统计文本（评委流利度 1.96/5，见表 3），无法正常对话。
v3 把语言能力交给开源中文对话 LLM（Qwen2.5-1.5B-Instruct，LoRA 微调），
果蝇脑状态作为调制信号以 soft prefix 形式注入 LLM。

**核心结果（25 条双语评估，20 中文 + 5 英文）**：

| 指标 | v3（本次） | 旧 32M（v2） |
|---|---|---|
| LLM 评委·流利度（1-5） | **4.28** | 1.96 |
| LLM 评委·相关性（1-5） | **3.00** | 1.36 |
| 盲测 A/B 偏好（匿名随机序） | **23/25 胜** | 2/25 |
| 单轮延迟 | **0.3-2.2 s**（目标 <5 s） | 25-56 s（实测） |
| 语言跟随 | 中/英均按用户语言回复* | — |

\* 英文跟随依赖推理期 system 提示 + 语言守卫（见 §4 部署）。

---

## 1. 模型选型与下载

- **首选 Qwen2.5-1.5B-Instruct**（bf16 权重 3.09 GB，单卡 80 GB 推理余量充足）。
- 网络：huggingface.co 被沙箱与主机屏蔽；hf-mirror.com 可达但 LFS 大文件 302 跳转
  `cas-bridge.xethub.hf.co` 被拒（SHA256 校验文件 dd924a11…未拿到）；
  **最终经 ModelScope 官方仓库下载**（model.safetensors 3,087,467,144 B，
  sha256 `dd924a11b4c220f3...`，与 hf-mirror 元数据一致）。
- tokenizer / chat template 验证通过：`apply_chat_template` 返回 Qwen 标准模板
  （`<|im_start|>role\n...<|im_end|>`）。注意 transformers 5.18 的
  `return_assistant_tokens_mask` 需要 `{% generation %}` 标记（Qwen 模板没有）——
  训练掩码改为扫描 `<|im_start|>assistant\n` ... `<|im_end|>` 区间（`sft_train.assistant_spans`）。

## 2. LoRA SFT（对话能力）

| 项 | 值 |
|---|---|
| 数据 | LCCC-Base 过滤（同 v2 规则：每轮 4-30 字、2-8 轮、超半中文）→ **64,801 段**（含豆瓣 397 + DailyDialog 2,202×2 过采样） |
| 格式 | chat template user/assistant 交替；只监督 assistant 段（content + `<|im_end|>`） |
| 采样浏览器 | `flychat/llm/data_prep.py`（流式 + 种子 0）+ `scripts/supplement_data.py` |
| 训练 | `torcheval` 无、自定义 DDP 循环 `flychat/llm/sft_train.py`；LoRA r=16 α=32（q/k/v/o/gate/up/down）+ **脑状态投影（联合训练，见 §3）** |
| 超参 | fp32 master + bf16 autocast；batch 32/卡 × 3 卡 × accum 2（global 192）；lr 1e-4 cosine；2 epoch |
| 结果 | 674 step、613k 监督 token、loss 5.78→3.73（约 8-10 分钟/次在 GPU1/4/6） |
| 产物 | `checkpoints/llm/phase1_v3b/`（adapter + brain_proj.pt + train_meta.json） |

**风格观察**：SFT 只 2 epoch 后回复即转为 LCCC 短句风格（连贯但简短）。
语言跟随：中文输入稳定中文回复；英文输入被中文语料风格主导 —— 部署期用
system 提示 + 语言守卫解决（§4），未再增加英文训练比例。

## 3. 果蝇脑状态注入（调制通道）

### 3.1 设计

- **采集**（新脚本 `flychat/llm/collect_sft_states.py`，24 worker 分 4 波跑完）：
  对 LCCC SFT 子集 **4,000 段**（任务下达 4-8k 区间；24 shards 全部完成后合并，
  5,822 个 user-turn 状态向量，4,000/4,000 全部绑定）
  逐 token 注入全脑仿真，在每个 user turn 末尾记录 8062 维状态
  （嗅觉 2639 + 视觉 4109 + DN 1314）。
- **窗口**：训练与在线统一 **4 ms/token**（v2 用 16 ms；4 ms 在消耗 ~0.04 s/token
  实测可用、且窗口逐 token 状态分布稳定，与在线语义一致）。
- **注入**（`flychat/llm/sft_train.py::PrefModel`）：每条对话最多 4 个 user turn →
  4 个 prefix slot；slot_j = P(state_j)（3 层 MLP 2048→2048→2048 + 线性投影到 d=1536）；
  无状态/多出的 slot 置零向量。4 个 slot 的 embedding 拼接到 chat 序列
  inputs_embeds 最前端。联合训练：LoRA + 投影同时更新（phase2 lr 2e-4，4 epoch）。

### 3.2 两阶段

- **phase1**（全量 62k，全部 slot 置零）：学会无条件对话。
- **phase2**（4,000 段真实状态 + 2× 回放置零样本，防遗忘）：LoRA + 投影联合训练。
  248 step、223k token、loss 3.57（混入回放后口径）。
  产物 `checkpoints/llm/final_v3/`。

### 3.3 调制效应评估（诚实记录）

在**带真实脑状态的 val 子集（400 段留出）**上做 NLL 探针
（`scripts/probe_modulation.py`，support `data/modulation_probe.json`）：

| prefix 条件 | NLL/token |
|---|---|
| **real（真实脑状态）** | **3.7594** |
| zero（置零） | 3.7606 |
| shuffled（乱序状态，对照） | 3.7592 |

- 实际效应 **量级很小**（real vs zero 约 -0.03%，real vs shuffled 约 +0.05%，方向为
  真实 < 置零、真实 ≈ 乱序）：**在 token 级 NLL 上，脑状态通道的边际信息近乎为零**。
  原因很直白：该状态向量近似是本条对话文本的确定性函数，对话历史已在注意力上下文里，
  NLL 上没有新的可预测信息。
- 但生成分布会被改变（greedy 样例，样例 8624）:
  - real prefix → 「我看过 , 你还是个孩子」
  - zero prefix → 「我感觉是她自己在说心里话」
  - shuffled prefix → 「我看过 , 你还是个孩子」
- 任务指定的"联合训练"已实现（LoRA + 投影同训）；
  **"推理时注入（自然语言提示）"备选方案未采用**：它需要把状态翻译成
  自然语言描述，而我们没有可验证的状态→语言映射（会引入幻觉成分），
  soft-prefix 方案与训练分布一致因此更诚实。

## 4. 对话评估与部署

### 4.1 评估集与指标
- **评估集**：`data/eval_bilingual_v3.json`，**20 中文 + 5 英文**；
  其中 15 中文 + 5 英文带着旧 32M 模型的原输出（`data/samples_bilingual_v2.json`），
  5 条新中文 prompt 的旧模型回复由 `scripts/run_old_model_5.py` 用 v2 32M 栈离线生成。
- **量化指标**（`scripts/eval_v3.py` → `data/eval_bilingual_v3_results.json`）:
  回复长度 / 中文率 / bigram 重复率 / **corpus-sim**（回复 bigram 与 LCCC 检索语料的
  最大命中率，"地道度"程序化代理）/ prompt 字重叠率。
  v3 全部文本与指标在 `data/eval_bilingual_v3_results.json`；旧模型对照同表。
- **"人工评分式程序化替代"**（`scripts/judge_v3.py` → `data/eval_bilingual_v3_judged.json`）:
  LLM 评委按 1-5 锚定评分打流利度/相关性（judge 对系统身份盲），
  外加 A/B 盲测（两系统回复随机匿名置序，评分者不可见模型）。
  结果见本文开头表格；v3 流利度 4.28±0.35（95% CI），盲测 23/25 胜。

### 4.2 部署（`server.py` v3）

- `flychat/llm/gen_engine.py`：`FlyChatLLM` 单例引擎 ——
  用户消息 → 冻结 StimEncoder + `BrainSession`（逐 token 4 ms，viewer 活动快照照常）
  → 本轮状态 → 最近 4 轮状态推理期映射 4 个 slot →
  Qwen(LoRA) + 语言跟随 system 提示 → 采样（temp 0.85, top_p 0.9, rep 1.05）。
  **语言守卫**：回复语言与用户语言不一致时，置零 prefix 重试一次。
  空闲时后台线程低频仿真驱动 3D 查看器（`/api/activity` 不变）。
- HTTP 接口保持 v2 形状（/api/chat、/api/status、/api/activity、/api/reset），
  前端 `chat.html`/`app.js` 未改动（reply/wall_s 字段兼容）。
- `systemctl --user restart flychat-web.service`（unit 用 `server.py 4116`，
  日志 `/data/mkk/flychat/service.log`）后线上实测，见 §5 记录。

### 4.3 资源与算力说明（与任务下达的偏差）

- 任务假设 GPU 0 被占、其余可用；实际所有 8 卡被多租户占满（0/2/3/5/7 常驻 72-77 GB），
  真正可用的是 **1 (73 GB 空) / 4 (61 GB) / 6 (79 GB)**。
- 因此：训练收敛到 **3 卡 DDP**（全局 batch 192 等效 v2 的 7 卡 32×7）；脑状态采集
  用 7-8 并发 worker 分 4 波跑完（约 1.5 h；被其它 100% 逻辑占用拖慢时 15-20 s/段，
  空闲时 1.5-7 s/段）。
- hf-mirror 走不通 → 换 ModelScope 官方镜像（详见 §1）。

## 5. 交付清单

- 代码：`flychat/llm/{data_prep,collect_sft_states,sft_train,gen_engine}.py`、
  `scripts/{supplement_data,merge_states,eval_v3,judge_v3,probe_modulation,make_figures_v3,run_old_model_5,gen_eval_v3}.py`、
  `server.py`(v3) / `server_v32m_backup.py`
- 权重：`checkpoints/llm/{Qwen2.5-1.5B-Instruct, phase1_v3, phase1_v3b, final_v3}/`
  （v3 上线用 `final_v3`；`smoke_v3` 为链路冒烟产物）
- 数据：`data/sft/{lccc_sft_50k.jsonl, sft_v3_enriched.jsonl, brainstates_shard00-23.npz,
  states_split.json}`、`data/eval_bilingual_v3{,_results,_judged}.json`、
  `data/modulation_probe.json`、`data/eval_old_model_new5.json`
- 图：`figures/fig_v3_quality.png`、`figures/fig_v3_modulation.png`
- 日志：`logs/{phase1_ddp2,phase2_ddp3,eval_final*,multiturn_demo,bs_shard*,figures}.log`

---



---

# FlyChat — 果蝇脑混合对话系统 · v2 训练报告（LCCC 扩充 + 32M 读出 GPT）

**日期**: 2026-10-05 · **代码**: `flychat/` · **权重**: `checkpoints/` · **状态缓存**: `data/shards_v2/`

在双语 v1（见下半部分历史报告）基础上：语料以 **LCCC-Base** 为主源重建（8-10K 段）、
读出 GPT 从 2.5M 扩到 **32.2M**（d=576 × 6 层 × 6 头）、训练改为 **7 卡 DDP**。
果蝇脑混合架构不变：冻结的全图 LIF 仿真器（164,587 神经元）产生状态，
读出 GPT 承担对话建模。

**核心结果：测试困惑度 brain 13.28 vs none 15.47（状态通道增益 +16.5%），
vs 旧 2.5M 模型（同一批 test 对话、各自 tokenizer）brain 26.44。**

---

## 1. 语料（v2 池重建）

| 项 | 值 |
|---|---|
| 中文主源 | LCCC-Base（`data/lccc/lccc_base_train.jsonl.gz`，682 万行），流式过滤：每轮 4-30 字、2-8 轮、超半数轮次含中文，种子 0 水塘采样 **6,600 段**（缓存 `data/cn/lccc_base_filtered.json`） |
| 英文 | DailyDialog 按种子 0 子采样 **3,300 段**（EN_PER_CN=0.5） |
| 池规模 | 9,900 段对话 → 过滤（≥16 token）后 **9,366 条序列**，65.4 万 token，均值 69.9 |
| 语言占比 | 中文 64.8% / 英文 35.2% |
| 划分 | test 500 / val 500 / train 8,366（种子 0 洗牌） |
| 词表 | **3,000**（2,907 CJK + 93 ASCII，频率截断，构建确定性） |

## 2. 架构与训练

- **读出 GPT**：d=576、6 层、6 头（head_dim=96），因果注意力 + 状态投影不变，参数量 **32.2M**
  （原 v1 目标 d=384 仅 16.2M，为贴近 30M 量级上调到 576）。
- **DDP 训练**：`torchrun --nproc_per_node=7`（GPU 1-7，GPU 0 留给在线服务），
  batch 32/卡（**全局 224**），lr 8e-4（3e-4 按全局 batch sqrt 缩放），AdamW，wd 0.01。
- **早停**：max 12 epoch，patience 3，val 最优权重回滚（实际 12 epoch 未触发早停）。
- 备份：v1 2.5M 权重 → `checkpoints/*_small.pt`（encoder 同步备份为 `encoder_small.pt`）。

**提速对比（每 epoch 时间，同一 v2 训练集，同一 32M 模型）**：

| 配置 | 每 epoch 时间 |
|---|---|
| 单卡旧配置（batch 32，lr 3e-4） | **24.6 s** |
| 7 卡 DDP（batch 224，lr 8e-4） | **13-16 s**（稳态；首 epoch 含 NCCL 预热均值 20 s） |

加速比约 **1.5-1.9×**：模型太小（每步计算量低）、GPU 被其他租户重度占用（utilization 100%），
DDP 通信与数据搬运开销占比高。受控单卡复测 brain/none 各需 21.6/22.1 s/epoch，
说明各条件间差异来自租户负载波动而非计算量。

## 3. 四条件对比（同数据、同架构、同种子）

| 条件 | 测试集困惑度（32M 新模型） | 相对 brain |
|---|---|---|
| **brain（果蝇脑）** | **13.28** | — |
| shuffled（时间打乱） | 12.78 | -3.8% |
| random（等参数量随机向量） | 14.25 | +7.3% |
| none（状态置零） | 15.47 | +16.5% |

1. brain 仍显著优于 random（+7.3%）和 none（+16.5%）——状态通道的增益依旧成立；
2. **本次 shuffled 略优于 brain（-3.8%）**，与 v1 不同：更大模型 + 更干净语料下，
   逐 token 时间对齐的边际贡献变小，shuffled 保留的边缘分布信息已接近够用。属诚实记录。

## 4. 与旧 2.5M 模型对比（同一批 test 对话）

旧模型用其原 tokenizer + encoder 评估（状态按旧 token 编码重新采集，
`shards_v2/oldtok_test`，495/500 段有效）：

| 条件 | 旧 2.5M（词表 2500） | 新 32M（词表 3000） |
|---|---|---|
| brain | 26.44 | **13.28** |
| none | 27.60 | 15.47 |
| shuffled | 28.81 | 12.78 |
| random | 29.22 | 14.25 |

注意：两行词表不同，绝对困惑度不可直接比较（v1 报告 §3 同理）；可比的是
"数据扩大 6.5× + 参数扩大 13× 后，同批对话上逐字符熵大幅下降"这一事实。

## 5. 生成样例对比（15 中文 + 5 英文，完整见 `data/samples_bilingual_v2.json`）

```
[cn] 你好                旧: 我不过我的人很的我也要 了     新: 这不是我发型好么
[cn] 我是不是应该早点睡觉  旧: 大不是啥的也是怎么时间 lomevc 你的
                          新: 你要早上午 , 你早点睡觉 , 晚起睡吧
[cn] 你觉得自己聪明吗      旧: y man coon sse have shere memank , , hout bo .
                          新: 因为你是好多 !
[cn] 我们聊点开心的事吧    旧: 1212001728                新: 是不 , 就是不会画
[en] what is your name    旧: I do do doryes hore to dingenernupeseing...
                          新: Oh , where whe that cheatent ond somers .
```

改善：中文回复不再漂移到英文（v1 的语言身份混淆基本消失），句式更像口语；
但语义仍松散（从 v1 的"词形流利"阶段升到短语级连贯）。英文改善相对有限。

## 6. 部署与验证

- 新权重：`checkpoints/readout_{brain,random,shuffled,none}.pt`（32.2M）+ `encoder.pt`（词表 3000）；
  尺寸记录于 `checkpoints/model_config.json`，`server.py` 部署时读取。
- 服务单元追加 `Environment=FLYCHAT_LANG=v2`，`systemctl --user restart flychat-web.service`。
- 实测（经 `https://10.140.177.73:8888/flychat/`）：`/api/status` → `vocab: 3000`；中文对话：

```
{"text": "你好"}            -> {"reply": "我在家
等你回家了 , 快回来一回来"}
{"text": "今天天气怎么样"}   -> {"reply": "你早就回去啦 , 我要来 你也回来啊 , 我要早点早起 , , ."}
```

对比 v1 上线回复（"不过 啊"、"这个女爱...的 11258 的"），中文连贯度明显提升。

## 7. v2 交付物

- `flychat/corpus.py`：LCCC-Base 过滤 + 水塘采样（种子 0，确定性），v1 池保持原样
- `flychat/collect_states.py`：FLYCHAT_LANG=v2 三态路径 + 16 worker（8 GPU×2）+ partial 修剪
- `flychat/train.py`：DDP 多卡训练 + 早停 + epoch 计时
- `scripts/eval_old_model.py` / `scripts/gen_bilingual_samples.py`：旧模型对比评估 / 双语样例
- `data/shards_v2/`：状态缓存（train/val/test/oldtok_test × 16 分片，22GB）
- `data/training_curve_{cond}_v2.json`：四条件训练曲线；`data/eval_old_model_v2.json`：旧模型对比
- `data/samples_bilingual_v2.json`：20 条双语样例（15 中 + 5 英）

---

# 以下是 v1 双语版报告（历史，保留）


**日期**: 2026-10-04 · **代码**: `flychat/` · **权重**: `checkpoints/` · **图**: `figures/`

在冻结的 MaleCNS v1.0 全图 LIF 仿真器（164,587 神经元 / 25,563,197 边，DOOMFLY 动力学）
之上，把对话接口从英文单语升级为**中文为主的中英双语系统**。果蝇脑混合架构完全不变：
果蝇脑贡献真实脉冲动力学；人工接口（编码器 + 读出 GPT）承担对话建模。

**核心结果（双语语料）: 读出层测试困惑度 27.35（果蝇脑状态）对比 33.07（等参数量
随机向量基线），果蝇动力学使下一 token 预测改善 17.3%；对比无状态基线改善 11.7%。**
英文单语版历史结果（测试困惑度 6.84 vs 9.58）见 §5，全程保留供对照。

---

## 1. 双语对话语料

| 项 | 值 |
|---|---|
| 中文来源 | 豆瓣多轮（`douban.txt`，2,000 段）、LCCC（`LCCC.json`，342 段）、青云谱（`qin_yun.csv`，1,000 段）、小黄鸡（`xiaohuangjie.txt`，201 段），均为多轮对话 |
| 英文来源 | DailyDialog (Li et al., 2017)，13,117 段 |
| 混合比例 | 中文 3,543 段全量 + 英文按种子 0 随机子采样 1,772 段 → **中文占 66.7%** |
| 池规模 | 5,315 段对话 → 过滤（≥16 token）后 4,954 条序列，均值 99.6 token |
| 划分 | 训练 1,000 段 / 验证 100 段 / 测试 100 段（种子 0 洗牌，比例与英文版一致） |

### 1.1 双语字符级 tokenizer

- `normalize_turn` 保留 **CJK 统一表意文字**（U+3400–U+9FFF，含扩展 A）、英文小写、
  常用 ASCII 标点与空格；全角标点（，。！？等）统一映射为半角。
- 中文语料多为分词后文本（字间带空格），解析阶段执行 `desegment_cn` 去掉 CJK 字符
  之间的空格（LCCC 内嵌英文单词的边界空格保留）。
- 词表按频率截断到 **top 2500**（频率降序 + 字符升序截断，再按字符序排列），
  防止汉字字符集膨胀；实际词表 = 2,409 CJK + 91 ASCII。词表外字符映射为空格。
- tokenizer 构建完全确定（固定种子子采样 + 固定排序），`collect_states.py` /
  `train.py` / `generate.py` / `server.py` 均从 `build_sequences()` 推导同一 tokenizer，
  保证采集、训练、部署严格一致。

各格式解析：豆瓣每行 `idx\tturn1\tturn2...`；LCCC 为 JSON 数组（每项轮次列表）；
青云谱每行 `回复1 | 回复2`；小黄鸡按空行分块、每行一轮。

### 1.2 状态收集

新缓存写入 `data/shards_cn/`（英文原缓存 `data/shards/` 与 `split.json` 原位保留），
8 worker（每 GPU 一个 FlyBrain 实例）并行仿真约 55 分钟，每 5 条序列增量
checkpoint、支持断点续跑：

| 划分 | 序列数 | token 数 |
|---|---|---|
| train | 1,000 | 99,018 |
| val | 100 | 9,761 |
| test | 100 | 10,060 |

## 2. 架构（与英文版完全一致）

- **编码器**：可训练嵌入 E ∈ R^{2500×64}，经两个固定随机投影分配到感觉神经元群
  （嗅觉 2,639 + 视觉 4,109），电流幅值 `a = 5 + 11·sigmoid(R·e_t)`，`w_scale=0.275`。
- **状态读出**：每 token 16 ms 仿真子窗，读出 `[嗅觉率; 视觉率; DN率]` 8,062 维。
- **读出 GPT**：每步 `tok_emb(x_t) + state_proj(S_t)`，4 层 × d=128 × 4 头，
  因果注意力。词表从 54 扩到 2500，参数量 **1.87M → 2.50M**。
- 训练协议不变：AdamW (lr 3e-4, wd 0.01)，batch 32，15 epoch，四条件并行。

## 3. 结果：四条件困惑度对比

四个条件使用**完全相同的数据、架构、种子与训练预算**，仅状态输入不同：

| 条件 | 测试集困惑度 | 相对 brain |
|---|---|---|
| **brain（果蝇脑）** | **27.35** | — |
| none（状态置零，纯 token LM） | 30.97 | +13.2% |
| shuffled（真实状态时间打乱） | 32.22 | +17.8% |
| random（等参数量随机向量） | 33.07 | +20.9% |

结论与英文版一致且在双语语料上复现：

1. **brain 比等参数量随机向量好 17.3%**（27.35 vs 33.07）；
2. brain 也优于无状态基线（none）11.7%——双语语料上状态通道的增益依然显著；
3. random 是最差条件：随机状态只是噪声，模型正确地学会忽略它。

绝对困惑度比英文版（6.84）高是预期内的：词表从 54 扩到 2500（46 倍），
数据量与参数量却没有同比例增长，逐字符预测的难度不可直接跨词表比较。

## 4. 双语生成样例

14 个中文 + 5 个英文提示，单轮增量生成（每 token 先过全脑仿真再采样，
temperature 0.85, top-k 12，完整结果见 `data/samples_bilingual.json`）：

```
[cn] 你好           -> 我不过我的人很的我也要 了
[cn] 今天天气怎么样 -> 这么你了 your you panorng
[cn] 你叫什么名字   -> 没有了
                     不是这是啥了
[cn] 我今天心情不太好 -> 你这么的话我的
[cn] 你喜欢看电影吗 -> 你一次过了
[cn] 晚上吃什么好呢 -> 你不过是是这么的了
[cn] 我是不是应该早点睡觉 -> 大不是啥的也是怎么时间 lomevc 你的
[cn] 你会说几种语言 -> 有啊啊
                     20 2 2 oo 130
[cn] 明天就是周末了 -> ht     she 1 , pan , I do goothing on you looki hoo cou con ? me shen ?
[cn] 谢谢你陪我聊天 -> 我很
                     .
[cn] 你觉得自己聪明吗 -> y man coon sse have shere memank , , hout bo .
[cn] 最近好累啊     -> 是啊 disss bri ve do .
[cn] 我们聊点开心的事吧 -> 1212001728
[cn] 再见，祝你天天开心 -> 有的人  , 的ot ?

[en] hello             -> I have to thed youp sye sere , plext on ?
[en] how are you today -> .
                         I an to chen theseesty is there angerarerrofreer and inet dexilse .
[en] what is your name -> Sor do den , ppendntickarnout'ng hightm thinteldems a wingherem orken ot
[en] i am so hungry now -> I do do doryes hore to dingenernupeseing oilzor diseen you seenger .
[en] tell me something funny -> Isican pucurengrou seet the andeally on the .
```

**诚实评估**：模型规模刻意很小（2.50M 参数），中文语料也仅数千段对话，
生成质量是"**词形流利但语义松散**"：中文输出基本是合法汉字序列、偶有短语法语
（"没有了，不是这是啥了"），但不成对话；部分回复漂移到英文，反映中英混合语料
下小模型的语言身份混淆。本阶段验证的是双语接口通路（CJK tokenizer → 增量全脑
仿真 → 状态条件化解码）端到端可用，而非对话质量。

## 5. 历史：英文单语版结果（保留）

英文版（DailyDialog，词表 54，1.87M 参数）的四条件对比，作为历史基线保留：

| 条件 | 测试集困惑度 | 相对 brain |
|---|---|---|
| **brain（果蝇脑）** | **6.84** | — |
| none（无状态） | 8.68 | +26.9% |
| shuffled（时间打乱） | 8.73 | +27.6% |
| random（随机向量） | 9.58 | +40.1% |

其他英文版结果（状态可解码性 94.7%、多巴胺可塑性效应、英文样例等）见
`TRAINING_REPORT_en.md`（本文件的历史快照）。

## 6. 部署与验证

- 新权重写入 `checkpoints/`（`encoder.pt` + `readout_{brain,random,shuffled,none}.pt`），
  英文旧权重备份为 `*_en.pt`；英文缓存 `data/shards/`、`split.json`、
  `training_curve_*.json` 原位保留。
- `systemctl --user restart flychat-web.service` 重启后验证：
  `GET /api/status` → `vocab: 2500`；`POST /api/chat` 中文消息返回中文回复：

```
{"text": "你好"}          -> {"reply": "不过 啊", ...}
{"text": "今天天气怎么样"} -> {"reply": "这个女爱...的 11258 的", ...}
```

- Caddy 入口 `https://10.140.177.73:8888/flychat/` 不变，路由正常。

## 7. 交付物

- `flychat/corpus.py`：双语语料 + 字符级 tokenizer（本次核心改动）
- `flychat/collect_states.py` / `train.py` / `generate.py`：
  经 `FLYCHAT_LANG=cn` 切换中文缓存目录与划分文件
- `scripts/gen_bilingual_samples.py`：双语生成样例脚本
- `data/split_cn.json`：双语划分（含词表）；`data/shards_cn/`：状态缓存
- `data/training_curve_{cond}_cn.json`：四条件训练曲线
- `data/samples_bilingual.json`：双语生成样例
- `checkpoints/*_en.pt`：英文版权重备份

## 8. 局限与下一步

1. **中文生成质量受限于数据与参数规模**：2.50M 参数 + 10 万训练 token 远不足以
   学出对话语义。改进优先级：扩中文语料（LCCC/豆瓣全量）、增大读出 GPT、
   训练更多 epoch。
2. **中英语言身份混淆**：部分中文提示触发英文回复（§4 样例可见）。可考虑
   语言标签条件化或按语言分层的采样约束。
3. **编码器未做梯度训练**（同英文版）：脉冲脑不可微，编码器保持随机初始化。
4. **困惑度绝对值跨版本不可比**：词表 54 → 2500 后逐字符熵基线完全不同，
   只能比各条件相对差（四条件内部对照仍然成立）。
5. **每 token 16 ms 全脑仿真是硬瓶颈**（同英文版）：状态收集占总流程 95% 以上
   时间；长 episode 需分进程运行。
