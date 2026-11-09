# FlyChat · 果蝇脑对话 + 实时全脑 3D

用户文本 → 编码器（token → 感觉神经元电流）→ **全脑 LIF 仿真**（每 token 4ms 子窗，
164,587 个 traced-only 神经元）→ 状态向量 `[嗅觉率; 视觉率; DN率]`（8062 维）→
`BrainPrefixProjector` 把最近 4 轮状态映射为 4 个 soft prefix token → 拼入
Qwen2.5-1.5B(LoRA) 的 `inputs_embeds` 最前端 → 生成回复。生成期间，3D 视图实时
高亮放电神经元与突触连线。

> 架构与训练细节见 `docs/TRAINING_REPORT.md`（v3 章节）；连接组模型说明见
> `docs/flybrain_README.md`。

## 一键部署

```bash
git clone https://github.com/maokangkun/flychat && cd flychat

# 1) 环境
conda env create -f environment.yml && conda activate flychat

# 2) 权重与数据(GitHub Release 约 325MB + Qwen 基座约 3GB, 镜像自动回退)
./scripts/download_weights.sh

# 3) 启动(需要 CUDA GPU; 首次加载权重约 1 分钟)
python server.py 4116
```

打开 `http://127.0.0.1:4116/` 即可使用（聊天 + 实时 3D）。

常驻服务：`sudo cp systemd/flychat-web.service /etc/systemd/system/ && sudo systemctl enable --now flychat-web`（按需改 `WorkingDirectory` 与 python 路径）。

## 接入 SigmaFlow Science

SigmaFlow（`虚拟果蝇`页面）经同源代理访问本引擎，同机部署零配置；异地部署在
SigmaFlow 服务端设置：

```
SIGMAFLOW_FLY_URL=http://<引擎IP>:4116
```

## 仓库布局

```
server.py                 # HTTP 入口(聊天/活动快照/3D 数据/静态页)
flychat/                  # 训练与推理包(v3: llm/gen_engine.py 为生成引擎)
flybrain/flybrain.py      # 全脑 LIF 仿真器(连接组 CSR + 递质符号 + DAN 可塑性)
web/                      # 独立单页 UI(index.html + app.js + three.js viewer)
scripts/download_weights.sh
systemd/flychat-web.service
docs/                     # 训练报告/使用说明/连接组笔记
checkpoints/  data/  web/data/   # 由下载脚本填充(不随 git 分发)
```

路径全部相对仓库根解析，可用环境变量整体重定位：`FLYCHAT_ROOT`、
`FLYCHAT_PROCESSED_DIR`、`FLYCHAT_VIEWER_META/JS/DATA`。

## 权重内容（GitHub Release `weights-v1`）

脚本自动从 [Release weights-v1](https://github.com/maokangkun/flychat/releases/tag/weights-v1)
下载并还原目录布局（也可用 `scripts/upload_to_hf.py` 镜像到 HuggingFace 后改用
`hf download`）。

| 文件 | 说明 |
| --- | --- |
| `checkpoints/encoder.pt` | token → 感觉神经元电流编码器 |
| `checkpoints/model_config.json` | 编码器结构超参 |
| `checkpoints/llm/final_v3/` | LoRA 适配器 + `brain_proj.pt`（前缀投影） |
| `data/processed/adjacency_*.npz` | 连接组邻接（CSR, traced-only） |
| `data/processed/node_ids_*.parquet` | 邻接行 → bodyId 映射 |
| `data/processed/node_features.parquet` | 神经递质类型等节点特征 |
| `web/data/meta.json.gz` | 3D viewer 胞体坐标元数据 |
| `web/data/lod2/chunk_*.bin` | 全脑线框 LOD2 分块(104 块) |
| `SHA256SUMS` | 全部文件校验和 |

Qwen2.5-1.5B-Instruct 基座从官方 `Qwen/Qwen2.5-1.5B-Instruct` 拉取，不重复托管。

## HTTP 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/status` | 服务状态（模型、轮次、累计 token） |
| POST | `/api/chat` | `{"text": "..."}` → `{"reply", ...}`（引擎内串行） |
| GET | `/api/activity` | 最近 token 子窗全脑放电快照（viewer 轮询） |
| POST | `/api/reset` | 清空对话历史 |
| GET | `/data/meta.json`、`/data/lod2/chunk_NNN.bin` | viewer 数据 |

## 硬件需求

- GPU：≥8GB 显存（Qwen2.5-1.5B fp16 + LoRA + 脑仿真）；纯 CPU 可跑但生成显著变慢。
- 磁盘：约 4GB（权重 + 数据）。
- 依赖：见 `environment.yml`（Python 3.11 / torch ≥2.1 / transformers ≥4.45 / peft ≥0.13）。
