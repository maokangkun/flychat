# FlyChat 使用指南 — 果蝇脑混合对话系统（第三阶段交付）

FlyChat 是"生物仿真 + 人工读出"的混合对话系统：用户文本经编码器注入
MaleCNS v1.0 全脑 LIF 仿真器（164,587 神经元 / 25.56M 突触），每个 token
对应 16 ms 全脑仿真子窗，读出 GPT 从果蝇脑状态采样生成回复。

## 1. 环境要求

- conda 环境 `malecns`（torch 2.13 + CUDA，1 块 GPU 即可，显存 ~1.5 GB）
- 全部依赖已就位：仿真器 `../flybrain/`、权重 `checkpoints/`、viewer 数据

## 2. 启动

```bash
conda activate malecns
cd /data/mkk/malecns/flychat

# ① FlyChat 聊天服务（默认端口 8910）
python server.py                # 或 python server.py <port>

# ② 3D 查看器（默认端口 8900，另一个终端）
cd /data/mkk/malecns/viewer
python3 serve.py                # 或 python3 serve.py <port>
```

> `server.py` 启动时加载连接组，约需 1 分钟；就绪时会打印
> `[server] FlyChat http://localhost:8910`。
> 两个服务端口可不同，若聊天服务不在 8910，实时模式用
> `live.html?chat=http://localhost:<port>` 指定。

## 3. Web 聊天 + 实时全脑活动（推荐）

1. 浏览器打开 **<http://localhost:8910>** —— 聊天单页。
2. 浏览器打开 **<http://localhost:8900/live.html>** —— 3D 实时模式。
3. 在聊天页输入一句话回车。生成期间（每 token 约 0.3 s，一轮约 10–30 s）
   实时模式页面每 0.5 s 轮询一次 `/api/activity`，把最近 16 ms 子窗内
   放电的神经元按放电率高亮（暗红 → 亮黄）。

生成一个 token 的流程（每次采样前都会真实推进全脑仿真）：

```
user text → 字符 tokenize → 编码器注入感觉神经元电流
    → 16 ms 全脑 LIF 仿真 → 读出 [嗅觉 2,639 | 视觉 4,109 | DN 1,314] 放电率
    → 读出 GPT 采样下一 token → 把新 token 注入脑 → 循环到 '\n'
```

## 4. CLI 聊天（可选）

```bash
conda activate malecns
cd /data/mkk/malecns/flychat
python chat_cli.py                     # 交互式；/reset 清空对话
python chat_cli.py http://localhost:8910 "hi there"   # 单次模式
```

## 5. HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 聊天单页 (chat.html) |
| GET | `/api/status` | 服务状态（神经元数、状态维度、对话轮数） |
| POST | `/api/chat` | `{"text": "..."}` → `{"reply", "n_context_tokens", "n_generated_tokens", "wall_s", ...}` |
| GET | `/api/activity` | 最近一帧全脑活动：`{"seq", "window_ms", "n_active_brain", "active": [[viewer_idx, hz], ...]}`，`active` 按放电率降序、上限 3,000 个 |
| POST | `/api/reset` | 清空对话历史 |

```bash
curl -X POST localhost:8910/api/chat -H 'Content-Type: application/json' \
     -d '{"text": "hello, how are you?"}'
curl localhost:8910/api/activity | head -c 400
```

## 6. 验证

`python validate_web.py` 起服务并程序化验证全流程（7 项检查：
状态 / 页面 / 生成回复非空 / 活动数据非空 / 多轮上下文 / live.html / reset），
结果写入 `validation_result.json`。

## 7. 文件清单

```
flychat/
  server.py        聊天服务（HTTP，含 CORS，供 viewer 实时模式跨源轮询）
  chat.html        聊天单页（由 server.py 提供）
  chat_cli.py      命令行客户端
  validate_web.py  全流程程序化验证脚本
  validation_result.json  最近一次验证结果
../viewer/live.html + live.js   3D 实时模式页面（随 serve.py 提供）
```

## 8. 常见问题

- **回复像乱码**：符合预期。读出 GPT 仅 1.87M 参数、语料 14 万字符，
  生成呈"词形但无语义"（见 FINAL_REPORT.md §局限性）。
- **回复慢**：每 token 需要 16 ms 全脑仿真（约 0.3 s 墙钟），
  上下文 + 生成共几十个子窗。缩短对话可加快。
- **`CUDA out of memory`**：确认没有其他进程占用 GPU；
  server.py 单实例约需 1.5 GB 显存。
