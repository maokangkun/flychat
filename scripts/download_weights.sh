#!/usr/bin/env bash
# 一键下载全部权重与数据(HuggingFace):
#   1) maokangkun/flychat-weights —— LoRA 适配器/前缀投影/编码器/连接组/3D viewer 数据
#   2) Qwen/Qwen2.5-1.5B-Instruct —— 基座模型(官方仓库直拉)
# 目录结构与本仓库一一对应, 下载完成即可 python server.py 4116。
set -euo pipefail
cd "$(dirname "$0")/.."

command -v hf >/dev/null 2>&1 || pip install -U "huggingface_hub[cli]"

echo "==> 下载 FlyChat 权重与连接组/3D 数据(约 320MB)..."
hf download maokangkun/flychat-weights --repo-type dataset --local-dir .

echo "==> 下载 Qwen2.5-1.5B-Instruct 基座(约 3GB)..."
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir checkpoints/llm/Qwen2.5-1.5B-Instruct

echo "==> 校验..."
if [ -f SHA256SUMS ]; then sha256sum -c SHA256SUMS --ignore-missing || echo "(部分文件校验失败, 请重跑下载)"; fi
echo "完成。启动: python server.py 4116"
