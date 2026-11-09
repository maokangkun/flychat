#!/usr/bin/env bash
# 一键下载全部权重与数据:
#   1) GitHub Release weights-v1 —— LoRA 适配器/前缀投影/编码器/连接组/3D viewer 数据(约 325MB)
#   2) Qwen/Qwen2.5-1.5B-Instruct —— 基座模型(官方仓库/HF 镜像直拉, 约 3GB)
# 资产名为扁平布局(a__b.pt), 下载后自动还原为仓库目录结构。
set -euo pipefail
cd "$(dirname "$0")/.."

command -v gh >/dev/null 2>&1 || { echo "需要 gh CLI(https://cli.github.com) 拉取 Release 资产"; exit 1; }
command -v hf >/dev/null 2>&1 || pip install -U "huggingface_hub[cli]"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "==> 下载 FlyChat 权重与连接组/3D 数据(GitHub Release, 约 325MB)..."
gh release download weights-v1 --repo maokangkun/flychat --dir "$TMP" --clobber

echo "==> 还原目录结构..."
python3 - "$TMP" <<'PY'
import os, shutil, sys
tmp = sys.argv[1]
for name in os.listdir(tmp):
    if "__" in name:
        rel = name.replace("__", "/")
        dst = os.path.join(os.getcwd(), rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
    else:  # SHA256SUMS 等顶层资产直接落仓库根
        rel = name
        dst = os.path.join(os.getcwd(), name)
    shutil.move(os.path.join(tmp, name), dst)
    print(" ", rel)
PY

echo "==> 下载 Qwen2.5-1.5B-Instruct 基座(约 3GB; 直连失败自动切换 hf-mirror)..."
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir checkpoints/llm/Qwen2.5-1.5B-Instruct \
  || HF_ENDPOINT=https://hf-mirror.com hf download Qwen/Qwen2.5-1.5B-Instruct \
       --local-dir checkpoints/llm/Qwen2.5-1.5B-Instruct

echo "==> 校验..."
sha256sum -c SHA256SUMS --ignore-missing || echo "(部分文件校验失败, 请重跑下载)"
echo "完成。启动: python server.py 4116"
