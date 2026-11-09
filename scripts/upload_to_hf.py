#!/usr/bin/env python3
"""可选: 把权重镜像到 HuggingFace dataset 仓库(需可直连 huggingface.co 的网络)。

用法:
    export HF_TOKEN=hf_xxx
    python scripts/upload_to_hf.py            # 默认 maokangkun/flychat-weights
    python scripts/upload_to_hf.py --repo someone/other-weights

上传后把 scripts/download_weights.sh 的 Release 下载段换成:
    hf download <repo> --repo-type dataset --local-dir .
"""
import argparse
import os

FILES = [
    "checkpoints/encoder.pt",
    "checkpoints/model_config.json",
    "checkpoints/llm/final_v3/adapter_config.json",
    "checkpoints/llm/final_v3/adapter_model.safetensors",
    "checkpoints/llm/final_v3/brain_proj.pt",
    "checkpoints/llm/final_v3/train_meta.json",
    "checkpoints/llm/final_v3/README.md",
    "data/processed/adjacency_minconf-0.5-traced-only_neurons.npz",
    "data/processed/node_ids_minconf-0.5-traced-only_neurons.parquet",
    "data/processed/node_features.parquet",
    "web/data/meta.json.gz",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="maokangkun/flychat-weights")
    args = ap.parse_args()
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = list(FILES)
    lod2 = sorted(
        f"web/data/lod2/{n}"
        for n in os.listdir(os.path.join(root, "web/data/lod2"))
        if n.endswith(".bin")
    )
    files += lod2 + ["SHA256SUMS"]
    for rel in files:
        src = os.path.join(root, rel)
        if not os.path.exists(src):
            print("skip(missing)", rel)
            continue
        api.upload_file(path_or_fileobj=src, path_in_repo=rel, repo_id=args.repo, repo_type="dataset")
        print("ok", rel)


if __name__ == "__main__":
    main()
