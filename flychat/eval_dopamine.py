"""eval_dopamine.py — 多巴胺门控可塑性对脑状态影响的示例。

协议（果蝇嗅觉回避学习协议的接口层版本）：
  1. 基线: 注入某 token 的刺激模式, 记录 DN 读出;
  2. 配对: 同一刺激 + DAN 子集刺激 (US), learning=True 反复配对;
  3. 测试: 再次注入同一刺激, 比较读出变化。
对照: unpaired (DAN 与刺激交替出现)。
用法: python eval_dopamine.py paired|unpaired   (每相独立进程, 控制显存峰值)

注: CUDA 显存峰值随仿真时长线性增长 (~3.3 GB/s 仿真时间, 疑为缓存分配器
在持续震荡吸引子下的碎片累积), 因此各相仿真窗口刻意压短以适配单进程
~8.3 GB 的显存上限。
"""
import json
import os
import sys

os.environ.setdefault("KMP_AFFINITY", "disabled")
sys.path.insert(0, "/data/mkk/malecns/flychat")

import torch

from flychat.corpus import build_sequences
from flychat.interface import StimEncoder, BrainSession

ROOT = "/data/mkk/malecns/flychat"
PRESENT_MS = 100.0


def main(phase):
    tok, _ = build_sequences()
    torch.manual_seed(0)
    enc = StimEncoder(len(tok.chars), 64, 2639, 4109, seed=0)
    sess = BrainSession(w_scale=0.275, device="cuda:0")
    enc = enc.to(sess.brain.device)

    token = torch.tensor([tok.stoi["a"]], device=sess.brain.device)
    vec_olf, vec_vis = enc(token)
    vec_olf, vec_vis = vec_olf[0], vec_vis[0]
    dan_subset = sess.brain.dan[:80]   # README §5: 避免全部 DAN 强刺激

    def present(n=2, ms=PRESENT_MS):
        """n 次 ms 毫秒的刺激, 返回最后一次子窗的状态读出。"""
        for _ in range(n):
            sess._set_tonic(vec_olf, vec_vis)
            sess.brain.run(ms)
        return sess._read_state(ms)

    results = {}
    # ---- 基线读出 ----
    sess.reset()
    s0 = present(2)
    results["baseline_dn_rate_hz"] = float(
        sess.brain.readout("dn", window_ms=PRESENT_MS).mean().item())

    if phase == "paired":
        for _ in range(3):
            sess._set_tonic(vec_olf, vec_vis)
            sess.brain.run(200.0, stim={dan_subset: 14.0}, learning=True)
    elif phase == "no_plasticity":
        # 相同时长、相同刺激, 但关闭可塑性 (learning=False)。
        # 注: 持续震荡吸引子会内源性激活部分 DAN, 因此仅去掉外源 DAN
        # 不足以构成对照; 这里直接关闭可塑性门控。
        for _ in range(3):
            sess._set_tonic(vec_olf, vec_vis)
            sess.brain.run(200.0, stim={dan_subset: 14.0})
    else:                              # DAN 与刺激交替 (不配对)
        for _ in range(3):
            sess.brain.run(200.0, stim={dan_subset: 14.0}, learning=True)
            present(1)
    results["after_weights"] = sess.brain.weights_summary()

    s1 = present(2)
    results["cond_dn_rate_hz"] = float(
        sess.brain.readout("dn", window_ms=PRESENT_MS).mean().item())
    results["state_L2_vs_baseline"] = float((s1 - s0).norm().item())
    results["state_L1_vs_baseline"] = float((s1 - s0).abs().sum().item())
    np_out = os.path.join(ROOT, "data", f"dopamine_states_{phase}.npz")
    import numpy as np
    np.savez(np_out, s0=s0.cpu().numpy(), s1=s1.cpu().numpy())
    out = os.path.join(ROOT, "data", f"dopamine_effect_{phase}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
