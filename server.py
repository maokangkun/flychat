"""server.py — FlyChat 对话入口：Web 单页聊天 + 全脑活动接口。

架构 v3（见 TRAINING_REPORT.md v3 章节）：
  用户文本 → 编码器（token -> 感觉神经元电流）
    → 全脑 LIF 仿真（每 token 4 ms 子窗，与训练状态采集同分布）
    → 状态向量 [嗅觉率; 视觉率; DN率]（8062 维）
    → BrainPrefixProjector 把最近 4 轮状态映射为 4 个 soft prefix token
    → 拼入 chat 模板 inputs_embeds 最前端 → Qwen2.5-1.5B(LoRA) 生成回复
  （v2 的 32M 字符级读出 GPT 保留在 server_v32m_backup.py；脑仿真器冻结不变，
    3D 查看器实时接口 /api/activity 继续工作，空闲时由后台线程驱动。）

HTTP 接口：
  GET  /                聊天单页 (chat.html)
  GET  /api/status      服务状态（模型、对话轮数）
  POST /api/chat        {"text": "..."} -> {"reply": "...", ...}
  GET  /api/activity    最近一个 token 子窗的全脑放电率快照（viewer 实时模式轮询）
  POST /api/reset       清空对话历史

启动：
  cd flychat && python server.py [port]   # 默认 8910
"""
import json
import os
import sys
import threading
import time
import urllib.parse

os.environ.setdefault("KMP_AFFINITY", "disabled")

import torch  # noqa: E402

# 可移植布局: 一切路径相对本文件解析, 可用环境变量整体重定位
ROOT = os.environ.get("FLYCHAT_ROOT") or os.path.dirname(
    os.path.abspath(__file__)
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "flybrain"))

from flychat.llm.gen_engine import FlyChatLLM  # noqa: E402

VIEWER_META = os.environ.get(
    "FLYCHAT_VIEWER_META", os.path.join(ROOT, "web", "data", "meta.json.gz")
)
VIEWER_JS = os.environ.get("FLYCHAT_VIEWER_JS", os.path.join(ROOT, "web", "js"))
VIEWER_DATA = os.environ.get(
    "FLYCHAT_VIEWER_DATA", os.path.join(ROOT, "web", "data")
)
PROCESSED_DIR = os.environ.get(
    "FLYCHAT_PROCESSED_DIR", os.path.join(ROOT, "data", "processed")
)
MIME = {".html": "text/html; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".gz": "application/gzip",
        ".bin": "application/octet-stream",
        ".json": "application/json; charset=utf-8"}


def _read(path):
    with open(path, "rb") as f:
        return f.read()

N_OLF, N_VIS, N_DN = 2639, 4109, 1314
STATE_DIM = N_OLF + N_VIS + N_DN
W_MS = 16.0                     # 每 token 仿真子窗（与训练/生成一致）
TOP_K_ACTIVE = 3000             # 活动快照最多返回的神经元数


class ChatEngine:
    """持有 v3 引擎（FlyChatLLM：脑仿真 + LLM + prefix 投影）的单例引擎。"""

    def __init__(self, device="cuda:0"):
        print("[engine] 启动 v3：脑仿真器 + Qwen2.5-1.5B(LoRA) ...", flush=True)
        self.state = {"seq": 0, "active": [], "n_active_brain": 0, "edges": []}
        self.turns_display = 0

        self._brain_meta = self._load_viewer_map()
        self._load_adjacency()
        self.engine = FlyChatLLM(
            device=device,
            activity_hook=self._on_activity)
        self.lock = self.engine.lock
        self.n_tokens_total = 0

    def _load_adjacency(self):
        """加载连接组邻接矩阵（CSR，traced-only 164587 节点），用于活动边计算。

        节点顺序 = node_ids parquet 的 bodyId 升序（build_dataset.py 生成），
        飞脑 body_id 同为升序 → 邻接行下标 == np.searchsorted(node_ids, brain_bid)。
        """
        import numpy as np
        import scipy.sparse as sp
        import pandas as pd
        try:
            self._adj = sp.load_npz(
                os.path.join(PROCESSED_DIR, "adjacency_minconf-0.5-traced-only_neurons.npz")).tocsr()
            nid = pd.read_parquet(
                os.path.join(PROCESSED_DIR, "node_ids_minconf-0.5-traced-only_neurons.parquet")
            )["bodyId"].to_numpy(np.int64)
            assert np.all(np.diff(nid) > 0), "node_ids 非升序，映射逻辑需修改"
            self._adj_bids = nid
            # flybrain body_id -> 邻接行下标（两个列表都按 bodyId 升序）
            import numpy as _np
            bb = _np.asarray(self._brain_bids, dtype=np.int64)
            p = np.searchsorted(nid, bb)
            self._brain2adj = np.where((p < len(nid)) & (nid[p.clip(max=len(nid)-1)] == bb),
                                       p, -1)
            print(f"[engine] 邻接矩阵就绪：{self._adj.shape}, nnz={self._adj.nnz}", flush=True)
        except Exception as e:  # noqa: BLE001
            self._adj = None
            print(f"[engine] 邻接矩阵加载失败（活动边将不可用）: {e}", flush=True)

    _EDGE_PERIOD_S = 0.25   # 边计算频率上限
    _EDGE_TOP_K = 1600      # 每窗最多返回的连线数
    _EDGE_MAX_ACTIVE = 3000 # 参与连线子矩阵的最大活动神经元数

    def _compute_edges(self, brain_idx, brain_rates):
        """活动神经元子图的最强连接，返回 [[brainA, brainB, w01], ...]（top-K）。"""
        import numpy as np
        if self._adj is None or not hasattr(self, "_brain2adj"):
            return []
        bs = np.asarray(brain_idx, dtype=np.int64)
        rates = np.asarray(brain_rates, dtype=np.float64)[:len(bs)]
        pos = self._brain2adj[bs]
        ok = pos >= 0
        bs = bs[ok]
        pos = pos[ok]
        if len(pos) < 2:
            return []
        if len(pos) > self._EDGE_MAX_ACTIVE:
            keep = np.argpartition(-rates[ok][:len(bs)], self._EDGE_MAX_ACTIVE)[:self._EDGE_MAX_ACTIVE]
            bs, pos = bs[keep], pos[keep]
        sub = self._adj[pos][:, pos].tocoo()
        if sub.nnz == 0:
            return []
        data = sub.data
        k = min(self._EDGE_TOP_K, len(data))
        if len(data) > k:
            top = np.argpartition(-data, k - 1)[:k]
        else:
            top = np.arange(len(data))
        pa, pb, data_top = sub.row[top], sub.col[top], data[top]
        wmax = float(data_top.max())
        out = []
        meta = self._brain_meta
        for a, b, wi in zip(pa.tolist(), pb.tolist(), data_top.tolist()):
            va, vbb = meta[bs[a]], meta[bs[b]]
            if va >= 0 and vbb >= 0:
                out.append([va, vbb, round(wi / wmax, 4)])
        return out


    def _load_viewer_map(self):
        import gzip
        print("[engine] 加载 viewer 元数据 ...", flush=True)
        with gzip.open(VIEWER_META, "rt") as f:
            vmeta = json.load(f)
        vpos = {bid: i for i, bid in enumerate(vmeta["bodyId"])}
        sys.path.insert(0, os.path.join(ROOT, "flybrain"))
        from flybrain import FlyBrain
        b = FlyBrain(w_scale=0.275, device="cpu", verbose=False)
        brain2viewer = [vpos.get(int(bid), -1) for bid in b.body_id]
        n_ok = sum(1 for v in brain2viewer if v >= 0)
        self._brain_bids = b.body_id.to_numpy() if hasattr(b.body_id, "to_numpy") \
            else list(b.body_id)
        del b
        print(f"[engine] bodyId 对齐：{n_ok}/{len(brain2viewer)} 个脑神经元"
              f" 可在 viewer 中显示", flush=True)
        return brain2viewer

    def _on_activity(self, active, n_active_brain):
        """FlyChatLLM 的活动回调：映射脑下标 -> viewer 下标。"""
        out = []
        for i, r in active:
            v = self._brain_meta[i] if i < len(self._brain_meta) else -1
            if v >= 0:
                out.append([v, float(r)])
        self.state["seq"] += 1
        self.state["window_ms"] = 4.0
        self.state["n_active_brain"] = n_active_brain
        self.state["active"] = out
        # 活动子图的强连接边（限频，避免每个 token 子窗都做稀疏切片）
        now = time.monotonic()
        if now - getattr(self, "_edge_ts", 0.0) >= self._EDGE_PERIOD_S:
            self._edge_ts = now
            try:
                self.state["edges"] = self._compute_edges(
                    [i for i, _ in active], [r for _, r in active])
            except Exception as e:  # noqa: BLE001
                self.state["edges"] = []

    # ------------------------------------------------------------------
    def chat(self, user_text: str):
        out = self.engine.chat(user_text)
        self.n_tokens_total += out.get("n_simulated_tokens", 0)
        self.turns_display = out.get("turns", 0)
        # 兼容 v2 前端字段
        return {**out,
                "n_context_tokens": out.get("n_context_msgs", 0),
                "n_brain_tokens": out.get("n_simulated_tokens", 0),
                "engine": "v3-llm"}

    def reset(self):
        self.engine.reset()
        self.turns_display = 0

    def activity(self):
        return dict(self.state)

    def status(self):
        st = self.engine.status()
        st.update({"turns": self.turns_display,
                   "n_tokens_total": self.n_tokens_total})
        return st


def main():
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8910
    engine = ChatEngine()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"),
                                            fmt % args))

        def do_GET(self):
            import re
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            def send_file(abspath):
                ext = os.path.splitext(abspath)[1].lower()
                try:
                    body = _read(abspath)
                except OSError:
                    self._send(404, b'{"error":"not found"}')
                    return
                self._send(200, body, MIME.get(ext, "application/octet-stream"))

            if path in ("/", "/index.html"):
                send_file(os.path.join(ROOT, "web", "index.html"))
            elif path == "/app.js":
                send_file(os.path.join(ROOT, "web", "app.js"))
            elif path == "/js/three.module.min.js":
                send_file(os.path.join(VIEWER_JS, "three.module.min.js"))
            elif path == "/js/OrbitControls.js":
                send_file(os.path.join(VIEWER_JS, "OrbitControls.js"))
            elif path == "/data/meta.json":
                if not hasattr(engine, "_meta_raw"):
                    import gzip as _gz
                    engine._meta_raw = _gz.open(VIEWER_META, "rb").read()
                self._send(200, engine._meta_raw, "application/json; charset=utf-8")
            elif re.fullmatch(r"/data/lod2/chunk_\d{3}\.bin", path):
                send_file(os.path.join(VIEWER_DATA, "lod2",
                                       os.path.basename(path)))
            elif path == "/api/status":
                with engine.lock:
                    self._send(200, json.dumps(engine.status()).encode())
            elif path == "/api/activity":
                with engine.lock:
                    self._send(200, json.dumps(engine.activity()).encode())
            else:
                self._send(404, b'{"error":"not found"}')

        def do_POST(self):
            if self.path == "/api/chat":
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    req = json.loads(self.rfile.read(length) or b"{}")
                    text = str(req.get("text", ""))[:2000]
                except Exception:
                    self._send(400, b'{"error":"bad request"}')
                    return
                with engine.lock:
                    try:
                        out = engine.chat(text)
                        self._send(200, json.dumps(
                            {**out, "ok": True},
                            ensure_ascii=False).encode())
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        self._send(500, json.dumps(
                            {"ok": False, "error": str(e)}).encode())
            elif self.path == "/api/reset":
                with engine.lock:
                    engine.reset()
                self._send(200, b'{"ok":true}')
            else:
                self._send(404, b'{"error":"not found"}')

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[server] FlyChat http://localhost:{port}  (Ctrl-C 停止)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
