"""corpus.py — 双语（中文为主）语料加载 + 字符级 tokenizer。

英文数据源: DailyDialog (Li et al., 2017), GitHub 镜像
  https://github.com/amitabhadey/DailyDialogDataset (DailyDialog.csv)
  CSV 格式: Dialogues_Text 列, 轮次以 " $$$ " 分隔; 13,117 段对话。

中文数据源 (data/cn/):
  douban.txt       豆瓣多轮对话, 每行一段: "idx\tturn1\tturn2..."（分词后, 字间有空格）
  LCCC.json        JSON 数组, 每项为轮次列表（分词后, 字间有空格）
  qin_yun.csv      每行 "回复1 | 回复2"
  xiaohuangjie.txt 空行分隔的轮次文本, 每个 block 为一段对话

混合比例: 中文为主（约 2/3）, 英文 DailyDialog 随机子采样补足。
tokenizer: 字符级, 词表按频率截断到 top MAX_VOCAB（固定顺序, 保证确定性）。
"""
import csv
import json
import os
import random
import re
from collections import Counter
from typing import List

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CN_DIR = os.path.join(DATA_DIR, "cn")
NLP_DIR = os.path.join(CN_DIR, "nlp-dialogue")       # github.com/DengBoCong/nlp-dialogue
V2SRC_DIR = os.path.join(CN_DIR, "v2src")             # v2 补充语料（GitHub 下载）
DAILYDIALOG_CSV = os.path.join(DATA_DIR, "dailydialog.csv")

TURN_SEP = "\n"          # 对话轮次分隔符（进入词表）
MAX_SEQ_LEN = 128        # 训练序列最大 token 数
MAX_VOCAB = 2500         # 词表按频率截断上限（防 CJK 字符集膨胀）
MAX_VOCAB_V2 = 3000      # v2 词表上限（语料更丰富，适度放宽）
SEED = 0                 # 子采样 / 词表构建的固定种子
EN_PER_CN = 0.5          # 英文段数 / 中文段数（中文约占 2/3）
DGK_SAMPLE = 600         # dgk_shooter（电影对白）子采样段数
NOVEL_SAMPLE = 500       # 小说/剧本对话子采样段数
LCCC_BASE_GZ = os.path.join(DATA_DIR, "lccc", "lccc_base_train.jsonl.gz")
LCCC_BASE_CACHE = os.path.join(CN_DIR, "lccc_base_filtered.json")
LCCC_CN_COUNT = 6600     # 从 LCCC-Base 筛选的中文段数（种子 0 水塘采样）

# 允许保留的字符：ASCII 常用集 + CJK 统一表意文字（含扩展A）
_CJK = "\u3400-\u9fff"
_KEEP = f"a-z0-9 ,.!?;:$&%()\\-\\\"/#{_CJK}"
# 全角/中文标点 -> 半角
_PUNCT_MAP = {
    "，": ",", "。": ".", "！": "!", "？": "?", "：": ":", "；": ";",
    "（": "(", "）": ")", "“": '"', "”": '"', "‘": "'", "’": "'",
    "、": ",", "…": "...",
}


def desegment_cn(text: str) -> str:
    """去掉 CJK 字符之间的空格（douban/LCCC 等为分词后文本）。"""
    pat = "(?<=[%s])\\s+(?=[%s])" % (_CJK, _CJK)
    while True:
        new = re.sub(pat, "", text)
        if new == text:
            return new
        text = new


def normalize_turn(turn: str) -> str:
    """小写化 + 全角标点转半角 + 压空白，只保留中英文常用字符。"""
    t = turn.lower()
    for k, v in _PUNCT_MAP.items():
        t = t.replace(k, v)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(f"[^{_KEEP}]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _lccc_filter(dialogue):
    """每轮 4-30 字、2-8 轮的干净对话才保留."""
    turns = []
    for t in dialogue:
        t = normalize_turn(desegment_cn(t))
        if t:
            turns.append(t)
    if not (2 <= len(turns) <= 8):
        return None
    for t in turns:
        if not (4 <= len(t) <= 30):
            return None
    if sum(1 for t in turns
           if any("\u3400" <= c <= "\u9fff" for c in t)) * 2 < len(turns):
        return None                              # 超半数轮次要含中文
    return turns


def _build_lccc_cache(n=LCCC_CN_COUNT, seed=SEED):
    """流式过滤 LCCC-Base + 水塘采样到固定段数，缓存到 JSON（一次构建）."""
    import gzip
    import json as _json
    rng = random.Random(seed)
    reservoir = []
    with gzip.open(LCCC_BASE_GZ, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            try:
                dialogue = _json.loads(line)
            except Exception:
                continue
            if not isinstance(dialogue, list):
                continue
            turns = _lccc_filter(dialogue)
            if turns is None:
                continue
            if len(reservoir) < n:
                reservoir.append(turns)
            else:
                j = rng.randint(0, i)
                if j < n:
                    reservoir[j] = turns
    with open(LCCC_BASE_CACHE, "w") as f:
        _json.dump(reservoir, f, ensure_ascii=False)
    return reservoir


def load_lccc_base(n=LCCC_CN_COUNT, seed=SEED):
    """LCCC-Base 主数据源: 每行一个 JSON 数组（分词后文本，需去空格）. """
    if not os.path.exists(LCCC_BASE_CACHE):
        _build_lccc_cache(n=n, seed=seed)
    with open(LCCC_BASE_CACHE) as f:
        import json as _json
        raw = _json.load(f)
    return _norm_dialogues(raw, segmented=False)


# ---------------------------------------------------------------- 语料加载
def load_dialogues(path: str = DAILYDIALOG_CSV) -> List[List[str]]:
    """DailyDialog。返回 List[dialogue], dialogue = List[turn str]。"""
    dialogues = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["Dialogues_Text"].strip()
            if not text:
                continue
            turns = [t.strip() for t in text.split("$$$") if t.strip()]
            if turns:
                dialogues.append(turns)
    return dialogues


def _norm_dialogues(raw, segmented=True):
    out = []
    for d in raw:
        turns = [normalize_turn(desegment_cn(t) if segmented else t) for t in d]
        turns = [t for t in turns if t]
        if turns:
            out.append(turns)
    return out


def load_douban(path=None) -> List[List[str]]:
    """豆瓣多轮: 每行 "idx\\tturn1\\tturn2...". """
    path = path or os.path.join(CN_DIR, "douban.txt")
    raw = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if fields and fields[0].strip().isdigit():
                fields = fields[1:]
            raw.append([t for t in fields if t.strip()])
    return _norm_dialogues(raw)


def load_lccc(path=None) -> List[List[str]]:
    """LCCC: JSON 数组, 每项为轮次列表. """
    path = path or os.path.join(CN_DIR, "LCCC.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return _norm_dialogues(raw)


def load_qinyun(path=None) -> List[List[str]]:
    """青云谱: 每行 "回复1 | 回复2". """
    path = path or os.path.join(CN_DIR, "qin_yun.csv")
    raw = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                turns = [t.strip() for t in line.rstrip("\n").split("|")]
                raw.append([t for t in turns if t])
    return _norm_dialogues(raw)


def load_xiaohuangjie(path=None) -> List[List[str]]:
    """小黄鸡: 空行分隔的轮次文本, 每个 block 为一段对话. """
    path = path or os.path.join(CN_DIR, "xiaohuangjie.txt")
    raw, cur = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cur.append(line)
            elif cur:
                raw.append(cur)
                cur = []
    if cur:
        raw.append(cur)
    return _norm_dialogues(raw)


def load_tieba(path=None) -> List[List[str]]:
    """贴吧对话: 每行 turn1\\ttturn2... （tab 分隔多轮）. """
    path = path or os.path.join(NLP_DIR, "db", "data", "tieba.dialogues")
    raw = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            turns = [t.strip() for t in line.rstrip("\n").split("\t") if t.strip()]
            if turns:
                raw.append(turns)
    return _norm_dialogues(raw)


def load_stc_weibo(path=None) -> List[List[str]]:
    """STC 微博: post/response 行对 -> 2 轮对话. """
    base = path or os.path.join(NLP_DIR, "db", "data")
    raw = []
    with open(os.path.join(base, "stc_weibo_train_post"), "r",
              encoding="utf-8") as f1, \
         open(os.path.join(base, "stc_weibo_train_response"), "r",
              encoding="utf-8") as f2:
        for post, resp in zip(f1, f2):
            post, resp = post.strip(), resp.strip()
            if post and resp:
                raw.append([post, resp])
    return _norm_dialogues(raw)


def load_dgk(path=None, n=DGK_SAMPLE, seed=SEED) -> List[List[str]]:
    """dgk_shooter 电影对白: 'E' 分块, 'M ' 开头每行一轮（字间以 '/' 分隔）. """
    path = path or os.path.join(V2SRC_DIR, "dgk_shooter_min.conv")
    raw, cur = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("M "):
                turn = line[2:].strip().replace("/", "")
                if turn:
                    cur.append(turn)
            elif line.strip() == "E":
                if cur:
                    raw.append(cur)
                cur = []
    if cur:
        raw.append(cur)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(raw)), min(n, len(raw))))
    return _norm_dialogues([raw[i] for i in idx])


def _clean_novel_turns(turns):
    """丢弃 ASS/SRT 字幕样式残留行（如 'blur0 bord0 fn华文中宋 c&h...'）. """
    junk = re.compile(r"blur|bord|&h|华文中宋|方正|[a-z]{2,}\(", re.I)
    turns = [t.replace("\\N", "") for t in turns]
    return [t for t in turns if t and not junk.search(t)]


def load_novel_dialog(n=NOVEL_SAMPLE, seed=SEED) -> List[List[str]]:
    """小说/剧本对话语料（01miaom/novel_corpus）: 空行分块, 每块为一轮一行. """
    raw = []
    for sub in ("doraemon", "novels"):
        d = os.path.join(V2SRC_DIR, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            with open(os.path.join(d, fn), "r", encoding="utf-8",
                      errors="ignore") as f:
                cur = []
                for line in f:
                    line = line.strip()
                    if line:
                        cur.append(line)
                    elif cur:
                        cur = _clean_novel_turns(cur)
                        if len(cur) >= 2:
                            raw.append(cur)
                        cur = []
                if cur:
                    cur = _clean_novel_turns(cur)
                    if len(cur) >= 2:
                        raw.append(cur)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(raw)), min(n, len(raw))))
    return _norm_dialogues([raw[i] for i in idx])


class CharTokenizer:
    """字符级 tokenizer（中英双语, 词表按频率截断, 固定顺序）。"""

    def __init__(self, texts=None, max_vocab=MAX_VOCAB):
        if texts is not None:
            self.build(texts, max_vocab=max_vocab)

    def build(self, texts, max_vocab=MAX_VOCAB):
        counts = Counter()
        for t in texts:
            counts.update(t)
        # 频率降序 + 字符升序截断, 再按字符序排列 -> 确定性词表
        items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        keep = set(c for c, _ in items[:max_vocab])
        keep.update("\n", " ")           # 分隔符与 unk 兜底必入词表
        self.chars = sorted(keep)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        return self

    def encode(self, text: str) -> List[int]:
        unk = self.unk
        return [self.stoi.get(c, unk) for c in text]

    @property
    def unk(self) -> int:
        return self.stoi[" "]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def _pool_v1():
    """v1 池（与历史双语版严格一致）: 4 个中文源 + DailyDialog 子采样. """
    cn = []
    for loader in (load_douban, load_lccc, load_qinyun, load_xiaohuangjie):
        cn.extend(loader())
    return cn, load_dialogues()


def _pool_v2():
    """v2 池: LCCC-Base 筛选（主源）+ DailyDialog 英文子采样. """
    cn = load_lccc_base()
    return cn, load_dialogues()


def build_sequences(max_len: int = MAX_SEQ_LEN, max_vocab: int = MAX_VOCAB,
                    version=None, return_texts=False):
    """把混合语料展开成 token 序列。

    每段对话 -> "turn1\\nturn2\\n..." (\\n 为轮次分隔符)，
    超长时截断到 max_len。中文全量保留, 英文按 EN_PER_CN 比例子采样
    （种子 SEED, 顺序确定 -> tokenizer 严格可复现）。

    version: "v1"=历史双语池(默认), "v2"=扩充池；默认取环境变量
    FLYCHAT_LANG=v2，便于采集/训练/服务全链路一致切换。
    """
    if version is None:
        version = "v2" if os.environ.get("FLYCHAT_LANG") == "v2" else "v1"
    if version == "v2":
        cn, en = _pool_v2()
        vocab_cap = MAX_VOCAB_V2 if max_vocab == MAX_VOCAB else max_vocab
    else:
        cn, en = _pool_v1()
        vocab_cap = max_vocab

    rng = random.Random(SEED)
    n_en = int(round(len(cn) * EN_PER_CN))
    idx = sorted(rng.sample(range(len(en)), n_en))
    pool = cn + [en[i] for i in idx]

    texts = ["\n".join(d) for d in pool]
    tok = CharTokenizer()
    tok.build(texts, max_vocab=vocab_cap)

    sequences, kept_texts = [], []
    for d, text in zip(pool, texts):
        ids = []
        for turn in d:
            t_ids = tok.encode(turn + "\n")
            ids.extend(t_ids)
            if len(ids) > max_len:
                break
        if len(ids) < 16:      # 太短不用
            continue
        sequences.append(ids[:max_len])
        kept_texts.append(text)
    if return_texts:
        return tok, sequences, kept_texts
    return tok, sequences


if __name__ == "__main__":
    tok, seqs = build_sequences()
    n_turns = [len(s) for s in seqs]
    print(f"sequences: {len(seqs)}, total tokens: {sum(n_turns)}, "
          f"mean len {sum(n_turns)/len(seqs):.1f}, max {max(n_turns)}")
    print(f"vocab size: {tok.vocab_size}")
    cn_chars = sum(1 for c in tok.chars if "\u3400" <= c <= "\u9fff")
    print(f"vocab: {tok.vocab_size - cn_chars} ASCII + {cn_chars} CJK")
    print("sample decode:", repr(tok.decode(seqs[0][:120])))
