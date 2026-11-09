"""model.py — 果蝇脑状态条件化的小型 GPT 读出解码器。

输入每一步 = tok_emb(x_t) + state_proj(S_t)，因果注意力，预测 x_{t+1}。
"""
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class ReadoutGPT(nn.Module):
    def __init__(self, vocab, state_dim, d=128, n_layers=4, n_head=4,
                 max_len=256, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.state_proj = nn.Linear(state_dim, d)
        self.state_ln = nn.LayerNorm(d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.max_len = max_len
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, tokens, states):
        """tokens [B,T] int64, states [B,T,D] float -> logits [B,T,vocab]"""
        B, T = tokens.shape
        x = self.tok_emb(tokens) + self.state_ln(self.state_proj(states))
        x = self.drop(x + self.pos_emb.weight[:T].unsqueeze(0))
        mask = torch.triu(torch.full((T, T), float("-inf"),
                                     device=x.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))
