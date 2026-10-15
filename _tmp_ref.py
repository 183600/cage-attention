import sys, math, time
sys.path.insert(0, "/home/runner/work/cage-attention/cage-attention")
import torch
import torch.nn as nn
import torch.nn.functional as F
from cage.data import make_batch


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.h = heads
        self.dk = d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x, mask):
        B, T, C = x.shape
        xn = self.ln1(x)
        qkv = self.qkv(xn).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk) + mask
        p = torch.softmax(s, -1)
        y = torch.matmul(p, v).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(y)
        xn = self.ln2(x)
        x = x + self.fc2(F.gelu(self.fc(xn)))
        return x


class Ref(nn.Module):
    def __init__(self, vocab, d=64, layers=2, heads=4, max_len=32):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._i)

    def _i(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos.weight[:T]
        mask = torch.full((T, T), float("-inf"))
        mask.triu_(1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln(x))


def run(vocab=16, n_pairs=4, d=64, layers=2, heads=4, steps=4000, lr=2e-3, seed=0):
    torch.manual_seed(0)
    m = Ref(vocab, d, layers, heads)
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    m.train()
    for step in range(steps):
        idx, _, meta = make_batch(64, n_pairs, vocab, "cpu", gen=gen)
        opt.zero_grad(set_to_none=True)
        logits = m(idx)
        loss = F.cross_entropy(logits[:, -1], meta["answer"])
        loss.backward()
        opt.step()
        if step % 2000 == 0:
            print("  ref step", step, "loss", round(loss.item(), 4))
    gen2 = torch.Generator().manual_seed(123)
    m.eval()
    c = t = 0
    with torch.no_grad():
        for _ in range(4):
            idx, _, meta = make_batch(256, 4, vocab, "cpu", gen=gen2)
            pred = m(idx)[:, -1].argmax(-1)
            c += (pred == meta["answer"]).sum().item()
            t += idx.size(0)
    return c / t


t0 = time.time()
print("ref acc", run(), "dt", round(time.time() - t0, 1))
