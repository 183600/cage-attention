import sys, time
sys.path.insert(0, "/home/runner/work/cage-attention/cage-attention")
import torch
import torch.nn.functional as F
from cage.model import build

torch.manual_seed(0)
V = 32
n = 4


def batch_ind(gen):
    keys = torch.rand(64, V - 2, generator=gen).topk(n, dim=1).indices + 2
    seq = torch.cat([keys, torch.full((64, 1), 1, dtype=torch.long), keys], dim=1)
    idx = seq[:, :-1]
    tgt = torch.full_like(seq, -1)
    tgt[:, n + 1] = seq[:, n + 2]           # t0 -> t1 at the first repeat position
    return idx, tgt[:, 1:]


def batch_rec(gen):
    keys = torch.rand(64, V - 2, generator=gen).topk(n, dim=1).indices + 2
    vals = torch.randint(2, V, (64, n), generator=gen)
    seq = torch.empty(64, 2 * n + 3, dtype=torch.long)
    seq[:, :2 * n:2] = keys
    seq[:, 1:2 * n:2] = vals
    seq[:, 2 * n] = 1
    seq[:, 2 * n + 1] = keys[:, 0]          # query
    seq[:, 2 * n + 2] = vals[:, 0]          # answer
    tgt = torch.full_like(seq, -1)
    tgt[:, 2 * n + 1] = vals[:, 0]          # at query position (idx[2n+1])
    return seq[:, :-1], tgt[:, 1:]


def run(style, steps=4000, lr=2e-3):
    torch.manual_seed(0)
    m = build(dict(d_model=64, n_layers=2, n_heads=4, vocab_size=V, max_len=32, gate="none"))
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    m.train()
    bfn = batch_ind if style == "induction" else batch_rec
    for step in range(steps):
        idx, tgt = bfn(gen)
        opt.zero_grad(set_to_none=True)
        _, loss, _ = m(idx, tgt, sup_pos=None)
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            print(f"  {style} step {step} loss {loss.item():.4f}")
    gen2 = torch.Generator().manual_seed(999)
    c = t = 0
    m.eval()
    with torch.no_grad():
        for _ in range(8):
            idx, tgt = bfn(gen2)
            logits, _, _ = m(idx, tgt, sup_pos=None)
            flat = tgt.reshape(-1)
            flat = flat[flat != -1]
            c += (logits.argmax(-1) == flat).sum().item()
            t += flat.numel()
    print(style, "acc", c / t)


t0 = time.time()
run("induction")
run("recall")
print("dt", round(time.time() - t0, 1))
