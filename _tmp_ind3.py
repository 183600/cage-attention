import sys, time
sys.path.insert(0, "/home/runner/work/cage-attention/cage-attention")
import torch
from cage.model import build

V, N = 32, 4
FILL = 2


def batch_ind_varoffset(gen, B=64):
    # [t0..t_{N-1}, SEP, <f fillers>, t0..t_{N-1}]  with per-sequence f -> varying offset
    keys = torch.rand(B, V - 3, generator=gen).topk(N, dim=1).indices + 3
    f = torch.randint(0, 8, (B,), generator=gen)          # 0..7 fillers
    L = N + 1 + 8 + N
    seq = torch.full((B, L), FILL, dtype=torch.long)
    seq[:, :N] = keys
    seq[:, N] = 1                                          # SEP
    pos = N + 1 + f
    b = torch.arange(B)
    for i in range(N):
        seq[b, pos + i] = keys[b, i]
    qi = torch.randint(0, N - 1, (B,), generator=gen)
    tgt = torch.full_like(seq, -1)
    tgt[b, pos + qi] = seq[b, pos + qi + 1]                # next key
    return seq[:, :-1], tgt[:, 1:]


def run(steps=6000, lr=2e-3):
    torch.manual_seed(0)
    m = build(dict(d_model=64, n_layers=2, n_heads=4, vocab_size=V, max_len=32, gate="none"))
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    m.train()
    for step in range(steps):
        idx, tgt = batch_ind_varoffset(gen)
        opt.zero_grad(set_to_none=True)
        _, loss, _ = m(idx, tgt, sup_pos=None)
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            print(f"  varoffset step {step} loss {loss.item():.4f}")
    c = t = 0
    m.eval()
    with torch.no_grad():
        for _ in range(8):
            idx, tgt = batch_ind_varoffset(gen, 256)
            logits, _, _ = m(idx, tgt, sup_pos=None)
            flat = tgt.reshape(-1)
            flat = flat[flat != -1]
            c += (logits.argmax(-1) == flat).sum().item()
            t += flat.numel()
    print("varoffset acc", c / t)


run()
