import sys, time
sys.path.insert(0, "/home/runner/work/cage-attention/cage-attention")
import torch
from cage.model import build

V, N = 32, 4


def batch_ind_rnd(gen, B=64):
    keys = torch.rand(B, V - 2, generator=gen).topk(N, dim=1).indices + 2
    seq = torch.cat([keys, torch.full((B, 1), 1, dtype=torch.long), keys], dim=1)
    qi = torch.randint(0, N - 1, (B,), generator=gen)  # random repeat index
    tgt = torch.full_like(seq, -1)
    b = torch.arange(B)
    # position N+1+i holds token t_i; next token is t_{i+1}
    tgt[b, N + 1 + qi] = seq[b, N + 2 + qi]
    return seq[:, :-1], tgt[:, 1:]


def run(steps=4000, lr=2e-3):
    torch.manual_seed(0)
    m = build(dict(d_model=64, n_layers=2, n_heads=4, vocab_size=V, max_len=32, gate="none"))
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    m.train()
    for step in range(steps):
        idx, tgt = batch_ind_rnd(gen)
        opt.zero_grad(set_to_none=True)
        _, loss, _ = m(idx, tgt, sup_pos=None)
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            print(f"  rnd-induction step {step} loss {loss.item():.4f}")
    c = t = 0
    m.eval()
    with torch.no_grad():
        for _ in range(8):
            idx, tgt = batch_ind_rnd(gen, 256)
            logits, _, _ = m(idx, tgt, sup_pos=None)
            flat = tgt.reshape(-1)
            flat = flat[flat != -1]
            c += (logits.argmax(-1) == flat).sum().item()
            t += flat.numel()
    print("rnd-induction acc", c / t)


run()
