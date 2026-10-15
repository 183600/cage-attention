import sys, time
sys.path.insert(0, "/home/runner/work/cage-attention/cage-attention")
import torch
from cage.data import make_batch
from cage.model import build

V, N = 32, 4


def run(steps=16000, lr=2e-3):
    torch.manual_seed(0)
    m = build(dict(d_model=64, n_layers=2, n_heads=4, vocab_size=V, max_len=32, gate="none"))
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    m.train()
    t0 = time.time()
    for step in range(steps):
        idx, tgt, meta = make_batch(64, N, V, "cpu", gen=gen)
        opt.zero_grad(set_to_none=True)
        _, loss, _ = m(idx, tgt, sup_pos=None)
        loss.backward()
        opt.step()
        if step % 2000 == 0:
            # quick eval
            m.eval()
            with torch.no_grad():
                ie, te, _ = make_batch(256, N, V, "cpu", gen=gen)
                lg, _, _ = m(ie, te, sup_pos=None)
                acc = (lg.argmax(-1) == te.reshape(-1)[te.reshape(-1) != -1]).float().mean().item()
            m.train()
            print(f"  step {step} loss {loss.item():.4f} acc {acc:.3f} t {time.time()-t0:.0f}s")


run()
