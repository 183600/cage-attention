import sys, time
sys.path.insert(0, "/home/runner/work/cage-attention/cage-attention")
import torch
import torch.nn.functional as F
from cage.model import build

torch.manual_seed(0)
V = 32
n = 8
SEP = 1
T = 2 * n + 1


def batch(B, gen):
    keys = torch.rand(B, V - 2, generator=gen).topk(n, dim=1).indices + 2
    seq = torch.empty(B, T, dtype=torch.long)
    seq[:, :n] = keys
    seq[:, n] = SEP
    seq[:, n + 1:] = keys
    idx = seq[:, :-1]
    tgt = seq[:, 1:].clone()
    # only supervise the second-half predictions (the induction part)
    mask = torch.zeros_like(tgt, dtype=torch.bool)
    mask[:, n + 1] = True          # ONLY supervised: current token t0 -> next t1
    tgt = tgt.masked_fill(~mask, -1)
    return idx, tgt


cfg = dict(d_model=128, n_layers=4, n_heads=4, vocab_size=V, max_len=64, gate='none', dropout=0.0)
model = build(cfg)
gen = torch.Generator().manual_seed(0)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()
for step in range(4000):
    idx, tgt = batch(64, gen)
    opt.zero_grad(set_to_none=True)
    _, loss, _ = model(idx, tgt)
    loss.backward()
    opt.step()
    if step % 1000 == 0:
        print("step", step, "loss", round(loss.item(), 4))
model.eval()
gen2 = torch.Generator().manual_seed(1)
correct = tot = 0
with torch.no_grad():
    for _ in range(4):
        idx, tgt = batch(512, gen2)
        logits, _, _ = model(idx, tgt)
        flat = tgt.reshape(-1)
        flat = flat[flat != -1]
        correct += (logits.argmax(-1) == flat).sum().item()
        tot += flat.numel()
print("induction acc", correct / tot)
