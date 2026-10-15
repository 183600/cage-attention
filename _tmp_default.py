import sys, time
sys.path.insert(0, "/home/runner/work/cage-attention/cage-attention")
import torch
from cage.data import make_batch
from cage.model import build
from cage.train import _make_adamw, eval_accuracy

gate = sys.argv[1] if len(sys.argv) > 1 else "none"
lr = float(sys.argv[2]) if len(sys.argv) > 2 else 3e-4
cfg = dict(vocab_size=512, d_model=256, n_layers=8, n_heads=4, max_len=64, gate=gate)
m = build(cfg, compile=False)
params = list(m.parameters())
print("nparams", sum(p.numel() for p in params), "lr", lr)
opt = _make_adamw(params, lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1500, eta_min=lr * 0.1)
gen = torch.Generator().manual_seed(0)
m.train()
t0 = time.time()
for step in range(1500):
    idx, tgt, meta = make_batch(64, 16, 512, "cpu", gen=gen)
    opt.zero_grad(set_to_none=True)
    _, loss, _ = m(idx, tgt, sup_pos=meta["query_pos"])
    loss.backward()
    opt.step(); sched.step()
    if step % 25 == 0:
        print(f"{gate} step {step} loss {loss.item():.4f} t {time.time()-t0:.0f}s", flush=True)
    if time.time() - t0 > 540:
        print("TIMECAP at step", step, flush=True)
        break
print(gate, "final acc", eval_accuracy(m, 16, 512, "cpu"))
