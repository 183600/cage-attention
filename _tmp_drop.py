import sys
sys.path.insert(0, "/home/runner/work/cage-attention/cage-attention")
import torch
import torch.nn as nn
from cage.model import build
from cage.data import make_batch

torch.manual_seed(0)
for kind in ['none','query','calib','null']:
    cfg = dict(d_model=32, n_heads=4, n_layers=2, max_len=64, vocab_size=64,
               gate=kind, dropout=0.5)
    m = build(cfg, compile=False)
    # replace every dropout with identity so train() uses explicit path but
    # applies no actual dropout
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.forward = lambda x: x
    idx, tgt, meta = make_batch(4, 6, 64, 'cpu')
    m.eval()
    with torch.no_grad():
        out_eval, _, st_eval = m(idx, stat_pos=meta['query_pos'])
    m.train()
    out_train, _, st_train = m(idx, stat_pos=meta['query_pos'])
    err = (out_eval - out_train).abs().max().item()
    serr = 0.0
    for a, b in zip(st_eval, st_train):
        for k in a:
            if a[k] is not None:
                serr = max(serr, (a[k]-b[k]).abs().max().item())
    print(f"{kind:6s}: out_err={err:.2e} stats_err={serr:.2e}")
