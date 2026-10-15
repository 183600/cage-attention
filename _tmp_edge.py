import sys, itertools
sys.path.insert(0, "/home/runner/work/cage-attention/cage-attention")
import torch
from cage.model import build
from cage.data import make_batch, length_sweep_lengths

torch.manual_seed(0)
fails=[]
def check(name, fn):
    try:
        fn()
    except Exception as e:
        fails.append((name, repr(e)))
        print("FAIL", name, repr(e))

def fwd_bwd(gate, norm, n_pairs, batch, n_heads, d_model, dropout, ls, ckpt, max_len=64, vocab=None):
    vocab = vocab or 64
    cfg=dict(d_model=d_model,n_layers=2,n_heads=n_heads,max_len=max_len,
             vocab_size=vocab,gate=gate,norm=norm,dropout=dropout,
             layer_scale=ls,use_gradient_checkpointing=ckpt)
    m=build(cfg,compile=False)
    m.train()
    idx,tgt,meta=make_batch(batch,n_pairs,vocab,'cpu')
    logits,loss,_=m(idx,tgt,sup_pos=meta['query_pos'])
    assert torch.isfinite(loss), "loss nan"
    loss.backward()
    for p in m.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "grad nan"
    m.eval()
    _,_,st=m(idx,stat_pos=meta['query_pos'])
    for s in st:
        for k,v in s.items():
            if isinstance(v,torch.Tensor):
                assert torch.isfinite(v).all(), f"stat {k} nan"

# edge cases
for gate in ['none','query','calib','null']:
    check(f"n_pairs=1 {gate}", lambda g=gate: fwd_bwd(g,'pre',1,1,4,32,0.0,None,False,64,3))
    check(f"n_heads=1 {gate}", lambda g=gate: fwd_bwd(g,'seednorm',6,2,1,16,0.0,None,False))
    check(f"dropout+ckpt {gate}", lambda g=gate: fwd_bwd(g,'mix',6,4,4,32,0.4,1e-2,True))

# full grid small
for gate,norm in itertools.product(['none','query','calib','null'],['pre','post','mix','seednorm']):
    for ls in [None,1e-2]:
        for drop in [0.0,0.3]:
            check(f"grid {gate}-{norm}-{ls}-{drop}", lambda g=gate,n=norm,l=ls,d=drop: fwd_bwd(g,n,6,3,4,32,d,l,False))

# boundary lengths
for T in [2,3,4,5,32,33,34]:
    check(f"T~{T}", lambda T=T: fwd_bwd('calib','pre',(T-2)//2,2,2,16,0.0,None,False,max_len=max(256,T)))
    check(f"T~null~{T}", lambda T=T: fwd_bwd('null','pre',(T-2)//2,2,2,16,0.0,None,False,max_len=max(256,T)))

print("total fails:", len(fails))
for n,e in fails: print(" ", n, e)
