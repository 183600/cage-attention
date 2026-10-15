import sys
sys.path.insert(0, "/home/runner/work/cage-attention/cage-attention")
import torch
import torch.nn.functional as F
from cage.model import build
from cage.data import make_batch
from _tmp_full import ref_model

torch.manual_seed(3)
for kind in ['none','query','calib','null']:
    cfg=dict(d_model=24,n_heads=4,n_layers=2,max_len=64,vocab_size=64,gate=kind)
    m=build(cfg,compile=False)
    idx,tgt,meta=make_batch(4,6,64,'cpu')
    logits,loss,_=m(idx,tgt,sup_pos=meta['query_pos'])
    # reference
    x=ref_model(m,idx)  # (B,T,V)
    B=idx.size(0)
    qp=meta['query_pos']
    ref_logits=x[torch.arange(B),qp]
    ref_loss=F.cross_entropy(ref_logits, meta['answer'])
    print(f"{kind:6s}: logit_err={(logits-ref_logits).abs().max().item():.2e} loss_err={abs(loss-ref_loss).item():.2e}")
    # gradient check
    loss.backward()
    g_attn_proj = m.blocks[0].attn.proj.weight.grad.clone()
    m.zero_grad()
    ref_loss.backward()
    g2 = m.blocks[0].attn.proj.weight.grad
    print(f"        grad err on proj: {(g_attn_proj-g2).abs().max().item():.2e}")
