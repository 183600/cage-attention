import sys, math
sys.path.insert(0, "/home/runner/work/cage-attention/cage-attention")
import torch
import torch.nn.functional as F
from cage.model import build, Cfg, make_norm

def ref_attn(att, x):
    cfg = att.cfg
    B,T,C = x.shape
    h, dk = att.h, att.dk
    kind = att.gate_kind
    qkv = F.linear(x, att.qkv.weight).view(B,T,3,h,dk).permute(2,0,3,1,4)
    q,k,v = qkv[0], qkv[1], qkv[2]
    scale = att.inv_scale
    mask = torch.full((T,T), float('-inf'))
    mask.triu_(1)
    H=None
    if kind == 'null':
        s = torch.matmul(q, k.transpose(-2,-1))*scale + mask
        s0 = torch.einsum('bhtd,hd->bht', q, att.null_k).unsqueeze(-1)*scale
        p = torch.softmax(torch.cat([s0,s],dim=-1),-1)
        y = torch.einsum('bhts,bhsd->bhtd', p[...,1:], v) + p[...,0:1]*att.null_v.view(1,h,1,dk)
    else:
        s = torch.matmul(q,k.transpose(-2,-1))*scale + mask
        p = torch.softmax(s,-1)
        y = torch.einsum('bhts,bhsd->bhtd', p, v)
        if kind=='calib':
            pc=p.clamp_min(1e-9)
            H=-(pc*pc.log()).sum(-1)
    if kind=='query':
        g=torch.sigmoid(F.linear(x,att.gate.weight,att.gate.bias))
        y=y*g.view(B,T,h,dk).permute(0,2,1,3)
    elif kind=='calib':
        logT=math.log(max(T,2))
        Hb=H.mean(1)/logT
        pre=F.linear(x,att.gate.weight,att.gate.bias)-att.calib_a*logT-att.calib_c*att.calib_scale*Hb.unsqueeze(-1)
        g=torch.sigmoid(pre/(F.softplus(att.calib_tau)+0.1))
        y=y*g.view(B,T,h,dk).permute(0,2,1,3)
    y=y.transpose(1,2).reshape(B,T,C)
    return F.linear(y, att.proj.weight)

def ref_mlp(blk,x):
    h=F.gelu(F.linear(x,blk.mlp[0].weight,blk.mlp[0].bias))
    return F.linear(h, blk.mlp[1].weight, blk.mlp[1].bias)

def ref_norm(nm,x):
    if type(nm).__name__=='SeeDNorm':
        s=torch.sigmoid(F.linear(x,nm.scale.weight,nm.scale.bias))
        return F.layer_norm(x,(x.size(-1),))*s*2.0
    return F.layer_norm(x,(x.size(-1),),nm.ln.weight,nm.ln.bias)

def ref_block(blk,x):
    post = blk.cfg.norm=='post' or (blk.cfg.norm=='mix' and blk.layer_idx>=blk.cfg.n_layers//2)
    if post:
        a=ref_attn(blk.attn,x); x = x+a if blk.ls1 is None else x+a*blk.ls1
        x=ref_norm(blk.ln1,x)
        m=ref_mlp(blk,x); x = x+m if blk.ls2 is None else x+m*blk.ls2
        x=ref_norm(blk.ln2,x)
    else:
        a=ref_attn(blk.attn,ref_norm(blk.ln1,x)); x = x+a if blk.ls1 is None else x+a*blk.ls1
        m=ref_mlp(blk,ref_norm(blk.ln2,x)); x = x+m if blk.ls2 is None else x+m*blk.ls2
    return x

def ref_model(m, idx):
    T=idx.size(1)
    x=m.tok(idx)+m.pos.weight[:T]
    for blk in m.blocks:
        x=ref_block(blk,x)
    x=ref_norm(m.ln_f,x)
    return F.linear(x,m.head.weight)

if __name__ == "__main__":
    for kind in ['none','query','calib','null']:
        for norm in ['pre','post','mix','seednorm']:
            for ls in [None, 1e-2]:
                torch.manual_seed(7)
                cfg=dict(d_model=24,n_heads=3,n_layers=3,max_len=32,vocab_size=40,
                         gate=kind,norm=norm,layer_scale=ls)
                m=build(cfg,compile=False); m.eval()
                idx=torch.randint(0,40,(2,10))
                logits,_,_=m(idx)
                ref=ref_model(m,idx)
                err=(logits-ref[:,-1:]).abs().max().item()
                print(f"{kind:6s} {kind and kind:6s} norm={kind and kind} {kind}", end="")
                print("")
                print(f"  {kind:6s} {kind:5s} norm={m.cfg.norm:8s} ls={'y' if m.cfg.layer_scale else 'n'}: err={err:.2e}")
