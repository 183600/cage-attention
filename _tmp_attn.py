import sys, math
sys.path.insert(0, "/home/runner/work/cage-attention/cage-attention")
import torch
import torch.nn.functional as F
from cage.model import Cfg, Attention

torch.manual_seed(0)

def ref_attn(att, x):
    cfg = att.cfg
    B,T,C = x.shape
    h, dk = att.h, att.dk
    kind = att.gate_kind
    W = att.qkv.weight
    qkv = F.linear(x, W).view(B,T,3,h,dk).permute(2,0,3,1,4)
    q,k,v = qkv[0], qkv[1], qkv[2]
    scale = att.inv_scale
    mask = torch.full((T,T), float('-inf'))
    mask.triu_(1)
    if kind == 'null':
        s = torch.matmul(q, k.transpose(-2,-1))*scale + mask
        s0 = torch.einsum('bhtd,hd->bht', q, att.null_k).unsqueeze(-1)*scale
        full = torch.cat([s0, s], dim=-1)
        p = torch.softmax(full, -1)
        y = torch.einsum('bhts,bhsd->bhtd', p[...,1:], v) + p[...,0:1]*att.null_v.view(1,h,1,dk)
        H = None
    else:
        s = torch.matmul(q, k.transpose(-2,-1))*scale + mask
        p = torch.softmax(s,-1)
        y = torch.einsum('bhts,bhsd->bhtd', p, v)
        if kind == 'calib':
            pc = p.clamp_min(1e-9)
            H = -(pc*pc.log()).sum(-1)  # (B,h,T)
    if kind == 'query':
        g = torch.sigmoid(F.linear(x, att.gate.weight, att.gate.bias))
        y = y * g.view(B,T,h,dk).permute(0,2,1,3)
    elif kind == 'calib':
        logT = math.log(max(T,2))
        Hbar = H.mean(1)/logT  # (B,T)
        pre = F.linear(x, att.gate.weight, att.gate.bias) - att.calib_a*logT - att.calib_c*att.calib_scale*Hbar.unsqueeze(-1)
        tau = F.softplus(att.calib_tau)+0.1
        g = torch.sigmoid(pre/tau)
        y = y * g.view(B,T,h,dk).permute(0,2,1,3)
    y = y.transpose(1,2).reshape(B,T,C)
    return F.linear(y, att.proj.weight)

for kind in ['none','query','calib','null']:
    for train in [False, True]:
        torch.manual_seed(1)
        cfg = Cfg(d_model=32, n_heads=4, n_layers=2, max_len=64, vocab_size=64,
                  gate=kind, dropout=0.0)
        att = Attention(cfg)
        if train: att.train()
        else: att.eval()
        x = torch.randn(3, 13, 32, requires_grad=True)
        out, stats = att(x)
        r = ref_attn(att, x)
        err = (out - r).abs().max().item()
        # gradient check
        go = torch.randn_like(out)
        g = torch.autograd.grad(out, x, go, retain_graph=True)[0]
        gr = torch.autograd.grad(r, x, go, retain_graph=True)[0]
        gerr = (g-gr).abs().max().item()
        print(f"{kind:6s} train={train}: out_err={err:.2e} grad_err={gerr:.2e}")
