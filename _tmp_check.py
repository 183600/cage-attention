import sys
sys.path.insert(0, "/home/runner/work/cage-attention/cage-attention")
import torch
import cage.model as M

torch.manual_seed(0)

def ref_full(q,k,v,scale):
    bh,T,dk=q.shape
    m=torch.full((T,T),float("-inf"),dtype=q.dtype)
    m.triu_(1)
    s=torch.baddbmm(m,q,k.transpose(1,2),beta=1,alpha=scale)
    p=torch.softmax(s,-1)
    y=torch.bmm(p,v)
    ent=-(p*torch.clamp_min(p,1e-9).log()).sum(-1)
    return y,ent

def test_calib(T=10, block=4, bh=6, dk=4):
    def qb(b, TT):
        if TT<=32:
            blocks=[(i,min(i+block,TT)) for i in range(0,TT,block)]
            return blocks
        return [(i,min(i+ (1<<20)//(b*TT),TT)) for i in range(0,TT,(1<<20)//(b*TT))]
    orig = M._query_blocks
    M._query_blocks = qb
    try:
        q=torch.randn(bh,T,dk,dtype=torch.double,requires_grad=True)
        k=torch.randn(bh,T,dk,dtype=torch.double,requires_grad=True)
        v=torch.randn(bh,T,dk,dtype=torch.double,requires_grad=True)
        scale=1.0/(dk**0.5)
        y,ent=M._CalibAttn.apply(q,k,v,scale,1e-9,True)
        yr,entr=ref_full(q,k,v,scale)
        print(f"calib T={T} block={block}: yerr {(y-yr).abs().max().item():.2e} enterr {(ent-entr).abs().max().item():.2e}")
        gy=torch.randn_like(y); ge=torch.randn_like(ent)
        g=torch.autograd.grad([y,ent],[q,k,v],[gy,ge])
        gr=torch.autograd.grad([yr,entr],[q,k,v],[gy,ge])
        for a,b,n in zip(g,gr,["dq","dk","dv"]):
            print("   ",n, f"{(a-b).abs().max().item():.2e}")
    finally:
        M._query_blocks = orig

def ref_null(q,k,v,null_k,null_v,scale,B,h):
    bh,T,dk=q.shape
    q4=q.view(B,h,T,dk)
    k4=k.view(B,h,T,dk)
    # scores real
    m=torch.full((T,T),float("-inf"),dtype=q.dtype); m.triu_(1)
    s=torch.baddbmm(m,q,k.transpose(1,2),beta=1,alpha=scale)  # (bh,T,T)
    s0=torch.matmul(q.view(M_ if False else bh,T,dk), k, ) # placeholder
    return None

def ref_null2(q,k,v,null_k,null_v,scale,B):
    bh,T,dk=q.shape
    h=null_k.size(0)
    # full (T+1) with null col 0
    m=torch.full((T,T),float("-inf"),dtype=q.dtype)
    m.triu_(1)
    s=torch.baddbmm(m,q,k.transpose(1,2),beta=1,alpha=scale)  # (bh,T,T)
    s0=torch.einsum('bhtd,hd->bht', q.view(B,h,T,dk), null_k).reshape(bh,T,1)*scale
    full=torch.cat([s0,s],dim=-1)
    p=torch.softmax(full,-1)
    y=torch.bmm(p[...,1:],v)+p[...,:1]*null_v.view(1,h,1,dk).expand(B,h,1,dk).reshape(bh,1,dk)
    return y

def test_null(T=10, block=4, B=2, h=3, dk=4):
    bh=B*h
    def qb(b, TT):
        return [(i,min(i+block,TT)) for i in range(0,TT,block)]
    orig=M._query_blocks
    M._query_blocks=qb
    try:
        q=torch.randn(bh,T,dk,dtype=torch.double,requires_grad=True)
        k=torch.randn(bh,T,dk,dtype=torch.double,requires_grad=True)
        v=torch.randn(bh,T,dk,dtype=torch.double,requires_grad=True)
        nk=torch.randn(h,dk,dtype=torch.double,requires_grad=True)
        nv=torch.randn(h,dk,dtype=torch.double,requires_grad=True)
        scale=1.0/(dk**0.5)
        y=M._NullAttn.apply(q,k,v,nk,nv,scale,B).view(B,h,T,dk)
        yr=ref_null2(q,k,v,nk,nv,scale,B).view(B,h,T,dk)
        print(f"null T={T} block={block}: yerr {(y-yr).abs().max().item():.2e}")
        gy=torch.randn_like(y)
        g=torch.autograd.grad(y,[q,k,v,nk,nv],gy)
        gr=torch.autograd.grad(yr,[q,k,v,nk,nv],gy)
        for a,b,n in zip(g,gr,["dq","dk","dv","dnk","dnv"]):
            print("  ",n,(a-b).abs().max().item())
    finally:
        M._query_blocks=orig

test_calib(T=10, block=4)
test_calib(T=10, block=7)
test_calib(T=13, block=5, bh=4)
test_null(T=10, block=4)
test_null(T=11, block=3)
