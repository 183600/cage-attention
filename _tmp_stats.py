import sys, math
sys.path.insert(0, "/home/runner/work/cage-attention/cage-attention")
import torch
import torch.nn.functional as F
from cage.model import build, make_norm
from cage.data import make_batch


def ref_norm(nm, x):
    if type(nm).__name__ == 'SeeDNorm':
        s = torch.sigmoid(F.linear(x, nm.scale.weight, nm.scale.bias))
        return F.layer_norm(x, (x.size(-1),)) * s * 2.0
    return F.layer_norm(x, (x.size(-1),), nm.ln.weight, nm.ln.bias)


def ref_attn_stats(att, x, stat_pos):
    cfg = att.cfg
    B, T, C = x.shape
    h, dk = att.h, att.dk
    kind = att.gate_kind
    qkv = F.linear(x, att.qkv.weight).view(B, T, 3, h, dk).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    scale = att.inv_scale
    mask = torch.full((T, T), float('-inf'))
    mask.triu_(1)
    b = torch.arange(B)
    qs = q[b, :, stat_pos]                              # (B,h,dk)
    ks = k[b]                                            # (B,h,T,dk)
    s = torch.einsum('bhd,bhtd->bht', qs, ks) * scale + mask[stat_pos, :].unsqueeze(1)
    live = torch.arange(T).view(1, 1, T) <= stat_pos.view(B, 1, 1)
    s = s.masked_fill(~live, float('-inf'))
    null_mass = None
    if kind == 'null':
        s0 = torch.einsum('bhd,hd->bh', qs, att.null_k).unsqueeze(-1) * scale
        p = torch.softmax(torch.cat([s0, s], dim=-1), -1)
        null_mass = p[..., 0]
        pr = p[..., 1:]
        pr = pr / pr.sum(-1, keepdim=True).clamp_min(1e-9)
    else:
        pr = torch.softmax(s, -1)
    pc = pr.clamp_min(1e-9)
    ent = -(pc * pc.log()).sum(-1)
    mp = pr.max(-1).values

    gate = None
    if kind == 'query':
        g = torch.sigmoid(F.linear(x, att.gate.weight, att.gate.bias))
        gate = g[b, stat_pos]
    elif kind == 'calib':
        pc_full = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale + mask, -1).clamp_min(1e-9)
        H = -(pc_full * pc_full.log()).sum(-1)
        logT = math.log(max(T, 2))
        Hb = H.mean(1) / logT
        pre = F.linear(x, att.gate.weight, att.gate.bias) - att.calib_a * logT \
            - att.calib_c * att.calib_scale * Hb.unsqueeze(-1)
        gate = torch.sigmoid(pre / (F.softplus(att.calib_tau) + 0.1))[b, stat_pos]
    return ent, mp, null_mass, gate


def ref_mlp(blk, x):
    h = F.gelu(F.linear(x, blk.mlp[0].weight, blk.mlp[0].bias))
    return F.linear(h, blk.mlp[1].weight, blk.mlp[1].bias)


def ref_forward(model, idx, stat_pos):
    cfg = model.cfg
    B, T, _ = idx.shape
    x = model.tok(idx) + model.pos.weight[:T]
    out = []
    for blk in model.blocks:
        post = cfg.norm == 'post' or (cfg.norm == 'mix' and blk.layer_idx >= cfg.n_layers // 2)
        if cfg.norm == 'post' or (cfg.norm == 'mix' and blk.layer_idx >= cfg.n_layers // 2):
            a, xin = None, x
        xin = x if not cfg.norm.startswith('pre') else ref_norm(blk.ln1, x)
        s_meta = ref_attn_stats(blk.attn, x, stat_pos)
        # replicate actual block
        if blk.post:
            x = x + (ref_mlp(blk, ref_norm(blk.ln1, x + 0)) if False else 0)
        # (mlp not needed for stats; recompute block exactly via ref_block below)
        break
    return None


def ref_block_capture(blk, x, stat_pos):
    post = blk.cfg.norm == 'post' or (blk.cfg.norm == 'mix' and blk.layer_idx >= blk.cfg.n_layers // 2)
    if not post:
        xn = ref_norm(blk.ln1, x)
        ent, mp, nm, gate = ref_attn_stats(blk.attn, xn, stat_pos)
        a = ref_attn_out(blk.attn, x)
        return (None, ent, mp, None, None) if False else (x, None, None, None, None)
    return None


def ref_attn_out(att, x):
    B, T, C = x.shape
    h, dk = att.h, att.dk
    kind = att.gate_kind
    qkv = F.linear(x, att.qkv.weight).view(B, T, 3, h, dk).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    scale = att.inv_scale
    mask = torch.full((T, T), float('-inf')); mask.triu_(1)
    if kind == 'null':
        s = torch.matmul(q, k.transpose(-2, -1)) * scale + mask
        s0 = torch.einsum('bhtd,hd->bht', q, att.null_k).unsqueeze(-1) * scale
        p = torch.softmax(torch.cat([s0, s], dim=-1), -1)
        y = torch.einsum('bhts,bhsd->bhtd', p[..., 1:], v) + p[..., 0:1] * att.null_v.view(1, att.h, 1, att.dk)
    else:
        p = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale + mask, -1)
        y = torch.einsum('bhts,bhsd->bhtd', p, v)
    B_, T_, C_ = B, T, x.size(-1)
    if kind == 'query':
        g = torch.sigmoid(F.linear(x, att.gate.weight, att.gate.bias))
        y = y * g.view(B, T, att.h, att.dk).permute(0, 2, 1, 3)
    elif kind == 'calib':
        pc = p.clamp_min(1e-9)
        H = -(pc * pc.log()).sum(-1)
        logT = math.log(max(T, 2))
        Hb = H.mean(1) / logT
        pre = F.linear(x, att.gate.weight, att.gate.bias) - att.calib_a * logT \
            - att.calib_c * att.calib_scale * Hb.unsqueeze(-1)
        g = torch.sigmoid(pre / (F.softplus(att.calib_tau) + 0.1))
        y = y * g.view(B, T, att.h, att.dk).permute(0, 2, 1, 3)
    y = y.transpose(1, 2).reshape(B, T, C)
    return F.linear(y, att.proj.weight)


def ref_full(model, idx, stat_pos):
    cfg = model.cfg
    B, T = idx.shape
    x = model.tok(idx) + model.pos.weight[:T]
    stats_list = []
    for blk in model.blocks:
        post = cfg.norm == 'post' or (cfg.norm == 'mix' and blk.layer_idx >= cfg.n_layers // 2)
        if post:
            a = ref_attn_stats(blk.attn, x, stat_pos) if False else None
            st = ref_attn_stats(blk.attn, x, stat_pos)
            out = ref_attn_out(blk.attn, x)
            x = x + out if blk.ls1 is None else x + out * blk.ls1
            x = ref_norm(blk.ln1, x)
            m = ref_mlp(blk, x)
            x = x + m if blk.ls2 is None else x + m * blk.ls2
            x = ref_norm(blk.ln2, x)
        else:
            xn = ref_norm(blk.ln1, x)
            st = ref_attn_stats(blk.attn, xn, stat_pos)
            out = ref_attn_out(blk.attn, xn)
            x = x + out if blk.ls1 is None else x + out * blk.ls1
            xn2 = ref_norm(blk.ln2, x)
            m = ref_mlp(blk, xn2)
            x = x + m if blk.ls2 is None else x + m * blk.ls2
        stats_list.append(st)
    return stats_list


for gate in ('none', 'query', 'calib', 'null'):
    for norm in ('pre', 'post', 'seednorm', 'mix'):
        torch.manual_seed(0)
        cfg = dict(d_model=32, n_layers=3, n_heads=4, vocab_size=64, max_len=64,
                   gate=gate, norm=norm, dropout=0.0)
        model = build(cfg)
        model.eval()
        gen = torch.Generator().manual_seed(7)
        idx, _, meta = make_batch(5, 6, 64, 'cpu', gen=gen)
        stat_pos = meta['query_pos']
        _, _, stats = model(idx, stat_pos=stat_pos)
        ref = ref_full(model, idx, stat_pos)
        errs = {'ent': 0.0, 'mp': 0.0, 'null': 0.0, 'gate': 0.0}
        for s, (ent, mp, nm, g) in zip(stats, ref):
            errs['ent'] = max(errs['ent'], (s['entropy'] - ent).abs().max().item())
            errs['mp'] = max(errs['mp'], (s['maxprob'] - mp).abs().max().item())
            if nm is not None:
                errs['null'] = max(errs['null'], (s['null_mass'] - nm).abs().max().item())
                assert s['null_mass'] is not None
            if g is not None:
                errs['gate'] = max(errs['gate'], (s['gate'] - g).abs().max().item())
                assert s['gate'] is not None
        print(f"gate={gate:5s} norm={norm:8s} errs={errs}")
