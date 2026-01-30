from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import AdaptiveAvgPool1d, AdaptiveMaxPool1d
import matplotlib.pyplot as plt


def exists(val):
    return val is not None
def default(val, d):
    return val if exists(val) else d


# pre-layernorm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, kv_include_self=False, similarity=None):#输入（batch_size, seq_len, dim）
        b, n, _ = x.shape
        h = self.heads
        context = default(context, x)

        if kv_include_self:
            # cross attention requires CLS token includes itself as key / value
            context = torch.cat((x, context), dim=1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if similarity is not None:
            similarity = similarity.unsqueeze(1).unsqueeze(-1)  # (batch_size, 1, seq_len, 1)
            similarity = similarity.expand(b, h, n, n)  # (batch_size, heads, seq_len, seq_len)
            dots = dots + similarity  # 将相似度信息加到注意力分数中


        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, context=None, similarity=None):
        for attn, ff in self.layers:
            x = attn(x, context=context, similarity=similarity) + x
            x = ff(x) + x
        return self.norm(x)
    
class CrossTransformer_MOD_AVG(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim, 1, heads, dim_head, mlp_dim, dropout=dropout),
                Transformer(dim, 1, heads, dim_head, mlp_dim, dropout=dropout)
            ]))
        self.gap = nn.Sequential(Rearrange('b n d -> b d n'),
                                 AdaptiveAvgPool1d(1),
                                 Rearrange('b d n -> b (d n)'))
        self.gmp = nn.Sequential(Rearrange('b n d -> b d n'),
                                 AdaptiveMaxPool1d(1),
                                 Rearrange('b d n -> b (d n)'))


    def forward(self, mri_tokens, pet_tokens):
        sim = F.cosine_similarity(mri_tokens, pet_tokens, dim=-1, eps=1e-08)
        for mri_enc, pet_enc in self.layers:
            mri_tokens = mri_enc(mri_tokens, context=pet_tokens, similarity=sim) + mri_tokens
            pet_tokens = pet_enc(pet_tokens, context=mri_tokens, similarity=sim) + pet_tokens
        mri_cls_avg = self.gap(mri_tokens)
        mri_cls_max = self.gmp(mri_tokens)
        pet_cls_avg = self.gap(pet_tokens)
        pet_cls_max = self.gmp(pet_tokens)
        cls_token = torch.cat([mri_cls_avg,  pet_cls_avg, mri_cls_max, pet_cls_max], dim=1)
        return cls_token, mri_tokens, pet_tokens
