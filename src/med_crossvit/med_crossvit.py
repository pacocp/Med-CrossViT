import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# feedforward layer

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# attention
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, kv_include_self=False):
        b, n, _, h = *x.shape, self.heads
        x = self.norm(x)
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim=1) # cross attention requires CLS token
                                                     # includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

# transformer encoder, for WSI patches and RNA-Seq

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
            Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
            FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

# projecting the CLS token, since WSI and RNA-Seq can have different dimensions

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x


# cross attention transformer

class CrossTransformer(nn.Module):
    def __init__(self, wsi_dim, rna_dim, depth, heads, dim_head, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(wsi_dim, rna_dim, Attention(wsi_dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                ProjectInOut(rna_dim, wsi_dim, Attention(rna_dim, heads=heads, dim_head=dim_head, dropout=dropout))]))

    def forward(self, wsi_tokens, rna_tokens):
        (wsi_cls, wsi_patch_tokens), (rna_cls, rna_seq_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (wsi_tokens, rna_tokens))

        for wsi_attend_rna, rna_attend_wsi in self.layers:
            wsi_cls = wsi_attend_rna(wsi_cls, context=rna_seq_tokens, kv_include_self=True) + wsi_cls
            rna_cls = rna_attend_wsi(rna_cls, context=wsi_patch_tokens, kv_include_self=True) + rna_cls

        wsi_tokens = torch.cat((wsi_cls, wsi_patch_tokens), dim=1)
        rna_tokens = torch.cat((rna_cls, rna_seq_tokens), dim=1)

        return wsi_tokens, rna_tokens


# bi-modal encoder

class BiModalEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        wsi_dim,
        rna_dim,
        wsi_enc_params,
        rna_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head=64,
        dropout=0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
            Transformer(dim=wsi_dim, dropout=dropout, **wsi_enc_params),
            Transformer(dim=rna_dim, dropout=dropout, **rna_enc_params),
            CrossTransformer(wsi_dim=wsi_dim, rna_dim=rna_dim, depth=cross_attn_depth, heads=cross_attn_depth, dim_head=cross_attn_dim_head, dropout=dropout)]
            ))

    def forward(self, wsi_tokens, rna_tokens):
        for wsi_enc, rna_enc, cross_attend in self.layers:
            wsi_tokens, rna_tokens = wsi_enc(wsi_tokens), rna_enc(rna_tokens)
            wsi_tokens, rna_tokens = cross_attend(wsi_tokens, rna_tokens)

        return wsi_tokens, rna_tokens

# tile-based wsi to token embedder

class TileEmbedder(nn.Module):
    def __init__(
    self,
    *,
    dim,
    num_tiles,
    dropout=0.,
    channels=3.,
    feature_extractor=None
    ):
        super().__init__()
        if feature_extractor:
            self.feature_extractor = feature_extractor

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tiles + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token,'() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        return self.dropout(x)

# rna to token embedder

class RNAEmbedder(nn.Module):
    def __init__(
    self,
    *,
    dim,
    num_genes,
    dropout=0.):
        super().__init__()
        hidden_dim = dim * num_genes
        self.to_rna_embedding = nn.Sequential(
                    nn.LayerNorm(num_genes),
                    nn.Linear(num_genes, hidden_dim),
                    Rearrange('b (n d) -> b n d', n=num_genes, d=dim),
                    nn.LayerNorm(dim)
            )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_genes + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.to_rna_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token,'() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        return self.dropout(x)

# Med-CrossViT

class MedCrossViT(nn.Module):
    def __init__(self,
    *,
    num_classes,
    wsi_dim,
    rna_dim,
    wsi_num_tiles=50,
    wsi_enc_depth=4,
    wsi_enc_heads=8,
    wsi_enc_mlp_dim=2048,
    wsi_enc_dim_head=64,
    rna_num_genes=100,
    rna_enc_depth=4,
    rna_enc_heads=8,
    rna_enc_mlp_dim=2048,
    rna_enc_dim_head=64,
    cross_atnn_depth=2,
    cross_attn_heads=8,
    cross_attn_dim_head=64,
    depth=3,
    dropout=0.1,
    emb_dropout=0.1,
    channels=3,
    pool='mean',
    return_attn=False
    ):
        super().__init__()
        self.wsi_embedder = TileEmbedder(dim=wsi_dim, num_tiles=wsi_num_tiles, dropout=emb_dropout)
        self.rna_embedder = RNAEmbedder(dim=rna_dim, num_genes=rna_num_genes, dropout=emb_dropout)

        self.bimodal_encoder = BiModalEncoder(
                depth=depth,
                wsi_dim=wsi_dim,
                rna_dim=rna_dim,
                cross_attn_heads=cross_attn_heads,
                cross_attn_depth=cross_atnn_depth,
                cross_attn_dim_head=cross_attn_dim_head,
                wsi_enc_params=dict(
                    depth=wsi_enc_depth,
                    heads=wsi_enc_heads,
                    mlp_dim=wsi_enc_mlp_dim,
                    dim_head=wsi_enc_dim_head
                    ),
                rna_enc_params=dict(
                    depth=rna_enc_depth,
                    heads=rna_enc_heads,
                    mlp_dim=rna_enc_mlp_dim,
                    dim_head=rna_enc_dim_head
                    ),
                dropout=dropout
        )

        self.wsi_mlp_head = nn.Sequential(nn.LayerNorm(wsi_dim), nn.Linear(wsi_dim, num_classes))
        self.rna_mlp_head = nn.Sequential(nn.LayerNorm(rna_dim), nn.Linear(rna_dim, num_classes))

        self.pool = pool
        self.return_attn = return_attn

    def forward(self, wsi_bag, rna):
        wsi_tokens = self.wsi_embedder(wsi_bag)
        rna_tokens = self.rna_embedder(rna)

        wsi_tokens, rna_tokens = self.bimodal_encoder(wsi_tokens, rna_tokens)

        if self.pool == 'mean':
            wsi_cls, rna_cls = map(lambda t: t.mean(dim=1), (wsi_tokens, rna_tokens))
        else:
            wsi_cls, rna_cls = map(lambda t: t[:, 0], (wsi_tokens, rna_tokens))

        wsi_logits = self.wsi_mlp_head(wsi_cls)
        rna_logits = self.wsi_mlp_head(rna_cls)

        if self.return_attn:
            return wsi_logits + rna_logits, (wsi_cls, rna_cls)

        return wsi_logits + rna_logits, None

if __name__ == '__main__':

    v = MedCrossViT(
        num_classes=2,
        depth=4,
        wsi_dim=768,
        rna_dim=768,
        wsi_num_tiles=50,
        wsi_enc_depth=2,
        wsi_enc_heads=8,
        wsi_enc_mlp_dim=2048,
        wsi_enc_dim_head=64,
        rna_enc_depth=2,
        rna_enc_heads=8,
        rna_enc_mlp_dim=2048,
        rna_enc_dim_head=64,
        rna_num_genes=100,
        cross_atnn_depth=2,
        cross_attn_heads=8,
        cross_attn_dim_head=64,
        dropout=0.1,
        emb_dropout=0.1)

    wsi_bag = torch.rand((16, 50, 768))
    rna_seq = torch.rand((16, 100))

    pred = v(wsi_bag, rna_seq)
    print(pred)
    print(pred.shape)
