import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    """
    Check if a value is not None.

    Args:
        val: The value to check.

    Returns:
        bool: True if the value is not None, False otherwise.
    """
    return val is not None

def default(val, d):
    """
    Returns the value `val` if it exists, otherwise returns the default value `d`.

    Args:
        val: The value to check for existence.
        d: The default value to return if `val` does not exist.

    Returns:
        The value `val` if it exists, otherwise the default value `d`.
    """
    return val if exists(val) else d

# feedforward layer

class FeedForward(nn.Module):
    """
    FeedForward neural network module.

    This module implements a simple feedforward neural network with the following layers:
    - Layer normalization
    - Linear transformation
    - GELU activation
    - Dropout
    - Linear transformation
    - Dropout

    Args:
        dim (int): The input and output dimension of the feedforward network.
        hidden_dim (int): The hidden dimension of the feedforward network.
        dropout (float, optional): The dropout rate. Default is 0.

    Methods:
        forward(x):
            Forward pass through the feedforward network.

            Args:
                x (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Output tensor after passing through the feedforward network.
    """
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
    """
    Attention mechanism for neural networks.

    Args:
        dim (int): Dimension of the input features.
        heads (int, optional): Number of attention heads. Default is 8.
        dim_head (int, optional): Dimension of each attention head. Default is 64.
        dropout (float, optional): Dropout rate. Default is 0.

    Methods:
        forward(x, context=None, kv_include_self=False):
            Forward pass of the attention mechanism.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim).
                context (torch.Tensor, optional): Context tensor for cross-attention. Default is None.
                kv_include_self (bool, optional): Whether to include the input tensor itself as key/value. Default is False.

            Returns:
                torch.Tensor: Output tensor after applying attention mechanism.
    """
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
    """
    A Transformer model consisting of multiple layers of attention and feed-forward networks.

    Args:
        dim (int): The dimension of the input and output features.
        depth (int): The number of layers in the transformer.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        mlp_dim (int): The dimension of the feed-forward network.
        dropout (float, optional): The dropout rate. Default is 0.

    Attributes:
        layers (nn.ModuleList): A list of layers, each containing an attention mechanism and a feed-forward network.
        norm (nn.LayerNorm): A layer normalization applied to the output.

    Methods:
        forward(x):
            Passes the input through the transformer layers and applies layer normalization.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The output tensor after passing through the transformer layers and normalization.
    """
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
    """
    A neural network module that conditionally projects input and output dimensions
    to match the required dimensions for a given function.

    Attributes:
        fn (callable): The function to be applied to the input tensor.
        project_in (nn.Module): A linear layer to project the input tensor to the required dimension,
                                or an identity layer if no projection is needed.
        project_out (nn.Module): A linear layer to project the output tensor back to the original dimension,
                                 or an identity layer if no projection is needed.

    Args:
        dim_in (int): The dimension of the input tensor.
        dim_out (int): The dimension required by the function `fn`.
        fn (callable): The function to be applied to the input tensor.

    Methods:
        forward(x, *args, **kwargs):
            Applies the input projection, then the function `fn`, and finally the output projection.
            Args:
                x (torch.Tensor): The input tensor.
                *args: Additional positional arguments to be passed to `fn`.
                **kwargs: Additional keyword arguments to be passed to `fn`.
            Returns:
                torch.Tensor: The output tensor after applying the projections and the function `fn`.
    """
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
    """
    A CrossTransformer module that performs cross-attention between WSI (Whole Slide Image) tokens and RNA tokens.

    Args:
        wsi_dim (int): Dimension of the WSI tokens.
        rna_dim (int): Dimension of the RNA tokens.
        depth (int): Number of cross-attention layers.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head.
        dropout (float, optional): Dropout rate. Default is 0.

    Methods:
        forward(wsi_tokens, rna_tokens):
            Performs the forward pass of the CrossTransformer.
            
            Args:
                wsi_tokens (torch.Tensor): Input tokens for WSI, shape (batch_size, seq_len, wsi_dim).
                rna_tokens (torch.Tensor): Input tokens for RNA, shape (batch_size, seq_len, rna_dim).
            
            Returns:
                Tuple[torch.Tensor, torch.Tensor]: The transformed WSI and RNA tokens.
    """
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
    """
    BiModalEncoder is a neural network module designed to encode two different modalities of data 
    (e.g., whole slide images (WSI) and RNA sequences) using separate transformers for each modality 
    and a cross-transformer to facilitate interaction between the two modalities.

    Args:
        depth (int): Number of layers in the encoder.
        wsi_dim (int): Dimensionality of the WSI tokens.
        rna_dim (int): Dimensionality of the RNA tokens.
        wsi_enc_params (dict): Parameters for the WSI transformer encoder.
        rna_enc_params (dict): Parameters for the RNA transformer encoder.
        cross_attn_heads (int): Number of attention heads in the cross-attention mechanism.
        cross_attn_depth (int): Depth of the cross-attention mechanism.
        cross_attn_dim_head (int, optional): Dimensionality of each attention head in the cross-attention mechanism. Default is 64.
        dropout (float, optional): Dropout rate. Default is 0.

    Methods:
        forward(wsi_tokens, rna_tokens):
            Forward pass through the BiModalEncoder.
            
            Args:
                wsi_tokens (torch.Tensor): Input tokens for the WSI modality.
                rna_tokens (torch.Tensor): Input tokens for the RNA modality.
            
            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Encoded WSI and RNA tokens after processing through the encoder.
    """
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
    """
    TileEmbedder is a neural network module that embeds input tiles into a higher-dimensional space
    and adds positional embeddings and a class token.

    Args:
        dim (int): The dimension of the embedding space.
        num_tiles (int): The number of tiles to embed.
        dropout (float, optional): Dropout rate. Default is 0.
        channels (float, optional): Number of input channels. Default is 3.
        feature_extractor (nn.Module, optional): A feature extractor module. Default is None.

    Attributes:
        feature_extractor (nn.Module): The feature extractor module if provided.
        pos_embedding (nn.Parameter): Positional embeddings for the tiles.
        cls_token (nn.Parameter): Class token to be prepended to the input.
        dropout (nn.Dropout): Dropout layer.

    Methods:
        forward(x):
            Forward pass of the TileEmbedder. Adds class token and positional embeddings to the input,
            then applies dropout.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, num_tiles, dim).

            Returns:
                torch.Tensor: Output tensor after adding class token, positional embeddings, and applying dropout.
    """
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
    """
    RNAEmbedder is a neural network module designed to embed RNA sequences into a high-dimensional space.

    Args:
        dim (int): The dimensionality of the embedding space.
        num_genes (int): The number of genes in the RNA sequence.
        dropout (float, optional): Dropout rate to apply after embedding. Default is 0.0.

    Attributes:
        to_rna_embedding (nn.Sequential): A sequential container to normalize, linearly transform, and rearrange the RNA sequence.
        pos_embedding (nn.Parameter): Positional embeddings for the RNA sequence.
        cls_token (nn.Parameter): A classification token added to the beginning of the sequence.
        dropout (nn.Dropout): Dropout layer applied to the final output.

    Methods:
        forward(x):
            Forward pass of the RNAEmbedder.
            
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, num_genes).
            
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, num_genes + 1, dim) after embedding, positional encoding, and dropout.
    """
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
    """
    Med-CrossViT: A WSI-RNA-Seq fusion model.

    Args:
        num_classes (int): Number of output classes.
        wsi_dim (int): Dimension of the whole slide image (WSI) embeddings.
        rna_dim (int): Dimension of the RNA embeddings.
        wsi_num_tiles (int, optional): Number of tiles in the WSI. Default is 50.
        wsi_enc_depth (int, optional): Depth of the WSI encoder. Default is 4.
        wsi_enc_heads (int, optional): Number of heads in the WSI encoder. Default is 8.
        wsi_enc_mlp_dim (int, optional): MLP dimension in the WSI encoder. Default is 2048.
        wsi_enc_dim_head (int, optional): Dimension of each head in the WSI encoder. Default is 64.
        rna_num_genes (int, optional): Number of genes in the RNA data. Default is 100.
        rna_enc_depth (int, optional): Depth of the RNA encoder. Default is 4.
        rna_enc_heads (int, optional): Number of heads in the RNA encoder. Default is 8.
        rna_enc_mlp_dim (int, optional): MLP dimension in the RNA encoder. Default is 2048.
        rna_enc_dim_head (int, optional): Dimension of each head in the RNA encoder. Default is 64.
        cross_atnn_depth (int, optional): Depth of the cross-attention layers. Default is 2.
        cross_attn_heads (int, optional): Number of heads in the cross-attention layers. Default is 8.
        cross_attn_dim_head (int, optional): Dimension of each head in the cross-attention layers. Default is 64.
        depth (int, optional): Depth of the bimodal encoder. Default is 3.
        dropout (float, optional): Dropout rate. Default is 0.1.
        emb_dropout (float, optional): Dropout rate for the embeddings. Default is 0.1.
        channels (int, optional): Number of channels in the input data. Default is 3.
        pool (str, optional): Pooling method ('mean' or 'cls'). Default is 'mean'.
        return_attn (bool, optional): Whether to return attention weights. Default is False.

    Attributes:
        wsi_embedder (TileEmbedder): Embeds the WSI data.
        rna_embedder (RNAEmbedder): Embeds the RNA data.
        bimodal_encoder (BiModalEncoder): Encodes the WSI and RNA data with cross-attention.
        wsi_mlp_head (nn.Sequential): MLP head for the WSI data.
        rna_mlp_head (nn.Sequential): MLP head for the RNA data.
        pool (str): Pooling method.
        return_attn (bool): Whether to return attention weights.

    Methods:
        forward(wsi_bag, rna):
            Forward pass of the model.

            Args:
                wsi_bag (torch.Tensor): Input WSI data.
                rna (torch.Tensor): Input RNA data.

            Returns:
                Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
                    Logits and optionally the class tokens.
    """
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
    print(pred[0].shape)
