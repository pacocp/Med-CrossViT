<!-- markdownlint-disable -->

<a href="../src/med_crossvit/med_crossvit.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `med_crossvit`





---

<a href="../src/med_crossvit/med_crossvit.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `exists`

```python
exists(val)
```

Check if a value is not None. 



**Args:**
 
 - <b>`val`</b>:  The value to check. 



**Returns:**
 
 - <b>`bool`</b>:  True if the value is not None, False otherwise. 


---

<a href="../src/med_crossvit/med_crossvit.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `default`

```python
default(val, d)
```

Returns the value `val` if it exists, otherwise returns the default value `d`. 



**Args:**
 
 - <b>`val`</b>:  The value to check for existence. 
 - <b>`d`</b>:  The default value to return if `val` does not exist. 



**Returns:**
 The value `val` if it exists, otherwise the default value `d`. 


---

<a href="../src/med_crossvit/med_crossvit.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FeedForward`
FeedForward neural network module. 

This module implements a simple feedforward neural network with the following layers: 
- Layer normalization 
- Linear transformation 
- GELU activation 
- Dropout 
- Linear transformation 
- Dropout 



**Args:**
 
 - <b>`dim`</b> (int):  The input and output dimension of the feedforward network. 
 - <b>`hidden_dim`</b> (int):  The hidden dimension of the feedforward network. 
 - <b>`dropout`</b> (float, optional):  The dropout rate. Default is 0. 

Methods: forward(x):  Forward pass through the feedforward network. 



**Args:**
 
         - <b>`x`</b> (torch.Tensor):  Input tensor. 



**Returns:**
 
         - <b>`torch.Tensor`</b>:  Output tensor after passing through the feedforward network. 

<a href="../src/med_crossvit/med_crossvit.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dim, hidden_dim, dropout=0.0)
```








---

<a href="../src/med_crossvit/med_crossvit.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```






---

<a href="../src/med_crossvit/med_crossvit.py#L79"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Attention`
Attention mechanism for neural networks. 



**Args:**
 
 - <b>`dim`</b> (int):  Dimension of the input features. 
 - <b>`heads`</b> (int, optional):  Number of attention heads. Default is 8. 
 - <b>`dim_head`</b> (int, optional):  Dimension of each attention head. Default is 64. 
 - <b>`dropout`</b> (float, optional):  Dropout rate. Default is 0. 

Methods: forward(x, context=None, kv_include_self=False):  Forward pass of the attention mechanism. 



**Args:**
 
         - <b>`x`</b> (torch.Tensor):  Input tensor of shape (batch_size, sequence_length, feature_dim). 
         - <b>`context`</b> (torch.Tensor, optional):  Context tensor for cross-attention. Default is None. 
         - <b>`kv_include_self`</b> (bool, optional):  Whether to include the input tensor itself as key/value. Default is False. 



**Returns:**
 
         - <b>`torch.Tensor`</b>:  Output tensor after applying attention mechanism. 

<a href="../src/med_crossvit/med_crossvit.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dim, heads=8, dim_head=64, dropout=0.0)
```








---

<a href="../src/med_crossvit/med_crossvit.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x, context=None, kv_include_self=False)
```






---

<a href="../src/med_crossvit/med_crossvit.py#L143"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Transformer`
A Transformer model consisting of multiple layers of attention and feed-forward networks. 



**Args:**
 
 - <b>`dim`</b> (int):  The dimension of the input and output features. 
 - <b>`depth`</b> (int):  The number of layers in the transformer. 
 - <b>`heads`</b> (int):  The number of attention heads. 
 - <b>`dim_head`</b> (int):  The dimension of each attention head. 
 - <b>`mlp_dim`</b> (int):  The dimension of the feed-forward network. 
 - <b>`dropout`</b> (float, optional):  The dropout rate. Default is 0. 



**Attributes:**
 
 - <b>`layers`</b> (nn.ModuleList):  A list of layers, each containing an attention mechanism and a feed-forward network. 
 - <b>`norm`</b> (nn.LayerNorm):  A layer normalization applied to the output. 

Methods: forward(x):  Passes the input through the transformer layers and applies layer normalization. 



**Args:**
 
         - <b>`x`</b> (torch.Tensor):  The input tensor. 



**Returns:**
 
         - <b>`torch.Tensor`</b>:  The output tensor after passing through the transformer layers and normalization. 

<a href="../src/med_crossvit/med_crossvit.py#L169"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dim, depth, heads, dim_head, mlp_dim, dropout=0.0)
```








---

<a href="../src/med_crossvit/med_crossvit.py#L179"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```






---

<a href="../src/med_crossvit/med_crossvit.py#L188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ProjectInOut`
A neural network module that conditionally projects input and output dimensions to match the required dimensions for a given function. 



**Attributes:**
 
 - <b>`fn`</b> (callable):  The function to be applied to the input tensor. 
 - <b>`project_in`</b> (nn.Module):  A linear layer to project the input tensor to the required dimension,  or an identity layer if no projection is needed. 
 - <b>`project_out`</b> (nn.Module):  A linear layer to project the output tensor back to the original dimension,  or an identity layer if no projection is needed. 



**Args:**
 
 - <b>`dim_in`</b> (int):  The dimension of the input tensor. 
 - <b>`dim_out`</b> (int):  The dimension required by the function `fn`. 
 - <b>`fn`</b> (callable):  The function to be applied to the input tensor. 

Methods: forward(x, *args, **kwargs):  Applies the input projection, then the function `fn`, and finally the output projection. 

**Args:**
 
         - <b>`x`</b> (torch.Tensor):  The input tensor. 
         - <b>`*args`</b>:  Additional positional arguments to be passed to `fn`. 
         - <b>`**kwargs`</b>:  Additional keyword arguments to be passed to `fn`. 

**Returns:**
 
         - <b>`torch.Tensor`</b>:  The output tensor after applying the projections and the function `fn`. 

<a href="../src/med_crossvit/med_crossvit.py#L215"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dim_in, dim_out, fn)
```








---

<a href="../src/med_crossvit/med_crossvit.py#L223"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x, *args, **kwargs)
```






---

<a href="../src/med_crossvit/med_crossvit.py#L232"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CrossTransformer`
A CrossTransformer module that performs cross-attention between WSI (Whole Slide Image) tokens and RNA tokens. 



**Args:**
 
 - <b>`wsi_dim`</b> (int):  Dimension of the WSI tokens. 
 - <b>`rna_dim`</b> (int):  Dimension of the RNA tokens. 
 - <b>`depth`</b> (int):  Number of cross-attention layers. 
 - <b>`heads`</b> (int):  Number of attention heads. 
 - <b>`dim_head`</b> (int):  Dimension of each attention head. 
 - <b>`dropout`</b> (float, optional):  Dropout rate. Default is 0. 

Methods: forward(wsi_tokens, rna_tokens):  Performs the forward pass of the CrossTransformer.  



**Args:**
 
         - <b>`wsi_tokens`</b> (torch.Tensor):  Input tokens for WSI, shape (batch_size, seq_len, wsi_dim). 
         - <b>`rna_tokens`</b> (torch.Tensor):  Input tokens for RNA, shape (batch_size, seq_len, rna_dim). 



**Returns:**
 
         - <b>`Tuple[torch.Tensor, torch.Tensor]`</b>:  The transformed WSI and RNA tokens. 

<a href="../src/med_crossvit/med_crossvit.py#L255"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(wsi_dim, rna_dim, depth, heads, dim_head, dropout=0.0)
```








---

<a href="../src/med_crossvit/med_crossvit.py#L263"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(wsi_tokens, rna_tokens)
```






---

<a href="../src/med_crossvit/med_crossvit.py#L278"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BiModalEncoder`
BiModalEncoder is a neural network module designed to encode two different modalities of data  (e.g., whole slide images (WSI) and RNA sequences) using separate transformers for each modality  and a cross-transformer to facilitate interaction between the two modalities. 



**Args:**
 
 - <b>`depth`</b> (int):  Number of layers in the encoder. 
 - <b>`wsi_dim`</b> (int):  Dimensionality of the WSI tokens. 
 - <b>`rna_dim`</b> (int):  Dimensionality of the RNA tokens. 
 - <b>`wsi_enc_params`</b> (dict):  Parameters for the WSI transformer encoder. 
 - <b>`rna_enc_params`</b> (dict):  Parameters for the RNA transformer encoder. 
 - <b>`cross_attn_heads`</b> (int):  Number of attention heads in the cross-attention mechanism. 
 - <b>`cross_attn_depth`</b> (int):  Depth of the cross-attention mechanism. 
 - <b>`cross_attn_dim_head`</b> (int, optional):  Dimensionality of each attention head in the cross-attention mechanism. Default is 64. 
 - <b>`dropout`</b> (float, optional):  Dropout rate. Default is 0. 

Methods: forward(wsi_tokens, rna_tokens):  Forward pass through the BiModalEncoder.  



**Args:**
 
         - <b>`wsi_tokens`</b> (torch.Tensor):  Input tokens for the WSI modality. 
         - <b>`rna_tokens`</b> (torch.Tensor):  Input tokens for the RNA modality. 



**Returns:**
 
         - <b>`Tuple[torch.Tensor, torch.Tensor]`</b>:  Encoded WSI and RNA tokens after processing through the encoder. 

<a href="../src/med_crossvit/med_crossvit.py#L306"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    depth,
    wsi_dim,
    rna_dim,
    wsi_enc_params,
    rna_enc_params,
    cross_attn_heads,
    cross_attn_depth,
    cross_attn_dim_head=64,
    dropout=0.0
)
```








---

<a href="../src/med_crossvit/med_crossvit.py#L328"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(wsi_tokens, rna_tokens)
```






---

<a href="../src/med_crossvit/med_crossvit.py#L337"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TileEmbedder`
TileEmbedder is a neural network module that embeds input tiles into a higher-dimensional space and adds positional embeddings and a class token. 



**Args:**
 
 - <b>`dim`</b> (int):  The dimension of the embedding space. 
 - <b>`num_tiles`</b> (int):  The number of tiles to embed. 
 - <b>`dropout`</b> (float, optional):  Dropout rate. Default is 0. 
 - <b>`channels`</b> (float, optional):  Number of input channels. Default is 3. 
 - <b>`feature_extractor`</b> (nn.Module, optional):  A feature extractor module. Default is None. 



**Attributes:**
 
 - <b>`feature_extractor`</b> (nn.Module):  The feature extractor module if provided. 
 - <b>`pos_embedding`</b> (nn.Parameter):  Positional embeddings for the tiles. 
 - <b>`cls_token`</b> (nn.Parameter):  Class token to be prepended to the input. 
 - <b>`dropout`</b> (nn.Dropout):  Dropout layer. 

Methods: forward(x):  Forward pass of the TileEmbedder. Adds class token and positional embeddings to the input,  then applies dropout. 



**Args:**
 
         - <b>`x`</b> (torch.Tensor):  Input tensor of shape (batch_size, num_tiles, dim). 



**Returns:**
 
         - <b>`torch.Tensor`</b>:  Output tensor after adding class token, positional embeddings, and applying dropout. 

<a href="../src/med_crossvit/med_crossvit.py#L366"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dim, num_tiles, dropout=0.0, channels=3.0, feature_extractor=None)
```








---

<a href="../src/med_crossvit/med_crossvit.py#L383"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```






---

<a href="../src/med_crossvit/med_crossvit.py#L392"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RNAEmbedder`
RNAEmbedder is a neural network module designed to embed RNA sequences into a high-dimensional space. 



**Args:**
 
 - <b>`dim`</b> (int):  The dimensionality of the embedding space. 
 - <b>`num_genes`</b> (int):  The number of genes in the RNA sequence. 
 - <b>`dropout`</b> (float, optional):  Dropout rate to apply after embedding. Default is 0.0. 



**Attributes:**
 
 - <b>`to_rna_embedding`</b> (nn.Sequential):  A sequential container to normalize, linearly transform, and rearrange the RNA sequence. 
 - <b>`pos_embedding`</b> (nn.Parameter):  Positional embeddings for the RNA sequence. 
 - <b>`cls_token`</b> (nn.Parameter):  A classification token added to the beginning of the sequence. 
 - <b>`dropout`</b> (nn.Dropout):  Dropout layer applied to the final output. 

Methods: forward(x):  Forward pass of the RNAEmbedder.  



**Args:**
 
         - <b>`x`</b> (torch.Tensor):  Input tensor of shape (batch_size, num_genes). 



**Returns:**
 
         - <b>`torch.Tensor`</b>:  Output tensor of shape (batch_size, num_genes + 1, dim) after embedding, positional encoding, and dropout. 

<a href="../src/med_crossvit/med_crossvit.py#L417"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dim, num_genes, dropout=0.0)
```








---

<a href="../src/med_crossvit/med_crossvit.py#L435"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```






---

<a href="../src/med_crossvit/med_crossvit.py#L445"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MedCrossViT`
Med-CrossViT: A WSI-RNA-Seq fusion model. 



**Args:**
 
 - <b>`num_classes`</b> (int):  Number of output classes. 
 - <b>`wsi_dim`</b> (int):  Dimension of the whole slide image (WSI) embeddings. 
 - <b>`rna_dim`</b> (int):  Dimension of the RNA embeddings. 
 - <b>`wsi_num_tiles`</b> (int, optional):  Number of tiles in the WSI. Default is 50. 
 - <b>`wsi_enc_depth`</b> (int, optional):  Depth of the WSI encoder. Default is 4. 
 - <b>`wsi_enc_heads`</b> (int, optional):  Number of heads in the WSI encoder. Default is 8. 
 - <b>`wsi_enc_mlp_dim`</b> (int, optional):  MLP dimension in the WSI encoder. Default is 2048. 
 - <b>`wsi_enc_dim_head`</b> (int, optional):  Dimension of each head in the WSI encoder. Default is 64. 
 - <b>`rna_num_genes`</b> (int, optional):  Number of genes in the RNA data. Default is 100. 
 - <b>`rna_enc_depth`</b> (int, optional):  Depth of the RNA encoder. Default is 4. 
 - <b>`rna_enc_heads`</b> (int, optional):  Number of heads in the RNA encoder. Default is 8. 
 - <b>`rna_enc_mlp_dim`</b> (int, optional):  MLP dimension in the RNA encoder. Default is 2048. 
 - <b>`rna_enc_dim_head`</b> (int, optional):  Dimension of each head in the RNA encoder. Default is 64. 
 - <b>`cross_atnn_depth`</b> (int, optional):  Depth of the cross-attention layers. Default is 2. 
 - <b>`cross_attn_heads`</b> (int, optional):  Number of heads in the cross-attention layers. Default is 8. 
 - <b>`cross_attn_dim_head`</b> (int, optional):  Dimension of each head in the cross-attention layers. Default is 64. 
 - <b>`depth`</b> (int, optional):  Depth of the bimodal encoder. Default is 3. 
 - <b>`dropout`</b> (float, optional):  Dropout rate. Default is 0.1. 
 - <b>`emb_dropout`</b> (float, optional):  Dropout rate for the embeddings. Default is 0.1. 
 - <b>`channels`</b> (int, optional):  Number of channels in the input data. Default is 3. 
 - <b>`pool`</b> (str, optional):  Pooling method ('mean' or 'cls'). Default is 'mean'. 
 - <b>`return_attn`</b> (bool, optional):  Whether to return attention weights. Default is False. 



**Attributes:**
 
 - <b>`wsi_embedder`</b> (TileEmbedder):  Embeds the WSI data. 
 - <b>`rna_embedder`</b> (RNAEmbedder):  Embeds the RNA data. 
 - <b>`bimodal_encoder`</b> (BiModalEncoder):  Encodes the WSI and RNA data with cross-attention. 
 - <b>`wsi_mlp_head`</b> (nn.Sequential):  MLP head for the WSI data. 
 - <b>`rna_mlp_head`</b> (nn.Sequential):  MLP head for the RNA data. 
 - <b>`pool`</b> (str):  Pooling method. 
 - <b>`return_attn`</b> (bool):  Whether to return attention weights. 

Methods: forward(wsi_bag, rna):  Forward pass of the model. 



**Args:**
 
         - <b>`wsi_bag`</b> (torch.Tensor):  Input WSI data. 
         - <b>`rna`</b> (torch.Tensor):  Input RNA data. 



**Returns:**
 Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:  Logits and optionally the class tokens. 

<a href="../src/med_crossvit/med_crossvit.py#L494"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
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
)
```








---

<a href="../src/med_crossvit/med_crossvit.py#L551"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(wsi_bag, rna)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
