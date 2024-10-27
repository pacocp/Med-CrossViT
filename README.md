# Med-CrossViT: A WSI-RNA-Seq fusion model

In this repository I have implemented a medicine-focused [CrossViT](https://arxiv.org/abs/2103.14899). Instead of using two augmentations of the same image as input, this model uses two different modalities: bags digital pathology tiles features and RNA-Seq. This serves as a proof of concept for an intermediate fusion architecture.

# Installation

The package can be installed with the associated Makefile associated using [uv](https://docs.astral.sh/uv/):

```bash
make build
```

# Usage

An example training-testing pipeline can be found in the  ````src/main.py```` file. The model expects as input two modalities:
- Bag of tiles features: Torch tensor with the format (batch_size, bag_size, feature_dim).
- RNA-Seq counts: Torch tensor with the format (batch_size, num_genes).

An example of the code would be:

```python
from med_crossvit.med_crossvit import MedCrossViT

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

# get a lot of this
wsi_bag = torch.rand((16, 50, 768))
rna_seq = torch.rand((16, 100))

# train for multiple steps
pred = v(wsi_bag, rna_seq)
print(pred)
print(pred.shape)
```

A Torch dataset is implemented that returns for every sample the bag of tiles features and the RNA-Seq counts. It uses as input a csv wher each row is a sample, contains one column per gene with the **rna_** prefix, and the label. An example csv is provided in the ```data``` folder. Features per tile are expected to be computed beforehand. An example would be:

```python
train_dataset = FeatureBagRNADataset(csv_path=train_df, bag_size=50, max_patches_total=100, feature_path=args.feature_path)
train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=0, pin_memory=True)
# Do this for multiple epochs
for x in train_dataloader:
    wsi_bag, rna, y_true = x
    rna = rna.squeeze(1)
    logits, _ = model(wsi_bag, rna)
    loss = loss_fn(logits, y_true)
    loss.backward()
    optimizer.step()
```

## Acknowledgments

This implementation is heavily based in the Cross ViT implementation by [lucidrains](https://github.com/lucidrains). Thanks for all the awesome work!
