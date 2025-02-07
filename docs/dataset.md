<!-- markdownlint-disable -->

<a href="../src/med_crossvit/dataset.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `dataset`





---

<a href="../src/med_crossvit/dataset.py#L155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `normalize_rna`

```python
normalize_rna(train_df, val_df, test_df)
```

Normalize RNA columns in the given dataframes. 

This function performs the following steps: 1. Identifies columns in the dataframes that contain 'rna_' in their names. 2. Applies a log2 transformation to these columns. 3. Standardizes these columns using a StandardScaler. 4. Saves the fitted scaler to a file named 'standard_scaler.joblib'. 



**Args:**
 
 - <b>`train_df`</b> (pd.DataFrame):  The training dataframe containing RNA columns. 
 - <b>`val_df`</b> (pd.DataFrame):  The validation dataframe containing RNA columns. 
 - <b>`test_df`</b> (pd.DataFrame):  The test dataframe containing RNA columns. 



**Returns:**
 
 - <b>`tuple`</b>:  A tuple containing the normalized training, validation, and test dataframes. 


---

<a href="../src/med_crossvit/dataset.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FeatureBagRNADataset`
A PyTorch Dataset class for handling RNA feature bags from a CSV file and corresponding HDF5 files. 



**Args:**
 
 - <b>`csv_path`</b> (str or pd.DataFrame):  Path to the CSV file or a pandas DataFrame containing metadata. 
 - <b>`bag_size`</b> (int, optional):  Number of patches per bag. Default is 40. 
 - <b>`max_patches_total`</b> (int, optional):  Maximum number of patches to select from each WSI. Default is 300. 
 - <b>`quick`</b> (bool, optional):  If True, sample a small subset of the CSV file for quick testing. Default is False. 
 - <b>`label_encoder`</b> (sklearn.preprocessing.LabelEncoder, optional):  Label encoder for transforming labels. Default is None. 
 - <b>`return_ids`</b> (bool, optional):  If True, return patient IDs along with the data. Default is False. 
 - <b>`feature_path`</b> (str, optional):  Path to the directory containing HDF5 feature files. Default is an empty string. 

Methods: _preprocess():  Preprocesses the CSV file and loads the data into memory. 

shuffle():  Shuffles the images within each WSI. 

__len__():  Returns the number of bags in the dataset. 

__getitem__(idx):  Retrieves the bag of features, RNA data, and label at the specified index. 



**Returns:**
 None 

<a href="../src/med_crossvit/dataset.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    csv_path,
    bag_size=40,
    max_patches_total=300,
    quick=False,
    label_encoder=None,
    return_ids=False,
    feature_path=''
)
```








---

<a href="../src/med_crossvit/dataset.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `shuffle`

```python
shuffle()
```

Shuffles the images within each WSI. 



**Args:**
  None 



**Returns:**
  None 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
