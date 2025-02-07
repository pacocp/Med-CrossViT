import os
import random
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
from tqdm import tqdm
import h5py
from sklearn.preprocessing import StandardScaler
import joblib

class FeatureBagRNADataset(Dataset):
    """
    A PyTorch Dataset class for handling RNA feature bags from a CSV file and corresponding HDF5 files.

    Args:
        csv_path (str or pd.DataFrame): Path to the CSV file or a pandas DataFrame containing metadata.
        bag_size (int, optional): Number of patches per bag. Default is 40.
        max_patches_total (int, optional): Maximum number of patches to select from each WSI. Default is 300.
        quick (bool, optional): If True, sample a small subset of the CSV file for quick testing. Default is False.
        label_encoder (sklearn.preprocessing.LabelEncoder, optional): Label encoder for transforming labels. Default is None.
        return_ids (bool, optional): If True, return patient IDs along with the data. Default is False.
        feature_path (str, optional): Path to the directory containing HDF5 feature files. Default is an empty string.

    Methods:
        _preprocess():
            Preprocesses the CSV file and loads the data into memory.

        shuffle():
            Shuffles the images within each WSI.

        __len__():
            Returns the number of bags in the dataset.

        __getitem__(idx):
            Retrieves the bag of features, RNA data, and label at the specified index.

    Returns:
        None
    """
    def __init__(self, csv_path, bag_size=40,
            max_patches_total=300, quick=False, label_encoder=None,
            return_ids = False, feature_path=''):
        self.csv_path = csv_path
        self.bag_size = bag_size
        self.max_patches_total = max_patches_total
        self.quick = quick
        self.le = label_encoder
        self.index = []
        self.data = {}
        self.return_ids = return_ids
        self.feature_path = feature_path
        self._preprocess()

    def _preprocess(self):
        """
        Preprocesses the CSV file and loads the data into memory.

        Args:
            None

        Returns:
            None
        """
        if type(self.csv_path) == str:
            self.csv_file = pd.read_csv(self.csv_path)
        else:
            self.csv_file = self.csv_path

        if self.quick:
            self.csv_file = csv_file.sample(15)

        self.rna_columns = [x for x in self.csv_file.columns if 'rna_' in x]
        for i, row in tqdm(self.csv_file.iterrows()):
            row = row.to_dict()
            WSI = row['wsi_file_name']
            project = row['project']
            label = np.asarray(row['label'])
            patient_id = row['patient_id']
            if self.le is not None:
                label = self.le.transform(label.ravel())


            path = os.path.join(self.feature_path, project, WSI+'.h5')
            if not os.path.exists(path):
                print(f'Not exist {path}')
                continue

            try:
                with h5py.File(path, 'r') as h5_file:
                    n_patches = len(h5_file['keys'][()])
            except Exception as e:
                print(e)
                continue
            n_selected = min(n_patches, self.max_patches_total)
            n_patches= list(range(n_selected))
            images = random.sample(n_patches, n_selected)
            self.data[WSI] = {w.lower(): row[w] for w in row.keys()}
            self.data[WSI].update({'WSI': WSI, 'images': images, 'n_images': len(images),
                                   'wsi_path': path, 'patient_id': patient_id})
            for k in range(len(images) // self.bag_size):
                self.index.append((WSI, path, self.bag_size * k, label))

    def shuffle(self):
        """
        Shuffles the images within each WSI.

        Args:
            None

        Returns:
            None
        """
        for k in self.data.keys():
            wsi_row = self.data[k]
            np.random.shuffle(wsi_row['images'])

    def __len__(self):
        """
        Returns the number of bags in the dataset.

        Args:
            None

        Returns:
            int: The number of bags in the dataset.
        """
        return len(self.index)

    def __getitem__(self, idx):
        """
        Retrieves the bag of features, RNA data, and label at the specified index.

        Args:
            idx (int): Index of the bag to retrieve.

        Returns:
            tuple: A tuple containing:
                - img (torch.Tensor): Tensor of image features.
                - rna_data (torch.Tensor): Tensor of RNA data.
                - label (torch.Tensor): Tensor of the label.
        """
        (WSI, wsi_path, i, label) = self.index[idx]
        features = []
        row = self.data[WSI]
        with h5py.File(wsi_path, 'r') as h5_file:
            imgs = [torch.from_numpy(h5_file['uni_features'][:][patch]) for patch in row['images'][i:i + self.bag_size]]

        img = torch.stack(imgs, dim=0)
        rna_data = self.csv_file.loc[self.csv_file['wsi_file_name'] == WSI][self.rna_columns].values
        rna_data = torch.tensor(rna_data, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return img, rna_data, label

def normalize_rna(train_df, val_df, test_df):
    """
    Normalize RNA columns in the given dataframes.

    This function performs the following steps:
    1. Identifies columns in the dataframes that contain 'rna_' in their names.
    2. Applies a log2 transformation to these columns.
    3. Standardizes these columns using a StandardScaler.
    4. Saves the fitted scaler to a file named 'standard_scaler.joblib'.

    Args:
        train_df (pd.DataFrame): The training dataframe containing RNA columns.
        val_df (pd.DataFrame): The validation dataframe containing RNA columns.
        test_df (pd.DataFrame): The test dataframe containing RNA columns.

    Returns:
        tuple: A tuple containing the normalized training, validation, and test dataframes.
    """
    rna_columns = [x for x in train_df.columns if 'rna_' in x]
    train_df[rna_columns] = np.log2(train_df[rna_columns].values + 1)
    val_df[rna_columns] = np.log2(val_df[rna_columns].values + 1)
    test_df[rna_columns] = np.log2(test_df[rna_columns].values + 1)

    scaler = StandardScaler()
    train_df[rna_columns] = scaler.fit_transform(train_df[rna_columns])
    val_df[rna_columns] = scaler.transform(val_df[rna_columns])
    test_df[rna_columns] = scaler.transform(test_df[rna_columns])
    joblib.dump(scaler, 'standard_scaler.joblib')

    return train_df, val_df, test_df
