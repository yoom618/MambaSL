# CHANGED: add data split code for UEA dataset, since the original repo doesn't provide the splited dataset
# CHANGED: Unlike the original paper's mentioned 8:2 split, we use the full training set to match with the time-series-library pipeline
import os

import numpy as np
import torch
from torch.utils.data import DataLoader


def normalize_time_series(data):
    mean = data.mean()
    std = data.std()
    normalized_data = (data - mean) / std
    return normalized_data


def zero_pad_sequence(input_tensor, pad_length):
    return torch.nn.functional.pad(input_tensor, (0, pad_length))


def calculate_padding(seq_len, patch_size):
    padding = patch_size - (seq_len % patch_size) if seq_len % patch_size != 0 else 0
    return padding


class Load_Dataset(torch.utils.data.Dataset):
    # Initialize your data, download, etc.
    def __init__(self, data_file):
        super(Load_Dataset, self).__init__()
        self.data_file = data_file

        # Load samples and labels
        x_data = data_file["samples"]  # dim: [#samples, #channels, Seq_len]

        # x_data = normalize_time_series(x_data)

        y_data = data_file.get("labels")
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data).squeeze()

        # Convert to torch tensor
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)

        # Check samples dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)

        self.x_data = x_data.float()
        self.y_data = y_data.long().squeeze() if y_data is not None else None

        self.len = x_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index] if self.y_data is not None else None
        return x, y

    def __len__(self):
        return self.len


class Load_Dataset_UEA30(torch.utils.data.Dataset):
    # Initialize your data, download, etc.
    def __init__(self, data, labels, seq_len, idx_list=None):
        super().__init__()
        self.IDs = idx_list if idx_list is not None else labels.index
        self.x_data = data
        self.y_data = labels.loc[self.IDs].values.copy()
        self.seq_len = seq_len
        self.feature_names = self.x_data.columns
        self.num_channels = len(self.feature_names)

    def __getitem__(self, index):
        x = torch.from_numpy(self.x_data.loc[self.IDs[index]].values.transpose()).float()
        x = torch.nn.functional.pad(x, (0, self.seq_len - x.shape[1]), value=0.0)  # Pad to seq_len
        y = torch.from_numpy(self.y_data[index]).long().squeeze()

        return x, y

    def __len__(self):
        return len(self.IDs)


def get_datasets(DATASET_PATH, args):
    ####### Original Code #######
    if args.data_type != 'uea':
        train_file = torch.load(os.path.join(DATASET_PATH, f"train.pt"))
        seq_len = train_file["samples"].shape[-1]
        required_padding = calculate_padding(seq_len, args.patch_size)

        val_file = torch.load(os.path.join(DATASET_PATH, f"val.pt"))
        test_file = torch.load(os.path.join(DATASET_PATH, f"test.pt"))

        if required_padding != 0:
            train_file["samples"] = zero_pad_sequence(train_file["samples"], required_padding)
            val_file["samples"] = zero_pad_sequence(val_file["samples"], required_padding)
            test_file["samples"] = zero_pad_sequence(test_file["samples"], required_padding)

        train_dataset = Load_Dataset(train_file)
        val_dataset = Load_Dataset(val_file)
        test_dataset = Load_Dataset(test_file)

    ############ Revised Code ###########
    # since train/val/test pt files are not provided in the original repo, we use the pkl files instead.
    # The pkl files are generated from the ts files using data_provider/data_loader.py's UEAloader class
    # (we revised the UEAloader class to save the data in pkl format since loading ts file takes a lot of time)
    else:
        import glob
        import pandas as pd
        
        train_path = f'{DATASET_PATH}/{args.data_name}_TRAIN__{{}}.{{}}'
        test_path = f'{DATASET_PATH}/{args.data_name}_TEST__{{}}.{{}}'

        df_path = glob.glob(train_path.format('df(*)', 'pkl'))[0]
        seq_len = int(df_path.split('(')[1].split(')')[0])
        seq_len = seq_len + calculate_padding(seq_len, args.patch_size)
        df_trainval = pd.read_pickle(df_path)
        labels_trainval = np.load(train_path.format('labels', 'npy'), allow_pickle=True)
        labels_trainval = pd.Series(labels_trainval, dtype="category")
        class_names = labels_trainval.cat.categories
        labels_trainval_df = pd.DataFrame(labels_trainval.cat.codes,
                                        dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss
        df_path = glob.glob(test_path.format('df(*)', 'pkl'))[0]
        df_test = pd.read_pickle(df_path)
        labels_test = np.load(test_path.format('labels', 'npy'), allow_pickle=True)
        labels_test = pd.Categorical(labels_test, categories=class_names)
        labels_test_df = pd.DataFrame(labels_test.codes,
                                dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        #################
        # # 1) (unused) stratified split - 8:2 for each class as much as possible
        # # the original paper mentioned that they used 8:2 split, but didn't provide the split
        # # thus, we tested stratified split, but the results were not good.
        # # so we used the full training set to match with the time-series-library pipeline
        # train_indices, val_indices = [], []
        # for class_idx in range(len(class_names)):
        #     class_indices = np.where(labels_trainval_df.values == class_idx)[0]
        #     np.random.shuffle(class_indices)
        #     split_index = int(len(class_indices) * 0.8)
        #     train_indices.extend(class_indices[:split_index])
        #     val_indices.extend(class_indices[split_index:])
        # while len(val_indices) > int(0.25 * len(train_indices)):
        #     tmp_idx = val_indices.pop(np.random.randint(0, len(val_indices) - 1))
        #     train_indices.append(tmp_idx)
        # train_dataset = Load_Dataset_UEA30(df_trainval, labels_trainval_df, seq_len, labels_trainval_df.index[train_indices])
        # val_dataset = Load_Dataset_UEA30(df_trainval, labels_trainval_df, seq_len, labels_trainval_df.index[val_indices])
        # test_dataset = Load_Dataset_UEA30(df_test, labels_test_df, seq_len)
        # num_channels = train_dataset.num_channels
        
        # 2) val = test
        # though the original paper used 8:2 split, we use full training set to match with the time-series-library pipeline
        
        train_dataset = Load_Dataset_UEA30(df_trainval, labels_trainval_df, seq_len)
        val_dataset = Load_Dataset_UEA30(df_test, labels_test_df, seq_len)
        test_dataset = Load_Dataset_UEA30(df_test, labels_test_df, seq_len)
        num_channels = train_dataset.num_channels

        #################
        print(f'Train : {len(train_dataset)}, Val : {len(val_dataset)}, Test : {len(test_dataset)}')

    # in case the dataset is too small ...
    num_samples = len(train_dataset)
    if num_samples < args.batch_size:
        batch_size = num_samples // 4
    else:
        batch_size = args.batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader, test_loader, class_names, seq_len, num_channels


