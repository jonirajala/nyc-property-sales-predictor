from torch.utils.data import Dataset
import torch
import pandas as pd
import os

class RealEstateDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, transform=None):
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        price = torch.tensor(sample['SALE PRICE'])
        features = torch.tensor(sample.drop('SALE PRICE'))
        return price, features.squeeze()

def load_df(path, filename):
    df = pd.read_csv(os.path.join(path, filename))
    return df

if __name__ == "__main__":
    df = load_df("data", "ready_data.csv")
    ds = RealEstateDataset(df)
    p, f = ds[0]

    print(p)
    print(f)