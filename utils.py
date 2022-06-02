from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
import os

class RealEstateDataset(Dataset):
    def __init__(self, df, transform=None):
        self.prices, self.features = self.normalize(df)
        self.transform = transform

    def normalize(self, df):
        scaler = MinMaxScaler()
        prices = torch.tensor(df['SALE PRICE'].values)
        features = torch.tensor(scaler.fit_transform(df.drop('SALE PRICE', axis=1)))
        return prices, features

    def __len__(self):
        return len(self.prices)

    def __getitem__(self, idx):
        price = self.prices[idx]
        features = self.features[idx]

        return price, features

def load_df(path, filename):
    df = pd.read_csv(os.path.join(path, filename))
    return df

if __name__ == "__main__":
    df = load_df("data", "ready_data.csv")
    ds = RealEstateDataset(df)
    p, f = ds[0]

    print(p)
    print(f)