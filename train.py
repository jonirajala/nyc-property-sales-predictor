import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from utils import load_df, RealEstateDataset
from model import MLP

PATH = "data"
FILENAME = "ready_data.csv"


def train(df, device):
    epochs = 100
    lr = 0.01
     
    bs = 32
    test_size, val_size = 0.15, 0.15
    train_df, test_df = train_test_split(df, test_size=test_size)
    train_df, val_df = train_test_split(train_df, test_size=val_size)
    
    train_dl = DataLoader(RealEstateDataset(train_df), batch_size=bs)
    val_dl = DataLoader(RealEstateDataset(val_df), batch_size=bs)
    test_dl = DataLoader(RealEstateDataset(test_df), batch_size=bs)

    input_size = len(train_df.iloc[0]) - 1
    net = MLP(input_size)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.1)

    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        print(f"Epoch {epoch}")
        for price, features in tqdm(train_dl):
            price = price.to(device).float()
            features = features.to(device).float()
            optim.zero_grad()
            out = net(features)
            loss_value = loss(out, price)
            loss_value.backward()
            optim.step()
            epoch_loss += loss_value.item()
        train_loss = epoch_loss / len(train_dl)
        
        # Val loss
        val_loss = 0
        with torch.no_grad():
            for price, features in val_dl:
                price = price.to(device).float()
                features = features.to(device).float()
                out = net(features.float())
                loss_value = loss(out, price)
                val_loss += loss_value.item()
        val_loss = val_loss / len(val_dl)
        
        scheduler.step()
        print(f"Train loss: {train_loss}, Val loss: {val_loss}")

        val_losses.append(val_loss)
        train_losses.append(train_loss)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.show()

if __name__ == "__main__":
    df = load_df(PATH, FILENAME)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    train(df, device)
