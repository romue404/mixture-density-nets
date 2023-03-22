from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from mixture_density_nets import MDN, MDDistribution
from tqdm import trange


if __name__ == '__main__':
    scaler = StandardScaler()
    data, labels = make_moons(n_samples=6000, noise=0.1)
    data = scaler.fit_transform(data)
    test_data, test_labels = make_moons(n_samples=10000, noise=0.1)
    test_data = scaler.transform(test_data)

    df = pd.DataFrame({'x': data[:, 0], 'y': data[:, 1], 'label': labels})
    sns.scatterplot(data=df, x='x', y='y', hue='label', palette='husl')
    plt.savefig('example.png')
    plt.close()

    ds = TensorDataset(torch.from_numpy(data).float())
    dl = DataLoader(ds, batch_size=128, num_workers=3, shuffle=True)

    class Net(nn.Module):
        def __init__(self, in_dim=1, out_dim=1, n_components=8):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.Dropout(0.2),
                nn.ELU(),
                nn.Linear(64, 64),
                nn.Dropout(0.2),
                nn.ELU()
            )
            self.mdn = MDN(64, out_dim, n_components)

        def forward(self, x):
            return self.mdn(self.net(x))

    net = Net(in_dim=1, out_dim=1, n_components=2)
    opt = torch.optim.AdamW(net.parameters())
    n_epochs = 128

    for epoch in trange(n_epochs):
        epoch_loss = []
        for batch in dl:
            xy = batch[0]
            x, y = xy[:, 0].unsqueeze(-1), xy[:, 1].unsqueeze(-1)

            mu, sigma, lambda_ = net(x)
            dist = MDDistribution(mu, sigma, lambda_)
            loss = dist.nll(y).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss.append(loss.item())

        print(f'NLL in epoch {epoch+1}: {torch.tensor(epoch_loss).mean():.4f}')

        with torch.inference_mode():
            net.eval()
            out = net(torch.from_numpy(test_data[:, 0]).float().unsqueeze(-1))
            y, l = MDDistribution(*out).sample()
            net.train()

        df = pd.DataFrame({'x': test_data[:, 0], 'y': y.numpy(), 'label': l.numpy()})
        sns.scatterplot(data=df, x='x', y='y', hue='label', palette='husl')
        plt.savefig(f'predictions/predictions_{epoch}.png')
        plt.close()
