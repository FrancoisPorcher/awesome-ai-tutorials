import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


class WineDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float), torch.tensor(
            self.targets[idx], dtype=torch.long
        )


class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(13, 64)
        self.fc2 = torch.nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Trainer:
    def __init__(self, model, train_data, optimizer, gpu_id, save_every):
        self.model = model
        self.train_data = train_data
        self.optimizer = optimizer
        self.gpu_id = gpu_id
        self.save_every = save_every
        self.losses = []

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        total_loss = 0.0
        num_batches = len(self.train_data)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss = self._run_batch(source, targets)
            total_loss += loss

        avg_loss = total_loss / num_batches
        self.losses.append(avg_loss)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    def _save_checkpoint(self, epoch):
        checkpoint = self.model.state_dict()
        PATH = f"model_{epoch}.pt"
        torch.save(checkpoint, PATH)
        print(f"Epoch {epoch} | Model saved to {PATH}")

    def train(self, max_epochs):
        self.model.train()
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    wine_data = load_wine()
    X = wine_data.data
    y = wine_data.target

    # Normalize and split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    train_set = WineDataset(X_train, y_train)
    test_set = WineDataset(X_test, y_test)

    print("Sample from dataset:")
    sample_data, sample_target = train_set[0]
    print(f"Data: {sample_data}")
    print(f"Target: {sample_target}")

    model = SimpleNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return train_set, model, optimizer


def prepare_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=True)


def main(device, total_epochs, save_every, batch_size):
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, device, save_every)
    trainer.train(total_epochs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Training a model on single node single GPU"
    )
    parser.add_argument(
        "total_epochs", type=int, help="Total number of epochs to train"
    )
    parser.add_argument("save_every", type=int, help="Save model every n epochs")
    parser.add_argument("batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(device, args.total_epochs, args.save_every, args.batch_size)
