import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, save_dir: str = 'checkpoints', device: str = 'cpu', train_autoencoder=False):
        self.save_dir = save_dir
        self.device = device
        self.train_autoencoder = train_autoencoder

    def train(self, num_epochs: int, model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader,
              optimizer: optim.Optimizer, criterion: nn.Module):
        model.to(self.device)

        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            train_loss = self._training_loop(model, train_loader, optimizer, criterion)
            train_losses.append(train_loss)

            val_loss = self._validation_loop(model, validation_loader, criterion)
            val_losses.append(val_loss)

        print(train_losses)
        print(val_losses)

    def _training_loop(self, model: nn.Module, train_loader: DataLoader,
                       optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader)
        for idx, (inputs, labels) in enumerate(progress_bar, 1):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = model(inputs)
            loss = criterion(outputs, inputs) if self.train_autoencoder else criterion(outputs, labels)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f'Training loss:\t\t{(running_loss / idx):.4f}')

        return running_loss / len(train_loader)

    def _validation_loop(self, model: nn.Module, validation_loader: DataLoader, criterion: nn.Module) -> float:
        model.eval()
        running_loss = 0.0

        progress_bar = tqdm(validation_loader)
        for idx, (inputs, labels) in enumerate(progress_bar, 1):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = model(inputs)
            loss = criterion(outputs, inputs) if self.train_autoencoder else criterion(outputs, labels)
            running_loss += loss.item()

            progress_bar.set_description(f'Validation loss:\t{(running_loss / idx):.4f}')

        return running_loss / len(validation_loader)

    def save_backup(self):
        pass

    def load_backup(self):
        pass
