import os

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from typing import Optional

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from .checkpointing import save_checkpoint, load_checkpoint
from .number_of_correct import number_of_correct

class Trainer:
    def __init__(self, save_dir: str = 'checkpoints', save_interval: int=10, device: torch.device = 'cpu', unsupervised_learning=False):
        self.save_dir = save_dir
        self.device = device
        self.save_interval = save_interval
        self.train_autoencoder = unsupervised_learning

    def train(self, num_epochs: int, model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader,
              optimizer: optim.Optimizer, criterion: nn.Module, scheduler: Optional[LRScheduler]=None, resume: Optional[str]=None):
        """
        Trains the model on the specified training and validation sets for the given number of epochs.
        :param num_epochs: Number of epochs to train the model.
        :param model: Model to be trained.
        :param train_loader: Training data loader.
        :param validation_loader: Validation data loader.
        :param optimizer: Optimizer to be used.
        :param criterion: Loss function to be used.
        :param scheduler: Scheduler to be used (Optional).
        :param resume: Resume training from saved checkpoint.
        :return:
        """

        model.to(self.device)
        if resume is not None:
            print(f'=> resuming from checkpoint {resume}')

            model, optimizer, scheduler, start_epoch, train_losses, val_losses = load_checkpoint(resume, model, optimizer, scheduler)
            start_epoch = start_epoch + 1
        else:
            train_losses, val_losses = [], []
            start_epoch = 0

        print(f'=> Starting training for {num_epochs} epochs', f'starting from {start_epoch}' if start_epoch > 0 else '')
        for epoch in range(start_epoch, num_epochs):
            train_loss = self._training_loop(epoch, model, train_loader, optimizer, criterion, scheduler)
            train_losses.append(train_loss)

            val_loss = self._validation_loop(epoch, model, validation_loader, criterion)
            val_losses.append(val_loss)

            # Save in intervals
            if (epoch + 1) % self.save_interval == 0:
                checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}_losses_{train_loss:.4f}_{val_loss:.4f}.pth')
                save_checkpoint(checkpoint_path, epoch, train_losses, val_losses, model, optimizer, scheduler)

            # Save best model
            if val_loss <= min(val_losses):
                checkpoint_path = os.path.join(self.save_dir, f'best_model.pth')
                save_checkpoint(checkpoint_path, epoch, train_losses, val_losses, model, optimizer, scheduler)

        # save final model
        checkpoint_path = os.path.join(self.save_dir, f'final_model.pth')
        save_checkpoint(checkpoint_path, num_epochs, train_losses, val_losses, model, optimizer, scheduler)

    def _training_loop(self, epoch: int, model: nn.Module, train_loader: DataLoader,
                       optimizer: optim.Optimizer, criterion: nn.Module, scheduler: LRScheduler|None) -> float:
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

            progress_bar.set_description(f'Epoch {epoch+1:02d} - Training loss:   {(running_loss / idx):.4f}')

        if scheduler is not None:
            scheduler.step()

        return running_loss

    def _validation_loop(self, epoch: int, model: nn.Module, validation_loader: DataLoader, criterion: nn.Module) -> float:
        model.eval()
        running_loss = 0.0

        correct = 0
        num_samples = 0

        progress_bar = tqdm(validation_loader)
        for idx, (inputs, labels) in enumerate(progress_bar, 1):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = model(inputs)
            loss = criterion(outputs, inputs) if self.train_autoencoder else criterion(outputs, labels)
            running_loss += loss.item()

            if self.train_autoencoder:
                progress_bar.set_description(f'-- Validation loss: {(running_loss / idx):.4f}')
            else:
                correct += number_of_correct(predictions=outputs, targets=labels)
                num_samples += inputs.shape[0]
                progress_bar.set_description(f'-- Validation loss: {(running_loss / idx):.4f} | Accuracy: {correct}/{num_samples} ({100. * correct / num_samples:.0f}%)')

        return running_loss
