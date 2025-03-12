import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T

from tqdm import tqdm
from typing import Optional, Tuple
from sklearn import metrics

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau, OneCycleLR

from .checkpointing import save_checkpoint, load_checkpoint

class Trainer:
    def __init__(self, save_dir: str = 'checkpoints', save_interval: int=10, device: torch.device = 'cpu', unsupervised_learning: bool=False):
        self.save_dir = save_dir
        self.device = device
        self.save_interval = save_interval
        self.train_autoencoder = unsupervised_learning

    def train(self, num_epochs: int, model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader,
              optimizer: optim.Optimizer, criterion: nn.Module, scheduler: Optional[LRScheduler]=None, resume: Optional[str]=None, accumulation_steps: int=1):
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
            print(f'\n=> resuming from checkpoint {resume}')

            model, optimizer, scheduler, start_epoch, train_losses, val_losses, accuracies = load_checkpoint(resume, model, optimizer, scheduler)
            start_epoch = start_epoch + 1
        else:
            train_losses, val_losses, accuracies = [], [], []
            start_epoch = 0

            print(f'\n=> Initial testing of the model')
            val_loss, accuracy = self._validation_loop(0, model, validation_loader, criterion)
            val_losses.append(val_loss)
            accuracies.append(accuracy)


        print(f'=> Starting training for {num_epochs} epochs', f'starting from {start_epoch}' if start_epoch > 0 else '')
        for epoch in range(start_epoch, num_epochs):
            train_loss = self._training_loop(epoch, model, train_loader, optimizer, scheduler, criterion, accumulation_steps)
            train_losses.append(train_loss)

            val_loss, accuracy = self._validation_loop(epoch, model, validation_loader, criterion)
            val_losses.append(val_loss)
            accuracies.append(accuracy)

            # Step with scheduler
            if scheduler is not None and not isinstance(scheduler, OneCycleLR):
                scheduler.step(val_loss) if isinstance(scheduler, ReduceLROnPlateau) else scheduler.step()

            # Save in intervals
            if (epoch + 1) % self.save_interval == 0:
                checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}_losses_{train_loss:.5f}_{val_loss:.5f}_acc_{accuracy}.pth')
                save_checkpoint(checkpoint_path, epoch, train_losses, val_losses, accuracies, model, optimizer, scheduler)

            # Save best val-loss model
            if val_loss <= min(val_losses):
                checkpoint_path = os.path.join(self.save_dir, f'{epoch}_best_val_model.pth')
                save_checkpoint(checkpoint_path, epoch, train_losses, val_losses, accuracies, model, optimizer, scheduler)

            # Save best accuracy model
            if accuracy >= max(accuracies):
                checkpoint_path = os.path.join(self.save_dir, f'{epoch}_best_acc_model.pth')
                save_checkpoint(checkpoint_path, epoch, train_losses, val_losses, accuracies, model, optimizer, scheduler)

        # save final model
        checkpoint_path = os.path.join(self.save_dir, f'final_model.pth')
        save_checkpoint(checkpoint_path, num_epochs, train_losses, val_losses, accuracies, model, optimizer, scheduler)

    def train_cross_validation(self, fold: int, num_epochs: int, model: nn.Module, dataloader: DataLoader,
              optimizer: optim.Optimizer, criterion: nn.Module, scheduler: Optional[LRScheduler]=None, resume: Optional[str]=None, accumulation_steps: int=1):

        model.to(self.device)
        if resume is not None:
            print(f'\n=> resuming from checkpoint {resume}')

            model, optimizer, scheduler, start_epoch, train_losses, val_losses, accuracies = load_checkpoint(resume, model, optimizer, scheduler)
            start_epoch = start_epoch + 1
        else:
            train_losses, val_losses, accuracies = [], [], []
            start_epoch = 0

            print(f'\n=> Fold {fold} Initial testing of the model')
            dataloader.dataset.cv_validation_mode()
            val_loss, accuracy = self._validation_loop(0, model, dataloader, criterion)
            val_losses.append(val_loss)
            accuracies.append(accuracy)


        print(f'=> Fold {fold} - Starting training for {num_epochs} epochs', f'starting from {start_epoch}' if start_epoch > 0 else '')
        for epoch in range(start_epoch, num_epochs):
            dataloader.dataset.cv_train_mode()
            train_loss = self._training_loop(epoch, model, dataloader, optimizer, scheduler, criterion, accumulation_steps)
            train_losses.append(train_loss)

            dataloader.dataset.cv_validation_mode()
            val_loss, accuracy = self._validation_loop(epoch, model, dataloader, criterion)
            val_losses.append(val_loss)
            accuracies.append(accuracy)

            # Step with scheduler
            if scheduler is not None and not isinstance(scheduler, OneCycleLR):
                scheduler.step(val_loss) if isinstance(scheduler, ReduceLROnPlateau) else scheduler.step()

            # Save in intervals
            if (epoch + 1) % self.save_interval == 0:
                checkpoint_path = os.path.join(self.save_dir, f'f{fold}_checkpoint_epoch_{epoch}_losses_{train_loss:.5f}_{val_loss:.5f}_acc_{accuracy}.pth')
                save_checkpoint(checkpoint_path, epoch, train_losses, val_losses, accuracies, model, optimizer, scheduler)

            # Save best val-loss model
            if val_loss <= min(val_losses):
                checkpoint_path = os.path.join(self.save_dir, f'f{fold}_{epoch}_best_val_model.pth')
                save_checkpoint(checkpoint_path, epoch, train_losses, val_losses, accuracies, model, optimizer, scheduler)

            # Save best accuracy model
            if accuracy >= max(accuracies) and not self.train_autoencoder:
                checkpoint_path = os.path.join(self.save_dir, f'f{fold}_{epoch}_best_acc_model.pth')
                save_checkpoint(checkpoint_path, epoch, train_losses, val_losses, accuracies, model, optimizer, scheduler)

        # save final model
        checkpoint_path = os.path.join(self.save_dir, f'f{fold}_{val_losses[-1]:.5f}_final_model.pth')
        save_checkpoint(checkpoint_path, num_epochs, train_losses, val_losses, accuracies, model, optimizer, scheduler)


    def _training_loop(self, epoch: int, model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, scheduler: Optional[LRScheduler], criterion: nn.Module, accumulation_steps: int) -> float:
        model.train()
        losses = []
        optimizer.zero_grad()  # through changed weight-update logic, call zero_grad before training to make sure we have no left-over gradients

        progress_bar = tqdm(train_loader)
        for idx, (inputs, labels) in enumerate(progress_bar, 1):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = model(inputs)
            loss = criterion(outputs, inputs) if self.train_autoencoder else criterion(outputs, labels)

            # Compute mean loss adjusted to accumulation_steps and backprop.
            steps_to_accumulate = accumulation_steps if idx <= len(train_loader) // accumulation_steps * accumulation_steps else len(train_loader) % accumulation_steps  # account for incomplete batches at the end
            mean_loss = loss.mean() / steps_to_accumulate
            mean_loss.backward()

            # Accumulate gradients until we either reached the end of the train_loader or we hit an accumulation-step intervall
            if idx % accumulation_steps == 0 or idx == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

                if isinstance(scheduler, OneCycleLR):
                    scheduler.step()

            losses.append(loss.detach().cpu().numpy())
            progress_bar.set_description(f'Epoch {epoch+1:02d} - Training loss: {np.stack(losses).mean():.5f}')

        losses = np.stack(losses)
        return losses.mean()


    @torch.no_grad()
    def _validation_loop(self, epoch: int, model: nn.Module, validation_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        model.eval()
        losses, targets, predictions = [], [], []

        progress_bar = tqdm(validation_loader)
        for idx, (inputs, labels) in enumerate(progress_bar, 1):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = model(inputs)
            loss = criterion(outputs, inputs) if self.train_autoencoder else criterion(outputs, labels)
            losses.append(loss.cpu().numpy())

            if self.train_autoencoder:
                progress_bar.set_description(f'-- Validation loss: {np.stack(losses).mean():.5f}')
            else:
                targets.append(labels.cpu().numpy())
                predictions.append(outputs.float().cpu().numpy())

                accuracy = metrics.accuracy_score(np.concatenate(targets).argmax(axis=1), np.concatenate(predictions).argmax(axis=1))
                progress_bar.set_description(f'-- Validation loss: {np.stack(losses).mean():.5f} | Accuracy: {accuracy:.4f}%')

        losses = np.stack(losses)

        if self.train_autoencoder:
            return losses.mean(), 0.0
        else:
            final_accuracy = metrics.accuracy_score(np.concatenate(targets).argmax(axis=1), np.concatenate(predictions).argmax(axis=1))
            return losses.mean(), final_accuracy