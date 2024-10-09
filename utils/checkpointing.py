import torch
import torch.nn as nn

from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # Using torch.save with weights_only=False gives a warning, we ignore that for now

def save_checkpoint(path: str, epoch:int, training_losses: list[float], validation_losses: list[float], accuracies: list[float],
                    model: nn.Module, optimizer: Optimizer, scheduler: Optional[LRScheduler]=None):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'training_losses': training_losses,
        'validation_losses': validation_losses,
        'accuracies': accuracies,
    }, path)

def load_checkpoint(path: str, model: nn.Module, optimizer: Optimizer, scheduler: Optional[LRScheduler]=None) -> tuple[nn.Module, Optimizer, Optional[LRScheduler], int, list[float], list[float], list[float]]:
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    training_losses = checkpoint['training_losses']
    validation_losses = checkpoint['validation_losses']
    accuracies = checkpoint['accuracies']

    return model, optimizer, scheduler, epoch, training_losses, validation_losses, accuracies

def load_model(path:str, model: nn.Module, checkpoint_is_dict: bool=True) -> nn.Module:
    checkpoints = torch.load(path)
    model.load_state_dict(checkpoints['model_state_dict'] if checkpoint_is_dict else checkpoints)
    return model