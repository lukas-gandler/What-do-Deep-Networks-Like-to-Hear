import torch
import torch.nn.functional as F

from models import AudioAutoencoder
from dataloading import load_ESC50
from utils import Trainer

def main():
    # Get DEVICE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'=> Using device {device}')

    # Get DATASET
    train_loader, _ = load_ESC50(batch_size=8, num_workers=4, load_mono=False, fold=1)

    # Load MODEL
    print(f'=> Loading autoencoder')
    autoencoder = AudioAutoencoder(mono_output=True, keep_channel_dim=True).to(device)
    num_params = sum(param.numel() for param in autoencoder.parameters())
    print(f'=> Successfully loaded autoencoder with {num_params:,} parameters')

    # Define OPTIMIZER, LOSS-CRITERION, and LR-SCHEDULER
    print(f'\n=> Setting-up training parameters')
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=1e-3, weight_decay=0.0)
    criterion = lambda prediction, target: F.mse_loss(prediction, target.mean(axis=1, keepdims=True), reduction='none')
    scheduler = None

    # Create MODEL TRAINER and TRAIN MODEL
    training_configs = {
        'num_epochs': 100,
        'train_loader': train_loader,
        'validation_loader': train_loader,
        'model': autoencoder,
        'optimizer': optimizer,
        'criterion': criterion,
        'scheduler': scheduler,
        'resume': None,
        'accumulation_steps': 16,
    }

    model_trainer = Trainer(save_interval=5, device=device, unsupervised_learning=True, use_cross_validation=True)
    model_trainer.train(**training_configs)
    print(f'\n=> Fine-tuning finished')

    print(f'\n=> Trained models saved')
    print(f'=> Done')


if __name__ == '__main__':
    main()
