import torch
import torch.nn.functional as F
import torchaudio.transforms as audio_transforms

from models import AudioAutoencoder
from dataloading import load_ESC50
from utils import Trainer

def main():
    # Get DEVICE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'=> Using device {device}')

    # Get DATASET
    wave_transform = audio_transforms.Resample(orig_freq=32_000, new_freq=48_000)
    dataloder, _ = load_ESC50(batch_size=4, num_workers=4, load_mono=False, fold=1, transform=wave_transform)

    # # Load MODEL
    # print(f'\n=> Loading autoencoder')
    # autoencoder = AudioAutoencoder(mono_output=True, keep_channel_dim=True).to(device)
    # num_params = sum(param.numel() for param in autoencoder.parameters())
    # print(f'=> Successfully loaded autoencoder with {num_params:,} parameters')
    #
    # # Define OPTIMIZER, LOSS-CRITERION, and LR-SCHEDULER
    # print(f'\n=> Setting-up training parameters')
    # optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=3e-5, weight_decay=0.0)
    # criterion = lambda prediction, target: F.mse_loss(prediction, target.mean(axis=1, keepdims=True), reduction='none')
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=70, eta_min=0.0)
    #
    # # Create MODEL TRAINER and TRAIN MODEL
    # training_configs = {
    #     'num_epochs': 70,
    #     'train_loader': train_loader,
    #     'validation_loader': test_loader,
    #     'model': autoencoder,
    #     'optimizer': optimizer,
    #     'criterion': criterion,
    #     'scheduler': scheduler,
    #     'resume': None,
    #     'accumulation_steps': 16,
    # }
    #
    # model_trainer = Trainer(save_interval=5, device=device, unsupervised_learning=True)
    # model_trainer.train(**training_configs)

    for folds in dataloder.dataset.dataset_folds:
        current_fold = dataloder.dataset.dataset_folds[dataloder.dataset.validation_fold_idx]

        # Load MODEL
        print(f'\n=> Loading autoencoder')
        autoencoder = AudioAutoencoder(mono_output=True, keep_channel_dim=True).to(device)
        num_params = sum(param.numel() for param in autoencoder.parameters())
        print(f'=> Successfully loaded autoencoder with {num_params:,} parameters')

        # Define OPTIMIZER, LOSS-CRITERION, and LR-SCHEDULER
        print(f'\n=> Setting-up training parameters')
        optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=3e-5, weight_decay=0.0)
        criterion = lambda prediction, target: F.mse_loss(prediction, target.mean(axis=1, keepdims=True), reduction='none')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=70, eta_min=0.0)

        # Create MODEL TRAINER and TRAIN MODEL
        training_configs = {
            'num_epochs': 70,
            'dataloader': dataloder,
            'model': autoencoder,
            'optimizer': optimizer,
            'criterion': criterion,
            'scheduler': scheduler,
            'resume': None,
            'accumulation_steps': 16,
        }

        model_trainer = Trainer(save_interval=5, device=device, unsupervised_learning=True)
        model_trainer.train_cross_validation(fold=current_fold, **training_configs)
        dataloder.dataset.select_next_validation_fold()

    print(f'\n=> Fine-tuning finished')

    print(f'\n=> Trained models saved')
    print(f'=> Done')


if __name__ == '__main__':
    main()
