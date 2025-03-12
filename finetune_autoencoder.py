import argparse
from argparse import Namespace
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio.transforms as audio_transforms

from models import *
from utils import *
from dataloading import *

def get_lr_scheduler(scheduler: str, optimizer: torch.optim.Optimizer, args: Namespace) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    match scheduler:
        case 'cosine':
            print("=> using cosine annealing")
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.eta_min)

        case 'plateau':
            print("=> using plateau annealing")
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience)

        case 'one_cycle':
            print("=> using one_cycle annealing")
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, div_factor=args.div_factor, epochs=args.num_epochs, steps_per_epoch=1600 // (args.batch_size * args.accumulation_steps) + 1)

        case 'step':
            print("=> using step annealing")
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        case 'none':
            print("=> not using an lr-scheduler")
            return None

        case _:
            raise RuntimeError(f'Unsupported scheduler {scheduler}')

def main(args: argparse.Namespace) -> None:
    # Get DEVICE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'=> Using device {device}')

    # Get DATASET
    print(f'\n=> Loading dataset @ {args.sample_rate:,} HZ for autoencoder pre-processing')
    load_transform = audio_transforms.Resample(orig_freq=32_000, new_freq=args.sample_rate)
    train_loader, test_loader = load_ESC50(batch_size=args.batch_size, num_workers=args.num_workers, load_mono=args.load_mono, fold=args.fold, transform=load_transform)

    # Instantiate MODEL
    print(f'\n=> Building models and assembling pipeline')
    autoencoder = get_autoencoder(args.autoencoder, keep_channel_dim=args.model_name != 'passt').to(device)
    classifier = get_classifier(args.model_name).to(device)

    # Assemble the autoencoder and classifier into the combined pipeline
    post_ae_transforms = [ audio_transforms.Resample(orig_freq=args.sample_rate, new_freq=32_000).eval().to(device), ]

    # NOTE: PaSST has its mel-transformation already built in -> the pipeline handles setting the train- and eval-mode
    if args.model_name != 'passt':
        mel_transformation = MelTransform().eval().to(device)
        post_ae_transforms.append(mel_transformation)

    pipeline = CombinedPipeline(autoencoder=autoencoder, classifier=classifier, finetune_encoder=args.finetune_encoder, finetune_decoder=args.finetune_decoder, post_ae_transform=post_ae_transforms)

    num_params = sum(param.numel() for param in pipeline.parameters())
    print(f'=> Successfully finished assembling pipeline - Total model params: {num_params:,}')

    # Define OPTIMIZER, LOSS-CRITERION and LR-SCHEDULER
    print(f'\n=> Setting-up training parameters')
    optimizer = torch.optim.AdamW(pipeline.autoencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = lambda prediction, target: F.cross_entropy(prediction, target, reduction='none')  # when we want to use Mix-Up we have to use one-hot vectors as targets, therefore need this lambda
    scheduler = get_lr_scheduler(args.lr_scheduler, optimizer, args)

    # Create MODEL TRAINER and TRAIN MODEL
    training_configs = {'num_epochs': args.num_epochs,
                        'train_loader': train_loader,
                        'validation_loader': test_loader,
                        'model': pipeline,
                        'optimizer': optimizer,
                        'criterion': criterion,
                        'scheduler': scheduler,
                        'resume': args.resume_checkpoint,
                        'accumulation_steps': args.accumulation_steps
                        }

    model_trainer = Trainer(save_interval=args.save_interval, device=device, unsupervised_learning=args.unsupervised_learning)
    model_trainer.train(**training_configs)
    print(f'\n=> Fine-tuning finished.')

    print(f'\n=> Trained models saved under {model_trainer.save_dir}/')
    print(f'=> Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of the arguments. ')


    # Data loading params
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--load_mono', action='store_true', default=False, help='Load mono audio')
    parser.add_argument('--fold', type=int, default=1, help='Defines Test-Fold')


    # Pipeline params
    parser.add_argument('--model_name', type=str, default='mn', choices=['mn', 'mn_rr1', 'mn_rr2', 'dymn', 'dymn_rr1', 'dymn_rr2', 'dymn_noCA', 'dymn_noDC', 'dymn_noDR', 'dymn_onlyCA', 'dymn_onlyDC', 'dymn_onlyDR', 'passt'], help='Name of the classifier to use (only mn, dymn or passt are valid)')
    parser.add_argument('--autoencoder', type=str, default='esc-pretrained', choices=['esc-pretrained', 'archisound', 'random'], help='The Autoencoder version to use  for analysis')
    parser.add_argument('--sample_rate', type=int, default=48_000, help='The sampling rate the autoencoder will work with. NOTE: The autoencoder is designed to work with 48kHZ.')
    parser.add_argument('--finetune_encoder', action='store_true', default=False, help='Finetune encoder')
    parser.add_argument('--finetune_decoder', action='store_true', default=False, help='Finetune decoder')


    # Training params
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Steps for accumulating gradients')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--save_interval', type=int, default=5, help='Interval of saving checkpoints')
    parser.add_argument('--unsupervised_learning', action='store_true', default=False, help='Unsupervised learning')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Checkpoint path to resume training')


    # Scheduler params
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'plateau', 'one_cycle', 'step', 'none'], help='Learning rate scheduler')
    parser.add_argument('--eta_min', type=float, default=0.0, help='Minimum learning rate for CosineAnnealingLR')
    parser.add_argument('--factor', type=float, default=0.1, help='Learning rate scheduler factor for ReduceOnPlateau')
    parser.add_argument('--patience', type=int, default=5, help='Patience for ReduceOnPlateau')
    parser.add_argument('--max_lr', type=float, default=1e-5, help='Maximum learning rate for OneCycleLR')
    parser.add_argument('--div_factor', type=float, default=25.0, help='Div factor for OneCycleLR')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for StepLR')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma for StepLR')


    command_line_args = parser.parse_args()
    main(command_line_args)
