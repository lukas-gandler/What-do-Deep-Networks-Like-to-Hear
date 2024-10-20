import argparse

import torch
import torch.nn.functional as F

from models import *
from utils import *
from dataloading import *

def main(args: argparse.Namespace):
    # Get DEVICE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'=> Using device {device}')

    # Get DATASET
    print(f'=> Loading dataset')
    train_loader, test_loader = load_ESC50(batch_size=args.batch_size, num_workers=args.num_workers, load_mono=args.load_mono)

    # Instantiate MODEL
    print(f'=> Building models and assembling pipeline')
    autoencoder = AudioAutoencoder(reduce_output=True).to(device)
    classifier = get_mobilenet(checkpoint='models/pretrained_weights/ESC50_mn10_esc50_epoch_79_acc_960.pt').to(device)
    # classifier = get_dynamic_mobilenet(checkpoint='models/pretrained_weights/ESC50_dymn10_esc50_epoch_79_acc_962.pt').to(device)

    # Assemble the autoencoder and classifier into the combined pipeline
    mel_transformation = MelTransform().to(device)
    pipeline = CombinedPipeline(autoencoder=autoencoder, classifier=classifier, finetune_encoder=args.finetune_encoder, finetune_decoder=args.finetune_decoder, post_ae_transform=mel_transformation)

    num_params = sum(param.numel() for param in pipeline.parameters())
    print(f'=> Successfully finished assembling pipeline - Total model params: {num_params:,}')

    # Define OPTIMIZER, LOSS-CRITERION and LR-SCHEDULER
    optimizer = torch.optim.AdamW(pipeline.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = lambda prediction, target: F.cross_entropy(prediction, target, reduction='none')  # when we want to use Mix-Up we have to use one-hot vectors as targets
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Create MODEL TRAINER and TRAIN MODEL
    training_configs = {'num_epochs': args.num_epochs,
                        'train_loader': train_loader,
                        'validation_loader': test_loader,
                        'model': pipeline,
                        'optimizer': optimizer,
                        'criterion': criterion,
                        'scheduler': scheduler,
                        'resume': args.resume_checkpoint,
                        }

    model_trainer = Trainer(save_interval=args.save_interval, device=device, unsupervised_learning=args.unsupervised_learning)
    model_trainer.train(**training_configs)
    print(f'=> Fine-tuning finished.')

    print(f'=> Trained models saved under {model_trainer.save_dir}/')
    print(f'=> Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of the arguments. ')

    # Data loading params
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--load_mono', action='store_true', default=False, help='Load mono audio')

    # Pipeline params
    parser.add_argument('--finetune_encoder', action='store_true', default=False, help='Finetune encoder')
    parser.add_argument('--finetune_decoder', action='store_true', default=False, help='Finetune decoder')

    # Training params
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--save_interval', type=int, default=5, help='Interval of saving checkpoints')
    parser.add_argument('--unsupervised_learning', action='store_true', default=False, help='Unsupervised learning')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Checkpoint path to resume training')

    args = parser.parse_args()
    main(args)
