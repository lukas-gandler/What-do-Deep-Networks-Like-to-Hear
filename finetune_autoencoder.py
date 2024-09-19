import torch
import torchaudio.transforms
import torchvision

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from models import *
from utils import *

def main():
    # Get DEVICE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'=> Using device {device}')

    # Get DATASET
    print(f'=> Loading dataset')
    # train_loader, _  = load_SPEECHCOMMANDS_h5(batch_size=16, num_workers=8, num_channels=2, prefetch_factor=2)
    train_loader, test_loader = load_SPEECHCOMMANDS(batch_size=16, num_workers=8, num_channels=2, prefetch_factor=2)

    # Instantiate MODEL
    print(f'=> Building models and assembling pipeline')
    autoencoder = AudioAutoencoder(reduce_output=True).to(device)
    # autoencoder = load_model('models/pretrained_weights/CIFAR10_AE_baseline_deconvs.pth', autoencoder)

    classifier = M5().to(device)
    classifier = load_model('models/pretrained_weights/SPEECHCOMMANDS_M5_acc_92.pth', classifier, checkpoint_is_dict=False)

    # Assemble the autoencoder and classifier into the combined pipeline
    post_transform = torchaudio.transforms.Resample(orig_freq=16_000, new_freq=8_000).to(device)
    pipeline = CombinedPipeline(autoencoder=autoencoder, classifier=classifier, finetune_encoder=False, post_ae_transform=post_transform)

    num_params = sum(param.numel() for param in pipeline.parameters())
    print(f'=> Successfully finished assembling pipeline - Total model params: {num_params:,}')

    # Define OPTIMIZER, LOSS-CRITERION and LR-SCHEDULER
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=3e-4, weight_decay=1e-5)
    criterion = torch.nn.NLLLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    # Create MODEL TRAINER and TRAIN MODEL
    training_configs = {'num_epochs': 20,
                        'train_loader': train_loader,
                        'validation_loader': test_loader,
                        'model': pipeline,
                        'optimizer': optimizer,
                        'criterion': criterion,
                        'scheduler': scheduler,
                        'resume': 'checkpoints/checkpoint_epoch_5_losses_2427.2948_229.9855.pth',
                        }

    model_trainer = Trainer(save_interval=1, device=device, unsupervised_learning=False)
    model_trainer.train(**training_configs)
    print(f'=> Fine-tuning finished.')

    # Fine-tuned ACCURACY
    # train_accuracy_pipeline = top_one_accuracy(pipeline, train_loader, device)
    # test_accuracy_pipeline = top_one_accuracy(pipeline, test_loader, device)
    # print(f'=> Accuracy of fine-tuned pipeline - Train {train_accuracy_pipeline:.3f} | Test {test_accuracy_pipeline:.3f}')

    print(f'=> Trained models saved under {model_trainer.save_dir}/')
    print(f'=> Done')

if __name__ == '__main__':
    main()
