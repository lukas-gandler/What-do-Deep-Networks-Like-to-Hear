import torch
import torch.nn.functional as F

from models import *
from utils import *
from dataloading import *

def main():
    # Get DEVICE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'=> Using device {device}')

    # Get DATASET
    print(f'=> Loading dataset')
    train_loader, test_loader = load_ESC50(batch_size=4, num_workers=4, load_mono=False)

    # Instantiate MODEL
    print(f'=> Building models and assembling pipeline')
    autoencoder = AudioAutoencoder(reduce_output=True).to(device)
    classifier = get_mobilenet(checkpoint='models/pretrained_weights/ESC50_mn10_esc50_epoch_79_acc_960.pt').to(device)
    # classifier = get_dynamic_mobilenet(checkpoint='models/pretrained_weights/ESC50_dymn10_esc50_epoch_79_acc_962.pt').to(device)

    # Assemble the autoencoder and classifier into the combined pipeline
    mel_transformation = MelTransform().to(device)
    pipeline = CombinedPipeline(autoencoder=autoencoder, classifier=classifier, finetune_encoder=False, post_ae_transform=mel_transformation)

    num_params = sum(param.numel() for param in pipeline.parameters())
    print(f'=> Successfully finished assembling pipeline - Total model params: {num_params:,}')

    # Define OPTIMIZER, LOSS-CRITERION and LR-SCHEDULER
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=3e-4, weight_decay=1e-5)
    criterion = lambda prediction, target: F.cross_entropy(prediction, target, reduction='none')  # when we want to use Mix-Up we have to use one-hot vectors as targets
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # Create MODEL TRAINER and TRAIN MODEL
    training_configs = {'num_epochs': 200,
                        'train_loader': train_loader,
                        'validation_loader': test_loader,
                        'model': pipeline,
                        'optimizer': optimizer,
                        'criterion': criterion,
                        'scheduler': scheduler,
                        'resume': None,
                        }

    model_trainer = Trainer(save_interval=5, device=device, unsupervised_learning=False)
    model_trainer.train(**training_configs)
    print(f'=> Fine-tuning finished.')

    print(f'=> Trained models saved under {model_trainer.save_dir}/')
    print(f'=> Done')

if __name__ == '__main__':
    main()
