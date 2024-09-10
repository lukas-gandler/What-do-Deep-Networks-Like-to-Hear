import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from models import CombinedPipeline
from models import CIFAR10_Autoencoder, CIFAR10_Autoencoder_MaxPooling, CIFAR10_CNN
from utils import load_CIFAR10, load_model, Trainer
from utils import number_of_correct

def main():
    # Get DEVICE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'=> Using device {device}')

    # Get DATASET
    print(f'=> Loading dataset')
    train_loader, test_loader = load_CIFAR10(batch_size=16, num_workers=4)

    # Instantiate MODEL
    print(f'=> Building models and assembling pipeline')
    autoencoder = CIFAR10_Autoencoder().to(device)
    autoencoder = load_model('models/pretrained_weights/CIFAR10_AE_baseline_deconvs.pth', autoencoder)

    classifier = CIFAR10_CNN().to(device)
    classifier = load_model('models/pretrained_weights/CIFAR10_CNN_classifier.pth', classifier)

    # Assemble the autoencoder and classifier into the combined pipeline
    pipeline = CombinedPipeline(autoencoder=autoencoder, classifier=classifier, finetune_encoder=False)
    num_params = sum(param.numel() for param in pipeline.parameters())
    print(f'=> Successfully finished assembling pipeline - Total model params: {num_params:,}')

    # Initial ACCURACY
    # train_accuracy_classifier = top_one_accuracy(classifier, train_loader, device)
    # test_accuracy_classifier = top_one_accuracy(classifier, test_loader, device)
    # print(f'f=> Accuracy of classifier - Train {train_accuracy_classifier:.3f} | Test {test_accuracy_classifier:.3f}')
    #
    # train_accuracy_pipeline = top_one_accuracy(pipeline, train_loader, device)
    # test_accuracy_pipeline = top_one_accuracy(pipeline, test_loader, device)
    # print(f'=> Accuracy of pipeline - Train {train_accuracy_pipeline:.3f} | Test {test_accuracy_pipeline:.3f}')

    # Define OPTIMIZER, LOSS-CRITERION and LR-SCHEDULER
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=3e-4, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = None

    # Create MODEL TRAINER and TRAIN MODEL
    training_configs = {'num_epochs': 10,
                        'train_loader': train_loader,
                        'validation_loader': test_loader,
                        'model': pipeline,
                        'optimizer': optimizer,
                        'criterion': criterion,
                        'scheduler': scheduler,
                        }

    model_trainer = Trainer(save_interval=100, device=device, unsupervised_learning=False)
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
