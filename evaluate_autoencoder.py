import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt

from models import *
from dataloading import *
from utils import *

NUM_ROWS = 4

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'=> Using device {device}')

    print(f'=> Loading dataset')
    train_loader, validation_loader = load_CIFAR10(batch_size=NUM_ROWS**2)

    print(f'=> Building model')
    autoencoder = CIFAR10_Autoencoder()
    # autoencoder = load_model('models/pretrained_weights/CIFAR10_AE_baseline_maxpooling_bigger_bn.pth', autoencoder.to(device))
    classifier = CIFAR10_CNN()

    pipeline = CombinedPipeline(autoencoder=autoencoder, classifier=classifier)
    pipeline = load_model('checkpoints/final_model.pth', pipeline.to(device))

    print(f'=> Running model')
    images, labels = next(iter(train_loader))
    original_img_grid = torchvision.utils.make_grid(images, nrow=NUM_ROWS, padding=2, normalize=True)

    rec_images = pipeline.autoencoder.forward(images.to(device)).cpu()
    # rec_images = autoencoder.forward(images.to(device)).cpu()
    rec_img_grid = torchvision.utils.make_grid(rec_images, nrow=NUM_ROWS, padding=2, normalize=True)

    print('=> Plotting results')
    f, axarr = plt.subplots(1, 2, figsize=(20, 10))

    axarr[0].imshow(np.transpose(original_img_grid.numpy(), (1, 2, 0)))
    axarr[0].set_title('Original Images')

    axarr[1].imshow(np.transpose(rec_img_grid.numpy(), (1, 2, 0)))
    axarr[1].set_title('Reconstructed Images')

    plt.show()
    print(f'=> Done')

if __name__ == '__main__':
    main()
