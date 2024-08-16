import torch

from utils import load_mnist
from utils import Trainer
from models import MNIST_Autoencoder

def main():
    # Get DEVICE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'=> Using device {device}')

    # Load DATASET
    print(f'=> Loading dataset')
    mnist_train, mnist_test = load_mnist(batch_size=128, num_workers=4)

    # Instantiate MODEL
    print('=> Building model')
    autoencoder = MNIST_Autoencoder()
    num_params = sum(param.numel() for param in autoencoder.parameters())
    print(f'=> Successfully finished building model - Total model params: {num_params:,}')

    # Define OPTIMIZER and LOSS-CRITERION
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()

    # Create MODEL TRAINER and TRAIN MODEL
    model_trainer = Trainer(save_interval=30, device=device, train_autoencoder=True)
    model_trainer.train(num_epochs=10, train_loader=mnist_train, validation_loader=mnist_test,
                        model=autoencoder, optimizer=optimizer, criterion=criterion) # , resume='checkpoints/checkpoint_epoch_2_losses_0.0674_0.0676.pth')

    print(f'=> Autoencoder pre-training finished.')
    print(f'=> Trained models saved under {model_trainer.save_dir}/')

if __name__ == '__main__':
    main()
