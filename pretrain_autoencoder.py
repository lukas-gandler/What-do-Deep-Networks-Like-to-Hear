import torch

from utils.dataloading import load_mnist
from utils.trainer import Trainer
from models.autoencoder.MNIST_autoencoder import MNIST_Autoencoder

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'=> Using device {device}')

    print(f'=> Loading dataset')
    mnist_train, mnist_test = load_mnist()

    print('=> Building model')
    autoencoder = MNIST_Autoencoder()
    num_params = sum(param.numel() for param in autoencoder.parameters())
    print(f'=> Successfully finished building model - Total model params: {num_params:,}')

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    model_trainer = Trainer(save_interval=3, device=device, train_autoencoder=True)
    model_trainer.train(num_epochs=5, train_loader=mnist_train, validation_loader=mnist_test,
                        model=autoencoder, optimizer=optimizer, criterion=criterion) # , resume='checkpoints/checkpoint_epoch_2_losses_0.0674_0.0676.pth')

    print(f'=> Autoencoder pre-training finished.')
    print(f'=> Trained models saved under {model_trainer.save_dir}/')

if __name__ == '__main__':
    main()
