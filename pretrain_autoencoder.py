import torch

from utils.dataloading import load_mnist
from utils.trainer import Trainer
from models.autoencoder.MNIST_autoencoder import MNIST_Autoencoder

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    mnist_train, mnist_test = load_mnist()
    autoencoder = MNIST_Autoencoder()

    num_params = sum(param.numel() for param in autoencoder.parameters())
    print('Model params:', num_params)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    model_trainer = Trainer(device=device, train_autoencoder=True)
    model_trainer.train(5, autoencoder, mnist_train, mnist_test, optimizer, criterion)

if __name__ == '__main__':
    main()
