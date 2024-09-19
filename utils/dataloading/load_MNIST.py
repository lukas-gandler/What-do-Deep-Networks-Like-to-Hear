from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_MNIST(batch_size: int=32, num_workers: int=1) -> tuple[DataLoader, DataLoader]:
    """
    Loads the MNIST dataset and returns two DataLoader objects for training and testing
    :param batch_size: the batch size to use
    :param num_workers: the number of workers used for data loading/processing
    :return: a tuple of two DataLoader objects for training and testing
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(root='data', download=True, train=True, transform=transform)
    test_dataset = datasets.MNIST(root='data', download=True, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
