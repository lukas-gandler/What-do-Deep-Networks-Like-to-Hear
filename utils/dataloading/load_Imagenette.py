import os

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

def load_Imagenette(batch_size: int=32, num_workers: int=1) -> tuple[DataLoader, DataLoader]:
    """
    Loads the Imagenette dataset and returns two DataLoader objects for training and testing
    :param batch_size: the batch size to use
    :param num_workers: the number of workers used for data loading/processing
    :return: a tuple of two DataLoader objects for training and testing
    """

    transform = transforms.Compose([
        transforms.RandomCrop((224, 224), pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Automatically decide if dataset has to be downloaded
    dataset_downloaded = os.path.isdir('data/imagenette2')

    # Since Imagenette is a sub-set of ImageNet we have to map the labels through this custom Dataset-class
    train_set = ImagenetteDataset(datasets.Imagenette(root='data', split='train', download=not dataset_downloaded, transform=transform))
    test_set = ImagenetteDataset(datasets.Imagenette(root='data', split='val', download=False, transform=transform))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

class ImagenetteDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

        # Mapping according to https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
        self.imagenette_to_imagenet = {
            0: 0,    # Tench
            1: 217,  # English Springer
            2: 482,  # Cassette player
            3: 491,  # Chain saw
            4: 497,  # Church
            5: 566,  # French horn
            6: 569,  # Garbage truck
            7: 571,  # Gas pump
            8: 574,  # Golf ball
            9: 701,  # Parachute
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        imagenet_label = self.imagenette_to_imagenet[label]
        return image, imagenet_label