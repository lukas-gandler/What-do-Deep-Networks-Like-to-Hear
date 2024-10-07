import os

import torch.nn.utils.rnn
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets import SPEECHCOMMANDS

LABELS = sorted(list({'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'}))

def load_SPEECHCOMMANDS(batch_size: int=32, num_workers: int=1, num_channels: int=1, pin_memory: bool=True, prefetch_factor: int=0) -> tuple[DataLoader, DataLoader]:
    train_set = SubsetSC('training')
    test_set = SubsetSC('testing')

    collate_fn = CollateFn(num_channels)  # Depending on how many channels are desired, we have to use different collate_fn
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=False, prefetch_factor=prefetch_factor)

    return train_loader, test_loader


def pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    return batch.permute(0, 2, 1)

class CollateFn:
    def __init__(self, num_channels: int = 1):
        self.num_channels = num_channels

    def __call__(self, batch):
        tensors, targets = [], []

        # Gather in lists and encode labels as indices
        for waveform, _, label, *_ in batch:
            tensors.append(waveform)
            targets.append(torch.tensor(LABELS.index(label)))

        # Group the list of tensors into a batched tensor
        tensors = pad_sequence(tensors) if self.num_channels <= 1 else pad_sequence(tensors).repeat(1, self.num_channels, 1)
        targets = torch.stack(targets)
        return tensors, targets

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__('data', download=True)

        # Store the sample rate here in the dataset
        _, sample_rate, *_ = self.__getitem__(0)
        self.sample_rate = sample_rate

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath, 'r') as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == 'validation':
            self._walker = load_list('validation_list.txt')
        elif subset == 'testing':
            self._walker = load_list('testing_list.txt')
        elif subset == 'training':
            excludes = load_list('validation_list.txt') + load_list('testing_list.txt')
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]