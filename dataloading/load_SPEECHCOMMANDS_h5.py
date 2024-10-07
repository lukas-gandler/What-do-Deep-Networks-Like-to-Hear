import h5py
import os
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
from tqdm import tqdm

LABELS = sorted(list({'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'}))

def load_SPEECHCOMMANDS_h5(batch_size: int=32, num_workers: int=1, num_channels: int=1, pin_memory: bool=True, prefetch_factor: int=0) -> tuple[DataLoader, DataLoader]:
    train_set = SubsetSC('training')
    test_set  = SubsetSC('testing')

    collate_fn = CollateFn(num_channels)
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
        for waveform, _, label in batch:
            tensors.append(waveform)
            targets.append(torch.tensor(LABELS.index(label)))

        # Group the list of tensors into a batched tensor
        tensors = pad_sequence(tensors) if self.num_channels <= 1 else pad_sequence(tensors).repeat(1, self.num_channels, 1)
        targets = torch.stack(targets)
        return tensors, targets

class SubsetSC(Dataset):
  def __init__(self, subset: str = None, hdf5_path='data/speech_commands.h5', download=True):
    self.hdf5_path = hdf5_path
    self.subset = subset
    self.dataset = SPEECHCOMMANDS('data', download=download)

    # If HDF5 file doesn't exist, create it and store the dataset
    if not os.path.exists(self.hdf5_path):
      self.create_hdf5(download)

    # Load the dataset from HDF5 file
    self.load_hdf5()

  def create_hdf5(self, download):
    sample_rates = []
    labels = []


    # Create the HDF5 file
    with h5py.File(self.hdf5_path, 'w') as hdf5_file:
      # Create datasets for audio, sample rates and labels
      audio_group = hdf5_file.create_group('audio')

      # Iterate through the dataset to extract audio and labels
      for i, (waveform, sample_rate, label, *_) in enumerate(tqdm(self.dataset, desc='=> Iterating through the dataset to extract audio, sampel rates and labels to build HDF5 file')):
        audio_group.create_dataset(f'audio_{i}', data=waveform.numpy(), dtype='float32')
        sample_rates.append(sample_rate)
        labels.append(label)


      # Store labels and sample rate in the HDF5 file
      hdf5_file.create_dataset('sample_rates', data=np.array(sample_rates, dtype='int'))    # Store sample rates
      hdf5_file.create_dataset('labels', data=np.array(labels, dtype='S'))                  # Store labels as a fixed-length strings

  def load_hdf5(self):
    self.hdf5_file = h5py.File(self.hdf5_path, 'r')
    self.audio_data = self.hdf5_file['audio']
    self.sample_rates = self.hdf5_file['sample_rates']
    self.labels = self.hdf5_file['labels']

    # Filter the dataset based on the subset
    if self.subset == 'validation':
      self._walker = self.load_list('validation_list.txt')
    elif self.subset == 'testing':
      self._walker = self.load_list('testing_list.txt')
    elif self.subset == 'training':
      excludes = self.load_list('validation_list.txt') + self.load_list('testing_list.txt')
      excludes = set(excludes)
      self._walker = [i for i, label in enumerate(self.labels) if label not in excludes]

  def load_list(self, filename):
    filepath = os.path.join(self.dataset._path, filename)
    with open(filepath) as fileobj:
      return [os.path.normpath(os.path.join(self.dataset._path, line.strip())) for line in fileobj]

  def __len__(self):
    return len(self._walker)

  def __getitem__(self, index):
    actual_index = self._walker[index]

    # Load the waveform, sample rate and label from HDF5
    waveform = self.hdf5_file[f'audio/audio_{actual_index}'][:]
    sample_rate = self.sample_rates[actual_index]
    label = self.labels[actual_index].decode('utf-8')
    return torch.Tensor(waveform), sample_rate, label
