"""
Taken from the EfficientAT GitHub project (https://github.com/fschmid56/EfficientAT/tree/main)
"""

import torch
from torch.utils.data import Dataset, DataLoader

import random
import librosa
import os

import numpy as np
import pandas as pd

from functools import partial
from typing import Tuple

DATASET_DIR = 'data/esc50'
assert DATASET_DIR is not None, "Specify ESC50 dataset location in variable 'DATASET_DIR'."

DATASET_CONFIG = {
    'meta_csv': os.path.join(DATASET_DIR, 'meta/esc50.csv'),
    'audio_path': os.path.join(DATASET_DIR, 'audio_32k/'),
    'num_classes': 50
}

def load_ESC50(batch_size: int=32, num_workers: int=1, load_mono: bool=True, fold: int=1) -> tuple[DataLoader, DataLoader]:
    train_set = get_training_set(resample_rate=32_000, roll=False, wav_mix=False, gain_augment=12, fold=fold, load_mono=load_mono)
    test_set = get_test_set(resample_rate=32_000, fold=fold, load_mono=load_mono)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_init_fn, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn, pin_memory=True)

    return train_loader, test_loader

#region Dataset classes
def pad_or_truncate(x: np.ndarray, audio_length: int) -> np.ndarray:
    """
    Pad all audio to specific length
    :param x: the audio data
    :param audio_length: the length of the audio data
    :return:
    """

    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x), dtype=x.dtype)), axis=0)
    else:
        return x[:audio_length]

def pydub_augment(waveform: np.ndarray, gain_augment: int=0) -> np.ndarray:
    if gain_augment == 0:
        gain = torch.randint(gain_augment * 2, (1,)).item() - gain_augment
        amp = 10 ** (gain / 20)
        waveform = waveform * amp
    return waveform

def roll_func(b, axis=1, shift=None, shift_range=4_000):
    x = b[0]
    x = torch.as_tensor(x)

    others = b[1:]

    sf = shift if shift is not None else int(np.random.random_integers(-shift_range, shift_range))
    return x.roll(sf, axis), *others

def get_roll_func(axis=1, shift=None, shift_range=4_000):
    return partial(roll_func, axis=axis, shift=shift, shift_range=shift_range)

class AudioSetDataset(Dataset):
    def __init__(self, meta_csv: str, audio_path: str, fold: int, train: bool=False, resample_rate: int=32_000,
                 classes_num: int=50, clip_length: int=5, gain_augment=0, load_mono:bool = True):
        """
        Reads the mp3 bytes from HDF file decodes using av and returns a fixed length audio wav
        :param meta_csv: CSV file indicating classes
        :param audio_path: path to audio file
        :param fold: fold to use for testing
        :param train: if True use all except given fold for training else load given fold for testing
        :param resample_rate: new sample rate
        :param classes_num: number of classes
        :param clip_length: clip length
        :param gain_augment: gain augmentation
        """

        self.resample_rate = resample_rate
        self.meta_csv = meta_csv
        self.dataset = pd.read_csv(self.meta_csv)
        self.fold = fold

        if train:  # training all except this
            print(f'=> Dataset training fold {self.fold} selection out of {len(self.dataset)}')
            self.df = self.dataset[self.dataset.fold != self.fold]
            print(f'=>  for training remains {len(self.df)}')
        else:
            print(f'=> Dataset testing fold {self.fold} selection out of {len(self.dataset)}')
            self.df = self.dataset[self.dataset.fold == self.fold]
            print(f'=>  for testing remains {len(self.df)}')

        self.clip_length = (clip_length * resample_rate) // 4 * 4  # make sure that the clip_length is multiple of 4
        self.classes_num = classes_num
        self.gain_augment = gain_augment
        self.audio_path = audio_path
        self.load_mono = load_mono

        # cross-validation
        self.dataset_folds = list(set(self.df.fold))
        self.validation_fold_idx = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Load waveform and target of an audio clip
        :param idx: index of the item
        :return: waveform, target
        """

        row = self.df.iloc[idx]
        waveform, _ = librosa.load(self.audio_path + row.filename, sr=self.resample_rate, mono=self.load_mono)
        # if self.gain_augment:
        #     waveform = pydub_augment(waveform, self.gain_augment)

        waveform = pydub_augment(waveform, self.gain_augment) if self.gain_augment else waveform
        waveform = pad_or_truncate(waveform, self.clip_length)
        waveform = waveform.reshape(1, -1)  # adds the channel dimension
        waveform = waveform.repeat(2, 0) if not self.load_mono and waveform.shape[0] == 1 else waveform

        target = np.zeros(self.classes_num)
        target[row.target] = 1

        return waveform, target

    def cv_train_mode(self):
        """
        Modify the dataset such that one of the training folds gets selected as the validation fold.
        The selected validation fold then gets excluded from the training set.
        :return:
        """

        validation_fold = self.dataset_folds[self.validation_fold_idx]
        self.df = self.dataset[self.dataset.fold != self.fold]
        self.df = self.df[self.df.fold != validation_fold]

    def cv_validation_mode(self):
        """
        Modify the dataset such that only the current validation fold gets selected.
        :return:
        """

        validation_fold = self.dataset_folds[self.validation_fold_idx]
        self.df = self.dataset[self.dataset.fold == validation_fold]

    def select_next_validation_fold(self):
        """
        Updates the current validation fold.
        :return:
        """

        self.validation_fold_idx = (self.validation_fold_idx + 1) % len(self.dataset_folds)

class PreprocessDataset(Dataset):
    """
    A base preprocessing dataset representing a preprocessing step of a Dataset preprocessed on the fly.
    Supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset: AudioSetDataset, preprocessor):
        self.dataset = dataset

        if not callable(preprocessor):
            print('preprocessor: ', preprocessor)
            raise ValueError('preprocessor should be callable')
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.preprocessor(self.dataset[idx])

class MixupDataset(Dataset):
    """
    Mixing Up wave forms
    """

    def __init__(self, dataset: AudioSetDataset, beta: int = 2, rate: float = 0.5):
        self.beta = beta
        self.rate = rate
        self.dataset = dataset
        print(f'=> Mixing up waveforms from dataset of len {len(dataset)}')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if torch.rand(1) < self.rate:
            x1, y1 = self.dataset[idx]

            idx2 = torch.randint(len(self.dataset), (1,)).item()
            x2, y2 = self.dataset[idx2]

            l = np.random.beta(self.beta, self.beta)
            l = max(l, 1.0 - l)

            x1 = x1 - x1.mean()
            x2 = x2 - x2.mean()

            x = (x1 * l + x2 * (1.0 - l))
            x = x - x.mean()

            return x, (y1 * l + y2 * (1.0 - l))
        else:
            return self.dataset[idx]

#endregion

def get_base_training_set(resample_rate: int=32_000, gain_augment: int=0, fold: int=1, load_mono: bool=True) -> AudioSetDataset:
    meta_csv = DATASET_CONFIG['meta_csv']
    audio_path = DATASET_CONFIG['audio_path']

    ds = AudioSetDataset(meta_csv, audio_path, fold, train=True, resample_rate=resample_rate, gain_augment=gain_augment, load_mono=load_mono)
    return ds

def get_base_test_set(resample_rate: int=32_000, fold: int=1, load_mono: bool=True) -> AudioSetDataset:
    meta_csv = DATASET_CONFIG['meta_csv']
    audio_path = DATASET_CONFIG['audio_path']

    ds = AudioSetDataset(meta_csv, audio_path, fold, train=False, resample_rate=resample_rate, load_mono=load_mono)
    return ds

def get_training_set(resample_rate: int=32_000, roll: bool=False, wav_mix: bool=False, gain_augment: int=0, fold: int=1, load_mono: bool=True) -> Dataset:
    ds = get_base_training_set(resample_rate=resample_rate, gain_augment=gain_augment, fold=fold, load_mono=load_mono)

    if roll:
        ds = PreprocessDataset(ds, get_roll_func())
    if wav_mix:
        ds = MixupDataset(ds)

    return ds

def get_test_set(resample_rate: int=32_000, fold: int=1, load_mono: bool=True) -> AudioSetDataset:
    ds = get_base_test_set(resample_rate, fold=fold, load_mono=load_mono)
    return ds


#region Dataloader helper functions
def worker_init_fn(wid):
    seed_sequence = np.random.SeedSequence([torch.initial_seed(), wid])

    to_seed = spawn_get(seed_sequence, 2, dtype=int)
    torch.random.manual_seed(to_seed)

    np_seed = spawn_get(seed_sequence, 2, dtype=np.ndarray)
    np.random.seed(np_seed)

    py_seed = spawn_get(seed_sequence, 2, dtype=int)
    random.seed(py_seed)

def spawn_get(seedseq, n_entropy, dtype):
    child = seedseq.spawn(1)[0]
    state = child.generate_state(n_entropy, dtype=np.uint32)

    if dtype == np.ndarray:
        return state
    elif dtype == int:
        state_as_int = 0
        for shift, s in enumerate(state):
            state_as_int = state_as_int + int((2 ** (32 * shift) * 2))
        return state_as_int
    else:
        raise ValueError(f'not a valid dtype "{dtype}"')

#endregion