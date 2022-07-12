from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import csv
from numpy.random import choice
from handwriting.data.basic_text_dataset import BasicTextDataset

class TrivialDataset(Dataset):
    def __init__(self, dataset):
        """ Just a wrapper around an iterator
        """
        super().__init__()
        self.update_dataset(dataset)

    def update_dataset(self, new_dataset):
        """

        Args:
            new_dataset: str or iterable

        Returns:

        """
        if isinstance(new_dataset, str):
            new_dataset = new_dataset.split(' ')
        self.dataset = new_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        #return self.cycle(idx)
        return self.dataset[idx]

    def cycle(self, idx):
        while True:
            for x in self.dataset[idx % len(self)]:
                yield x


if __name__ == '__main__':
    U = TrivialDataset(dataset="This is a line of text".split())
    word = next(iter(U))
    print(word)

    basic_text_dataset = BasicTextDataset(["This", "is", "my", "dataset"])
    DL = torch.utils.data.DataLoader(basic_text_dataset, batch_size=12, collate_fn=basic_text_dataset.collate_fn)
    for word in DL:
        print(word)