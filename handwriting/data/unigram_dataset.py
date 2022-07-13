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

from pathlib import Path
folder = Path(os.path.dirname(__file__))

csv_file=folder / "./datasets/unigram_freq.csv"

class Unigrams(Dataset):
    def __init__(self,
                 csv_file=csv_file,
                 top_k=10000,
                 sample="weighted"):
        """
        Args:
            csv_file (string): Path to unigram next_text_dataset with frequencies/weight
        """
        super().__init__()

        self.csv_file = Path(csv_file)
        self.words, self.counts = self.load_dataset(top_k)
        if sample == "weighted":
            self.sample = self.weighted_sample
        elif sample:
            self.sample = self.unweighted_sample
        else:
            self.sample = self.no_sample

    def load_dataset(self, top_k):
        with self.csv_file.open() as fb:
            reader = csv.reader(fb)
            data = list(reader)[1:]
        words = [x[0] for x in data[:top_k]]
        counts = np.array([int(x[1]) for x in data[:top_k]])
        counts = counts / np.sum(counts)
        return words, counts

    def weighted_sample(self, index, n=1):
        return choice(self.words, n,p=self.counts)

    def unweighted_sample(self, index, n=1):
        return choice(self.words, n)

    def no_sample(self, idx=0, n=1):
        return self.words[idx:idx+n]

    def get_text(self):
        return self.sample()

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        """ Only return 1 item

        Args:
            idx:

        Returns:

        """
        return self.sample(idx)[0]

if __name__ == '__main__':
    U = Unigrams(csv_file="./datasets/unigram_freq.csv")
    word = next(iter(U))
    words = U.sample(n=20)
    print(words)