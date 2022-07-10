""" Prespecified forms that can be sliced up into a dataloader but with labels that allow them to be rebuilt after
"""

from data.utils import chunkify
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
from data.basic_text_dataset import BasicTextDataset

from pathlib import Path
folder = Path(os.path.dirname(__file__))

csv_file=folder / "./datasets/unigram_freq.csv"

class FormWords(Dataset):
    """ Dataset that reads in text from forms that go together, chunks it, then rebuilds it after
    """
    def __init__(self,
                 dataset_path,
                 max_words=64):
        """
        Args:
            csv_file (string): Path to unigram next_text_dataset with frequencies/weight
        """
        super().__init__()
        self.dataset = self.load_dataset(dataset_path)


    def load_dataset(self, dataset_path):
        loaded = np.load()
        for i,item in enumerate(loaded_data):
            chunkify()
            # add ids in 
        # output "chunked" pieces that are defined how to put them back together


    def parse_results(self):
        """ Take unorganized results and build them into a single group

        Returns:

        """

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        return self.sample()[0]

if __name__ == '__main__':
    U = FormWords(csv_file="./datasets/unigram_freq.csv")
    word = next(iter(U))
    words = U.sample(n=20)
    print(words)