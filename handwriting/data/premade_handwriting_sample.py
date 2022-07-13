from __future__ import print_function, division
import random
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
from handwriting.data.utils import show

class HandwritingSaved(Dataset):

    def __init__(self, dataset_path):
        """ Dataset {author: {word1:[word1_img1, word1_img2], word2:[word2_img1, ...]}}
        """
        super().__init__()
        self.dataset = np.load(dataset_path, allow_pickle=True).item()
        pass

    def __len__(self):
        return len(self.dataset)

    def get_random_author(self):
        return random.choice(list(self.dataset.keys()))

    def get_random_word_from_author(self, author=None):
        if author is None:
            author = self.get_random_author()
        return author, random.choice(list(self.dataset[author].keys()))

    def get(self, author=None, word=None):
        if author is None:
            author = self.get_random_author()
        if word is None:
            _, word = self.get_random_word_from_author(author)
        word_img = random.choice(self.dataset[author][word])

        return word_img

    def __getitem__(self, idx):
        return self.get()

    def render_item(self):
        pass

if __name__ == '__main__':
    dataset = HandwritingSaved("./datasets/synth_hw/style_298_samples_0.npy")
    author, word = dataset.get_random_word_from_author()
    show(dataset.get(author=author, word=word))
    print(word)