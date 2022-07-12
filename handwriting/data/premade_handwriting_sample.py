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

class HandwritingSaved(Dataset):

    def __init__(self, dataset_path):
        """ Dataset {author: {word1:[word1_img1, word1_img2], word2:[word2_img1, ...]}}
        """
        super().__init__()
        self.dataset = np.load(dataset_path)

    def __len__(self):
        return len(self.dataset)

    def get_random_author(self):
        return random.choice([self.dataset.keys()])

    def get_random_word_from_author(self, author):
        return random.choice(self.dataset[author])

    def get(self, author=None, word=None):
        return self.dataset[author][word]

    def __getitem__(self, idx):
        author = self.get_random_author()
        word = self.get_random_word_from_author(author)
        return self.get(author,word)

    def render_item(self):
        pass

if __name__ == '__main__':
    dataset = HandwritingSaved("./data/synth_hw/style_187_samples_0.npy")
