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

class HandwritingSaved(Dataset):

    def __init__(self, dataset_path):
        """
        """
        super().__init__()
        self.dataset = np.load(dataset_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.cycle(idx)
