from __future__ import print_function, division
import random
import warnings
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
from PIL import Image, ImageDraw, ImageFilter
from cv2 import resize
from typing import Literal

class SavedHandwriting(Dataset):
    """ !!! This should inherit from the same thing as the font renderer?
        Font render / HW render could be same package???
        Move img conversion utilties somewhere
    """
    def __init__(self,
                 format: Literal['numpy', 'PIL'],
                 dataset_path,
                 font_size=None,
                 random_ok=False,
                 conversion=None):
        """ Dataset {author: {word1:[word1_img1, word1_img2], word2:[word2_img1, ...]}}

        Args:
            dataset_path:
            random_ok (bool): Supply random word if requested word not available
            author (str): Author ID
            conversion (func): conversion function to run on images
        """
        super().__init__()
        self.format = format
        self.dataset = np.load(dataset_path, allow_pickle=True).item()
        self.random_ok = random_ok
        self.conversion = conversion
        self._font_size = font_size

    def __len__(self):
        return len(self.dataset)

    @property
    def font_size(self):
        """ Possibly make this more intelligent to determine the font size from items?

        Returns:

        """
        return self._font_size

    def resize_to_height_numpy(self, image, height):
        width = int(image.shape[1] * height / image.shape[0])
        return resize(image, [width, height])

    def get_random_author(self):
        return random.choice(list(self.dataset.keys()))

    def get_random_word_from_author(self, author=None):
        if author is None:
            author = self.get_random_author()
        return author, random.choice(list(self.dataset[author].keys()))

    def get(self,
            author=None,
            word=None):
        word_cln=word.strip()
        if author is None:
            author = self.get_random_author()
        if word is None:
            _, word = self.get_random_word_from_author(author)
        elif self.random_ok and word_cln not in self.dataset[author]:
            warnings.warn("Requested word not available, using random word")
            _, word = self.get_random_word_from_author(author)

        word_img = random.choice(self.dataset[author][word_cln])
        if self.format=="numpy":
            word_img = np.array(word_img)

        return {"image":word_img,
                "font": author,
                "raw_text": word
                }

    def __getitem__(self, idx):
        return self.get()

    def render_word(self, word,
                    font=None,
                    size=None,
                    conversion=None,
                    *args,
                    **kwargs):
        """

        Args:
            word:
            font (str): author_id
            size (int)
        Returns:
            dict:

        """
        img_dict = self.get(word=word,
                        author=font
                        )
        if not size is None:
            img_dict["image"] = self.resize_to_height_numpy(img_dict["image"], height=size)

        if self.format=="PIL":
            img_dict["image"] = Image.fromarray(np.uint8(img_dict["image"]*255))

        if not self.conversion is None:
            img_dict["image"] = self.conversion(img_dict["image"])
        img_dict["raw_text"] = word

        return img_dict

    def render_phrase(self, max_spacing, min_spacing):
        pass

if __name__ == '__main__':
    dataset = SavedHandwriting("./datasets/synth_hw/style_298_samples_0.npy")
    author, word = dataset.get_random_word_from_author()
    show(dataset.get(author=author, word=word)["image"])
    print(word)