import os
import time
from collections import defaultdict
from handwriting.data.basic_text_dataset import BasicTextDataset
from handwriting.data.dataset import TextDataset, TextDatasetval
from handwriting.data.wikipedia_dataset import Wikipedia
from handwriting.data.unigram_dataset import Unigrams
import torch
import cv2
import os
import numpy as np
from models.model import TRGAN
from params import *
from torch import nn
from handwriting.data.dataset import get_transform
import pickle
from PIL import Image
from tqdm import tqdm
import shutil
import sys
from datasets import load_dataset
from torch.utils.data import DataLoader

from handwriting.data.trivial_dataset import TrivialDataset
from util import render
from math import ceil

MODEL = "IAM"
STYLE = "IAM"
models = {"IAM": 'data/files/iam_model.pth', "CVL": 'data/files/cvl_model.pth'}
styles = {"IAM": 'data/files/IAM-32.pickle', "CVL": 'data/files/CVL-32.pickle'}

def get_model(model_path):
    print('(2) Loading model...')
    model = TRGAN()
    model.netG.load_state_dict(torch.load(model_path))
    print(model_path + ' : Model loaded Successfully')
    model.path = model_path
    return model

class Generator():
    def __init__(self, model, next_text_dataset, batch_size=8, output_path="results"):
        self.model = model
        self.model_path = model.path
        self.output_path = output_path
        self.images_path = styles[STYLE]
        self.next_text_dataset = next_text_dataset
        print ('(1) Loading style and style text next_text_dataset files...')
        self.style_image_and_text_dataset = TextDatasetval(base_path=self.images_path, num_examples=15)
        self.style_loader = DataLoader(
            self.style_image_and_text_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.style_image_and_text_dataset.collate_fn)
        self.new_text_loader = torch.utils.data.DataLoader(
            basic_text_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=basic_text_dataset.collate_fn,)

        item = next(iter(self.new_text_loader))

    def encode_text(self, text_list):
        repeat = False

        if isinstance(text_list,str):
            text_encode = [j.encode() for j in text_list.split(' ')]
            eval_text_encode, eval_len_text = self.model.netconverter.encode(text_encode)
            eval_text_encode = eval_text_encode.repeat(batch_size, 1, 1)

        else:
            for text in text_list:
                text_encode =  [j.encode() for j in text.split(' ')]
                eval_text_encode, eval_len_text = self.model.netconverter.encode(text_encode)

            eval_text_encode = eval_text_encode.repeat(batch_size, 1, 1)
        return eval_text_encode, eval_len_text

    def render_all(self, master_list):
        for i, item in enumerate(master_list):
            page = render.get_page_from_words(item["words"])
            cv2.imwrite(self.output_path+'/image' + str(i) + '.png', page)
        print ('\nOutput images saved in : ' + self.output_path)

    def truncate_if_needed(self, _style, batch_size):
        if _style['imgs_padded'].shape[0] > batch_size:
            _style = self.style_loader.dataset.truncate(_style, batch_size)
        assert _style['imgs_padded'].shape[0] == batch_size
        return _style

    def process_style(self, style, batch_size):
        if style is None:
            assert self.style_image_and_text_dataset.shuffle # this should always be true, but needed if using next(iter())
            _style = next(iter(self.style_loader))
        elif isinstance(style, dict):
            # If at the end of novel text dataset, might need to truncate styles used
            _style = self.truncate_if_needed(style, batch_size)
        else: # optimal way to create one style is to declare it in the dataloader constructions
              # otherwise, it can still be done on the fly directly from the dataset though
              _style = self.style_image_and_text_dataset.get_one_author(n=batch_size, author_id=style)
        return _style

    def generate_new_samples(self, style=None, save_path=None, master_list=None):
        """

        Args:
            style (int or dict with imgs_padded, img_wids, wcl, author_ids):
            save_path:

        Returns:

        """
        if master_list is None:
            master_list = defaultdict(dict)
        for d in tqdm(self.new_text_loader):
            eval_text_encode = d["text_encoded"].to('cuda:0')
            eval_len_text = d["text_encoded_l"] # [d.to('cuda:0') for d in d["text_encoded_l"]]
            _style = self.process_style(style, batch_size=eval_text_encode.shape[0])
            results =  self.model.generate_word_list(
                style_images=_style['imgs_padded'].to(DEVICE),
                style_lengths=_style['img_wids'],
                style_references=_style["wcl"],
                author_ids=_style["author_ids"],
                raw_text=d["text_raw"],
                eval_text_encode=eval_text_encode,
                eval_len_text=eval_len_text,
                source=f"{MODEL}_{STYLE}"
            )
            for result in results:
                author_id = f"{result['author_id']}_{MODEL}_{STYLE}"
                if not author_id in master_list:
                    master_list[author_id] = defaultdict(list)

                # Deal with multiple words
                words = result["raw_text"].split(" ")
                for i, word in enumerate(result['words']):
                    master_list[author_id][words[i]].append(word)

        if save_path:
            np.save(save_path, master_list, allow_pickle=True)
        return master_list



uni = Unigrams(csv_file="./data/datasets/unigram_freq.csv")
trivial = TrivialDataset("This is some data right here")

if __name__ == '__main__':
    # Load novel text next_text_dataset
    text_data = trivial
    from data.unigram_dataset import Unigrams
    model = get_model(models[MODEL])

    if False:
        basic_text_dataset = Wikipedia(
                dataset=load_dataset("wikipedia", "20220301.en")["train"],
                vocabulary=set(ALPHABET),  # set(self.model.netconverter.dict.keys())
                encode_function=model.netconverter.encode,
                min_sentence_length=32,
                max_sentence_length=64
            )
    elif False:
        basic_text_dataset = BasicTextDataset(["This","is","my","dataset","This","is","my"],
                                              set(ALPHABET),
                                              encode_function=model.netconverter.encode)
        basic_text_dataset = BasicTextDataset(["This is a sentence.",
                                               "So is this one."],
                                              set(ALPHABET),
                                              encode_function=model.netconverter.encode)

    else:
        basic_text_dataset = BasicTextDataset(Unigrams(sample=False),
                         set(ALPHABET))
    g = Generator(model=model, next_text_dataset=basic_text_dataset, )
    # Get random style
    #style = next(iter(g.style_image_and_text_dataset))

    # Get specific style
    author_id = g.style_image_and_text_dataset.random_author()
    style = g.style_image_and_text_dataset.get_one_author(n=batch_size, author_id=author_id)
    i = 0
    master_list = None
    while True:
        master_list = g.generate_new_samples(style, save_path=f"./data/datasets/synth_hw/style_{author_id}_samples_{i}.npy", master_list=master_list)
        i+=1
        print(i)

