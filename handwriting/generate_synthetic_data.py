import os
import time
from data.dataset import TextDataset, TextDatasetval, Wikipedia
import torch
import cv2
import os
import numpy as np
from models.model import TRGAN
from params import *
from torch import nn
from data.dataset import get_transform
import pickle
from PIL import Image
import tqdm
import shutil
import sys
from datasets import load_dataset
from util import render

MODEL = "IAM"
STYLE = "IAM"
models = {"IAM": 'files/iam_model.pth', "CVL": 'files/cvl_model.pth'}
styles = {"IAM": 'files/IAM-32.pickle', "CVL": 'files/CVL-32.pickle'}


class Generator():
    def __init__(self, output_path="results"):
        self.output_path = output_path
        self.model_path = models[MODEL]
        self.images_path = styles[STYLE]
        #self.model_path = 'files/cvl_model.pth'; self.images_path = 'files/CVL-32.pickle' #(cvl)
        #self.model_path = 'files/iam_model.pth'; self.images_path = 'files/CVL-32.pickle' #(iam-cvl-cross)
        #self.model_path = 'files/cvl_model.pth'; self.images_path = 'files/IAM-32.pickle' #(cvl-iam-cross)#

        print ('(1) Loading next_text_dataset files...')

        self.TextDatasetObjval = TextDatasetval(base_path = self.images_path, num_examples = 15)
        self.style_data = torch.utils.data.DataLoader(
                    self.TextDatasetObjval,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True, drop_last=True,
                    collate_fn=self.TextDatasetObjval.collate_fn)

        print ('(2) Loading model...')

        self.model = TRGAN()
        self.model.netG.load_state_dict(torch.load(self.model_path))
        print (self.model_path+' : Model loaded Successfully')

        # Load novel text next_text_dataset
        self.text_data = Wikipedia(
            dataset=load_dataset("wikipedia", "20220301.en")["train"],
            vocabulary=set(ALPHABET), # set(self.model.netconverter.dict.keys())
            encode_function=self.model.netconverter.encode,
            min_sentence_length=32,
            max_sentence_length=64
        )

        self.text_loader=torch.utils.data.DataLoader(
                    self.text_data,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=self.text_data.collate_fn,
                    )
        item = next(iter(self.text_loader))
        item = next(iter(self.text_loader))

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

    def generate_new_sample(self, text):
        print ('(3) Loading text content...')
        eval_text_encode_old, eval_len_text_old = self.encode_text(text)

        if os.path.isdir(self.output_path): shutil.rmtree(self.output_path)
        os.makedirs(self.output_path, exist_ok = True)
        master_list = []
        # Each is 1 page

        for d in self.text_loader:
            eval_text_encode = d["text_encoded"].to('cuda:0')
            eval_len_text = d["text_encoded_l"] # [d.to('cuda:0') for d in d["text_encoded_l"]]
            #for i,style in enumerate(tqdm.tqdm(self.style_loader)):
            style = next(iter(self.style_data))
            master_list += self.model.generate_word_list(
                style_images=style['imgs_padded'].to(DEVICE),
                style_lengths=style['img_wids'],
                style_references=style["wcl"],
                author_ids=style["author_ids"],
                raw_text=d["text_raw"],
                eval_text_encode=eval_text_encode,
                eval_len_text=eval_len_text,
                source=f"{MODEL}_{STYLE}"
            )
            break
            # style_references", words, author_ids, source (next_text_dataset)
        for i, item in enumerate(master_list):
            page = render.get_page_from_words(item["words"])
            cv2.imwrite(self.output_path+'/image' + str(i) + '.png', page)

        print ('\nOutput images saved in : ' + self.output_path)

if __name__ == '__main__':
    g = Generator()
    text = "A paragraph is a series of related sentences developing a central idea, called the topic. Try to think about paragraphs in terms of thematic unity: a paragraph is a sentence or a group of sentences that supports one central, unified idea. Paragraphs add one idea at a time to your broader argument"
    g.generate_new_sample(text)