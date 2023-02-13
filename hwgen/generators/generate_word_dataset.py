from collections import defaultdict
from textgen.basic_text_dataset import BasicTextDataset

from textgen.data.dataset import  TextDatasetval
from textgen.wikipedia_dataset import Wikipedia
from textgen.unigram_dataset import Unigrams
import torch
import cv2
import numpy as np
from hwgen.models.model import TRGAN
from hwgen.params import *
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader

from textgen.trivial_dataset import TrivialDataset
from hwgen.util import render

MODEL = "CVL"
STYLE_DATA_SOURCE = "CVL"
models = {"IAM": 'data/files/iam_model.pth', "CVL": 'data/files/cvl_model.pth'}
styles = {"IAM": 'data/files/IAM-32.pickle', "CVL": 'data/files/CVL-32.pickle'}

def get_model(model_name):
    model_path = models[model_name]
    print('(2) Loading model...')
    model = TRGAN()
    model.netG.load_state_dict(torch.load(model_path))
    print(model_path + ' : Model loaded Successfully')
    model.path = model_path
    model.name = model_name
    return model

class Generator():
    def __init__(self, model,
                 next_text_dataset,
                 batch_size=8,
                 output_path="results",
                 style_data_source=STYLE_DATA_SOURCE):
        self.style_data_source = style_data_source
        self.model = model
        self.model_name = model.name
        self.model_path = model.path
        self.output_path = output_path
        self.images_path = styles[style_data_source]
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
            page = render.get_page_from_words(item["word_imgs"])
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
            eval_text_encode = d["text_encoded"].to(self.device)
            eval_len_text = d["text_encoded_l"] # [d.to('cuda:0') for d in d["text_encoded_l"]]
            m = torch.max(eval_text_encode)
            #print(m)
            _style = self.process_style(style, batch_size=eval_text_encode.shape[0])
            model_and_style_data_source = f"{self.model_name}_{self.style_data_source}"
            results =  self.model.generate_word_list(
                style_images=_style['imgs_padded'].to(DEVICE),
                style_lengths=_style['img_wids'],
                style_references=_style["wcl"],
                author_ids=_style["author_ids"],
                raw_text=d["text_raw"],
                eval_text_encode=eval_text_encode,
                eval_len_text=eval_len_text,
                source=model_and_style_data_source
            )
            for i, result in enumerate(results):
                #print(i)
                author_id = f"{result['author_id']}_{model_and_style_data_source}"
                if not author_id in master_list:
                    master_list[author_id] = defaultdict(list)

                # Deal with multiple words
                words = result["raw_text"].split(" ")
                for i, word in enumerate(result['words']):
                    master_list[author_id][words[i]].append(word)

        if save_path:
            np.save(save_path, master_list, allow_pickle=True)
        return master_list



uni = Unigrams(csv_file="../data/datasets/unigram_freq.csv")
trivial = TrivialDataset("This is some data right here")

def misc(model):
    basic_text_dataset = Wikipedia(
        dataset=load_dataset("wikipedia", "20220301.en")["train"],
        vocabulary=set(ALPHABET),  # set(self.model.netconverter.dict.keys())
        encode_function=model.netconverter.encode,
        min_sentence_length=32,
        max_sentence_length=64
    )
    basic_text_dataset = BasicTextDataset(["This", "is", "my", "dataset", "This", "is", "my"],
                                          set(ALPHABET),
                                          encode_function=model.netconverter.encode)
    basic_text_dataset = BasicTextDataset(["This is a sentence.",
                                           "So is this one."],
                                          set(ALPHABET),
                                          encode_function=model.netconverter.encode)

def random_style_loop():
    # Get random style
    author_id = g.style_image_and_text_dataset.random_author()
    style = g.style_image_and_text_dataset.get_one_author(n=batch_size, author_id=author_id)
    i = 0
    master_list = None
    while True:
        OUTPUT = f"./data/datasets/synth_hw/style_{author_id}_samples_{i}.npy"
        master_list = g.generate_new_samples(style, save_path=OUTPUT, master_list=master_list)
        i += 1
        print(i)


def iterative_style_loop():
    # Get specific style
    ROOT = Path("./data/datasets/synth_hw")
    ROOT = Path("/home/taylor/anaconda3/HANDWRITING_WORD_DATA")
    src_data = g.style_image_and_text_dataset
    for author in range(0, len(src_data.author_ids)):
        author_id = src_data.author_ids[author]
        print(f"Working on {author_id}")
        OUTPUT = ROOT / f"style_{author_id}_{g.model_name}_{g.style_data_source}_samples.npy"
        print(OUTPUT)
        if OUTPUT.exists():
            print(f"ALREADY EXISTS {OUTPUT}")
            continue
        style = g.style_image_and_text_dataset.get_one_author(n=batch_size, author_id=author_id)
        g.generate_new_samples(style, save_path=OUTPUT)


if __name__ == '__main__':
    # Load novel text next_text_dataset
    text_data = trivial
    from textgen.unigram_dataset import Unigrams
    models_styles = [["CVL","CVL"],["IAM","IAM"]]
    for model_name,style_source in models_styles:
        print(f"Working on {model_name}{style_source}")
        model = get_model(model_name)
        basic_text_dataset = BasicTextDataset(Unigrams(sample=False),
                         set(ALPHABET))
        g = Generator(model=model,
                      next_text_dataset=basic_text_dataset,
                      style_data_source=style_source)
        iterative_style_loop()
