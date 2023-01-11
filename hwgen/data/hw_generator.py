from collections import defaultdict
from textgen.basic_text_dataset import BasicTextDataset
from hwgen.data.dataset import  TextDatasetval
from textgen.wikipedia_dataset import Wikipedia
from textgen.unigram_dataset import Unigrams
import cv2
import numpy as np
from hwgen.models.model import TRGAN
from hwgen.params import *
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from textgen.trivial_dataset import TrivialDataset
from hwgen.util import render
import random
import warnings
from pathlib import Path
folder = Path(os.path.dirname(__file__))

VOCABULARY = """Only thewigsofrcvdampbkuq.A-210xT5'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/"""

def get_model(model_path):
    print('(2) Loading model...')
    model = TRGAN()
    model.netG.load_state_dict(torch.load(model_path))
    print(str(model_path) + ' : Model loaded Successfully')
    model.path = model_path
    return model

class HWGenerator(Dataset, BasicTextDataset):
    def __init__(self,
                 next_text_dataset,
                 model="IAM",
                 batch_size=8,
                 output_path="results",
                 style="IAM"):

        self.model_name = model
        self.style_name = style

        models = {
            "IAM": folder / 'files/iam_model.pth',
            "CVL": folder / 'files/cvl_model.pth'}
        styles = {
            "IAM": folder / 'files/IAM-32.pickle',
            "CVL": folder / 'files/CVL-32.pickle'}

        self.model = get_model(models[model])
        self.style_images_path = styles[style]

        self.model_path = self.model.path
        self.output_path = output_path
        self.next_text_dataset = next_text_dataset
        print ('(1) Loading style and style text next_text_dataset files...')
        self.style_image_and_text_dataset = TextDatasetval(base_path=self.style_images_path, num_examples=15)
        self.style_loader = DataLoader(
            self.style_image_and_text_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.style_image_and_text_dataset.collate_fn)
        self.new_text_loader = torch.utils.data.DataLoader(
            next_text_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=next_text_dataset.collate_fn,)

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
            m = torch.max(eval_text_encode)
            print(m)

            results =  self.model.generate_word_list(
                style_images=_style['imgs_padded'].to(DEVICE),
                style_lengths=_style['img_wids'],
                style_references=_style["wcl"],
                author_ids=_style["author_ids"],
                raw_text=d["text_raw"],
                eval_text_encode=eval_text_encode,
                eval_len_text=eval_len_text,
                source=f"{self.model_name}_{self.style_name}"
            )
            for i, result in enumerate(results):
                # print(i)
                author_id = f"{result['author_id']}_{self.model_name}_{self.style_name}"
                if not author_id in master_list:
                    master_list[author_id] = defaultdict(list)

                # Deal with multiple words
                words = result["raw_text"].split(" ")
                for i, word in enumerate(result['words']):
                    master_list[author_id][words[i]].append(word)

        if save_path:
            np.save(save_path, master_list, allow_pickle=True)
        return master_list

    def process_batch(self, data_dict, style=None, save_path=None, master_list=None):
        """

        Args:
            style (int or dict with imgs_padded, img_wids, wcl, author_ids):
            save_path:

        Returns:

        """
        eval_text_encode = data_dict["text_encoded"].to('cuda:0')
        eval_len_text = data_dict["text_encoded_l"] # [d.to('cuda:0') for d in d["text_encoded_l"]]
        _style = self.process_style(style, batch_size=eval_text_encode.shape[0])

        results =  self.model.generate_word_list(
            style_images=_style['imgs_padded'].to(DEVICE),
            style_lengths=_style['img_wids'],
            style_references=_style["wcl"],
            author_ids=_style["author_ids"],
            raw_text=data_dict["text_raw"],
            eval_text_encode=eval_text_encode,
            eval_len_text=eval_len_text,
            source=f"{self.model_name}_{self.style_name}"
        )
        for i, result in enumerate(results):
            author_id = f"{result['author_id']}_{self.model_name}_{self.style_name}"
            result.update({"author_id": author_id})
            yield result

    def __len__(self):
        return len(self.dataset)

    def get_random_author(self):
        return random.choice(list(self.dataset.keys()))

    def get_random_word_from_author(self, author=None):
        if author is None:
            author = self.get_random_author()
        return author, random.choice(list(self.dataset[author].keys()))

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

        """
        img_dict = self.get(word=word,
                            author=font
                            )
        if not self.conversion is None:
            img_dict["image"] = self.conversion(img_dict["image"])

        return img_dict


    def get(self,
            author=None,
            word=None):
        if author is None:
            author = self.get_random_author()
        if word is None:
            _, word = self.get_random_word_from_author(author)
        elif self.random_ok and word not in self.dataset[author]:
            warnings.warn("Requested word not available, using random word")
            _, word = self.get_random_word_from_author(author)

        word_img = random.choice(self.dataset[author][word])

        return {"image":word_img,
                "font": author,
                }

    def __getitem__(self, idx):
        return self.get()


"""
g.model.netconverter.decode(torch.tensor([80]),torch.tensor([1]))
"""

if __name__ == '__main__':
    uni = Unigrams(csv_file="./data/datasets/unigram_freq.csv")
    trivial = TrivialDataset("This is some data right here")

    # Load novel text next_text_dataset
    text_data = trivial

    basic_text_dataset = Wikipedia(
                dataset=load_dataset("wikipedia", "20220301.en")["train"],
                vocabulary=set(VOCABULARY),  # set(self.model.netconverter.dict.keys())
                encode_function=Wikipedia.encode,
                min_sentence_length=60,
                max_sentence_length=64
            )

    g = HWGenerator(model="IAM",
                    next_text_dataset=basic_text_dataset,
                    vocabulary=set(VOCABULARY),
                    encoder=HWGenerator.encode,
    )
    # Get random style
    #style = next(iter(g.style_image_and_text_dataset))
    words = basic_text_dataset.sample() # just return text

    # Get specific style
    author_id = g.style_image_and_text_dataset.random_author()
    style = g.style_image_and_text_dataset.get_one_author(n=batch_size, author_id=author_id)
    i = 0
    master_list = None
    while True:
        master_list = g.generate_new_samples(style, save_path=f"./data/datasets/synth_hw_wiki/style_{author_id}_samples_{i}.npy", master_list=master_list)
        i+=1
        print(i)

