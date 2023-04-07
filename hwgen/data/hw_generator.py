from hwgen.util.util import get_available_gpus
import traceback
from collections import defaultdict
from textgen.basic_text_dataset import BasicTextDataset
from textgen.data.dataset import TextDatasetval
from textgen.wikipedia_dataset import WikipediaEncodedTextDataset
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
from hwgen import resources
from torch.nn.parallel import DataParallel

folder = Path(os.path.dirname(__file__))
"""
This file contains the code for pre-generating handwriting from text.
"""


VOCABULARY = """Only thewigsofrcvdampbkuq.A-210xT5'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/"""

def get_model(model_path, english_words, device, gpu_ids="all"):
    print('(2) Loading model...')
    model = TRGAN(english_words=english_words, device=device)
    if device=="cpu":
        warnings.warn("Loading model on CPU. This will be slow.")
        model.netG.load_state_dict(torch.load(model_path , map_location=torch.device(device)))
    else:
        state_dict = torch.load(model_path)
        model.netG.load_state_dict(state_dict)

        # if gpu_ids:
        #     if gpu_ids == "all":
        #         gpu_ids = get_available_gpus()
        #     print("Using GPUs: {}".format(gpu_ids))
        #     model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    print(str(model_path) + ' : Model loaded Successfully')
    model.path = model_path
    return model

class HWGenerator(Dataset, BasicTextDataset):
    def __init__(self,
                 next_text_dataset,
                 model="IAM",
                 batch_size=8,
                 sequence_length=16,
                 output_path="results",
                 style="IAM",
                 english_words_path=None,
                 device=None,
                 words_before_new_style=1000):
        """ Why does this inherit from BasicTextDataset?
            This should not be a dataloader, should not be batched

        Args:
            next_text_dataset: a BasicTextDataset that produces the text the generator will generate as HW
            model: IAM or CVL or path to .pth file
            batch_size: how many "lines" to produce
            sequence_length: how many words to sample for one "line"; some text generators already have this set
            output_path:
            style: IAM, CVL, or path to .pickle file
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model
        self.style_name = style
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.output_path = output_path
        self.next_text_dataset = next_text_dataset
        self.words_before_new_style = 1000
        self.current_style_id = 0
        if model in resources.models.keys():
            resources.download_model_resources()
            model = resources.models[model]

        self.model = get_model(model, english_words_path, device=self.device)

        self.model_path = self.model.path
        self.style_images_path = resources.styles[style] if style in resources.styles.keys() else style

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
            drop_last=True,
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
        """ this is actuall the whole eval loop
            new_text_loaded is already batched

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
            _style = self.process_style(style, batch_size=eval_text_encode.shape[0])
            m = torch.max(eval_text_encode)
            print(m)

            results =  self.model.generate_word_list(
                style_images=_style['imgs_padded'].to(self.device),
                style_lengths=_style['img_wids'],
                style_references=_style["wcl"],
                author_ids=_style["author_ids"],
                raw_text=d["text"],
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
                for i, word in enumerate(result['word_imgs']):
                    master_list[author_id][words[i]].append(word)

        if save_path:
            np.save(save_path, master_list, allow_pickle=True)
        return master_list

    def process_batch(self, text_dict, style=None):
        """

        Args:
            text_dict (dict):
            style (int or dict with imgs_padded, img_wids, wcl, author_ids):

        Returns:

        """
        eval_text_encode = text_dict["text_encoded"].to(self.device)
        eval_len_text = text_dict["text_encoded_l"] # [d.to('cuda:0') for d in d["text_encoded_l"]]
        _style = self.process_style(style, batch_size=eval_text_encode.shape[0])

        results =  self.model.generate_word_list(
            style_images=_style['imgs_padded'].to(self.device),
            style_lengths=_style['img_wids'],
            style_references=_style["wcl"],
            author_ids=_style["author_ids"],
            text_dict=text_dict,
            eval_text_encode=eval_text_encode,
            eval_len_text=eval_len_text,
            source=f"{self.model_name}_{self.style_name}"
        )
        for i, result in enumerate(results):
            try:
                author_id = f"{result['author_id']}_{self.model_name}_{self.style_name}"
                result.update({"text_list": text_dict["text_list"][i],
                               "author_id": author_id}
                              )
                yield result
            except Exception as e:
                print(e)


    def __len__(self):
        return len(self.next_text_dataset)

    def get_random_author(self):
        author_id = self.style_image_and_text_dataset.random_author()
        return author_id

    def get_random_word(self):
        word_idx = random.randint(0, len(self.next_text_dataset)-1)
        return self.next_text_dataset[word_idx]

    def get_style(self, author_style_id=None):
        if author_style_id is None and self.word_idx % self.words_before_new_style < self.prev_word_idx:
            author_style_id = self.current_style_id = self.get_random_author()
        else:
            author_style_id = self.current_style_id
        self.prev_word_idx = self.word_idx
        return self.style_image_and_text_dataset.get_one_author(n=self.batch_size, author_id=author_id)

    def get(self, author_style_id=None):
        """

        Args:
            data_dict: {"text": readable text,
                "text_encoded": (index? encoding over alphabet),
                "text_encoded_l": (lengths)}
            author_style_id:

        Returns:

        """
        for data_dict in self.new_text_loader:
            for item in self.process_batch(data_dict, style=author_style_id):
                yield item

    def get_data_dict_from_word_list(self, word_list, author_style_id=None):
        raise Exception("Not implemented")
        eval_text_encode, eval_len_text = self.encode(word_list)
        seq_length = min(self.max_seq_length, 1+remaining_words // self.batch_size)
        for i in range(0, len(word_list), seq_length):
            yield {"text": " ".join(word_list[i:i+seq_length]),
                   "text_encoded": eval_text_encode[i:i+seq_length],
                   "text_encoded_l": eval_len_text[i:i+seq_length]}

    def __iter__(self):
        for item in self.get():
            try:
                yield item
            except:
                traceback.print_exc()
                continue

    def __next__(self):
        return next(self.get())

    def __getitem__(self, item):
        return next(self.get())

"""
g.model.netconverter.decode(torch.tensor([80]),torch.tensor([1]))
"""

if __name__ == '__main__':
    uni = Unigrams(csv_file="./data/datasets/unigram_freq.csv")
    trivial = TrivialDataset("This is some data right here")

    # Load novel text next_text_dataset
    text_data = trivial

    basic_text_dataset = WikipediaEncodedTextDataset(
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

