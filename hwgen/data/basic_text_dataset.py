#from hwgen.data.dataset import *
from datasets import load_dataset
from textgen.data.dataset import *
from textgen.utils import *
import re
import torch

VOCABULARY = """Only thewigsofrcvdampbkuq.A-210xT5'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%"""
VOCABULARY = """Only thewigsofrcvdampbkuq.A-210xT5'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/"""
ALPHA_VOCABULARY = """ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"""
DICT = {'O': 1, 'n': 2, 'l': 3, 'y': 4, ' ': 5, 't': 6, 'h': 7, 'e': 8, 'w': 9, 'i': 10, 'g': 11, 's': 12, 'o': 13, 'f': 14, 'r': 15, 'c': 16, 'v': 17, 'd': 18, 'a': 19, 'm': 20, 'p': 21, 'b': 22, 'k': 23, 'u': 24, 'q': 25, '.': 26, 'A': 27, '-': 28, '2': 29, '1': 30, '0': 31, 'x': 32, 'T': 33, '5': 34, "'": 35, 'M': 36, 'D': 37, 'L': 38, ',': 39, 'R': 40, 'Y': 41, 'H': 42, 'J': 43, '"': 44, 'I': 45, 'S': 46, 'P': 47, 'W': 48, 'E': 49, 'N': 50, 'j': 51, '&': 52, 'B': 53, 'C': 54, '9': 55, '3': 56, 'V': 57, 'G': 58, 'F': 59, 'K': 60, 'z': 61, '(': 62, ')': 63, ';': 64, '#': 65, ':': 66, '!': 67, '7': 68, 'U': 69, '6': 70, '4': 71, 'Q': 72, '8': 73, '?': 74, '+': 75, '*': 76, 'Z': 77, 'X': 78, '/': 79} #'%': 80}
space = re.compile(r"\s")

class BasicTextDataset:
    """ Just a wrapper that goes on top of traditional PyTorch datasets
    """
    def __init__(self, dataset,
                 vocabulary=VOCABULARY,
                 encode_function=None,
                 exclude_chars="",
                 *args, **kwargs):
        self.dataset = dataset
        self.vocabulary = vocabulary # all chars must be present, will become the encode function
        self.filtered_vocabulary = [x for x in vocabulary if not x in exclude_chars] # exclude some valid characters (if they were learned badly)
        self.exclude_chars = exclude_chars
        self.translate_map = str.maketrans("", "", "".join(self.exclude_chars))
        if not exclude_chars:
            self.filter = lambda x:x
        self.filter_sentence = filter_vocab(self.filtered_vocabulary)
        self.encode_function = encode_function if encode_function else self._encode_function
        self.collate_fn = collate_fn_text
        self.vocab_size = 0
        self.dict = DICT

    def filter(self, text):
        return space.sub(str.translate(text, self.translate_map), " ")

    def encode(self, text):
        if isinstance(text, str):
            text = text.split(' ')
        text_encode = [self.filter_sentence(j).encode() for j in text]
        eval_text_encode, eval_len_text = self.encode_function(text_encode)
        return eval_text_encode, eval_len_text

    def _encode_function(self, text):
        length = []
        result = []
        results = []
        for item in text:
            item = item.decode('utf-8', 'strict')
            length.append(len(item))
            for char in item:
                # If missing character in decoding
                if not char in self.dict:
                    self.vocab_size += 1
                    self.dict[char] = self.vocab_size
                index = self.dict[char]

                result.append(index)
            results.append(result)
            result = []

        return (torch.nn.utils.rnn.pad_sequence([torch.LongTensor(text) for text in results], batch_first=True),
                torch.IntTensor(length))

    def get_text(self, i):
        return self.dataset[i]

    def __getitem__(self, index) -> dict:
        text_raw = self.get_text(index)
        text_filtered = self.filter(text_raw)
        text_encode, text_encode_l = self.encode(text_filtered)
        return {
            "text_raw": text_filtered,
            "text_encoded": text_encode,
            "text_encoded_l": text_encode_l,
        }

    def __len__(self):
        return len(self.dataset)
