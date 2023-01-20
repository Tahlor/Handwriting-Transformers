from __future__ import print_function, division

import random
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from cv2 import resize
from torch.utils.data import Dataset

from textgen.basic_text_dataset import BasicTextDataset
from hwgen.data.utils import show

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

class SavedHandwriting(BasicTextDataset, Dataset):
    """ !!! This should inherit from the same thing as the font renderer?
        Font render / HW render could be same package???
        Move img conversion utilties somewhere
        Don't inherit torch.utils.data.Dataset, it must be initialized on some kind of data
    """
    def __init__(self,
                 format: Literal['numpy', 'PIL'],
                 dataset_path=None,
                 font_size=None,
                 random_ok=False,
                 conversion=None):
        """ Dataset {author: {word1:[word1_img1, word1_img2], word2:[word2_img1, ...]}}

        Args:
            dataset_path: path to dataset; 'sample' will download a sample from S3
            random_ok (bool): Supply random word if requested word not available
            author (str): Author ID
            conversion (func): conversion function to run on images
        """
        self.dataset_path = dataset_path
        self.dataset = np.load(dataset_path, allow_pickle=True).item()

        super().__init__(self.dataset)
        self.format = format
        if dataset_path == "sample":
            from hwgen.resources import download_handwriting
            dataset_path = download_handwriting()

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
            author,
            word):
        return self._get(author, word)

    def _get(self,
            author=None,
            word=None):

        if author is None:
            author = self.get_random_author()
        if word is None:
            _, word = self.get_random_word_from_author(author)
        elif self.random_ok and word not in self.dataset[author]:
            warnings.warn("Requested word not available, using random word")
            _, word = self.get_random_word_from_author(author)

        word=word.strip()

        word_img = random.choice(self.dataset[author][word])
        if self.format=="numpy":
            word_img = np.array(word_img)

        return {"image":word_img,
                "font": author,
                "raw_text": word
                }

    def _batch_get(self, author_list, word_list, batch_size = None):
        """ Temporary hack to function as a dataloader
            In reality, we need to pass this dataset as a dataloader and create a collate function
            See RenderImageTextPair for possible solution
        Args:
            author_list:
            word_list:

        Returns:

        """
        if author_list is None:
            author_list = [None] * batch_size
        return [self._get(author, word) for author, word in zip(author_list, word_list)]

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

class SavedHandwritingRandomAuthor(SavedHandwriting):
    """ Returns {"image":word_img,
                "font": author,
                "raw_text": word
                }
    """
    def __init__(self,
                 format: Literal['numpy', 'PIL'],
                 dataset_root=None,
                 font_size=None,
                 random_ok=False,
                 conversion=None,
                 switch_frequency=1000,
                 *args,
                 **kwargs):
        """ Each batch will be all the same author
            Switch to new author every switch_frequency number of words
        """
        if dataset_root == "sample":
            from hwgen.resources import download_handwriting
            dataset_root = Path(download_handwriting()).parent
        elif dataset_root == "eng_latest":
            from hwgen.resources import download_handwriting_zip
            dataset_root = download_handwriting_zip()

        self.dataset_root = Path(dataset_root)
        assert self.dataset_root.is_dir()
        self.data_files = list(self.dataset_root.rglob("*.npy"))
        self.switch_frequency = switch_frequency
        if not self.data_files:
            raise Exception("No handwriting data files found")

        self.dataset_path = self.reset()
        self.global_step = 0

        super().__init__(format=format,
                         dataset_path=self.dataset_path,
                         font_size=font_size,
                         random_ok=random_ok,
                         conversion=conversion)

    def next_author(self):
        self.dataset_path = self.dataset_queue.pop() if self.dataset_queue else self.reset()
        self.dataset = np.load(self.dataset_path, allow_pickle=True).item()
        print("Next author")

    def reset(self):
        self.dataset_queue = self.data_files.copy()
        random.shuffle(self.dataset_queue)
        return self.dataset_queue.pop()

    def get(self,
            author,
            word):
        self.global_step +=1
        if self.global_step % self.switch_frequency == 0:
            self.next_author()
        return self._get(author, word)



if __name__ == '__main__':
    dataset = SavedHandwriting("./datasets/synth_hw/style_298_samples_0.npy")
    author, word = dataset.get_random_word_from_author()
    show(dataset.get(author=author, word=word)["image"])
    print(word)
