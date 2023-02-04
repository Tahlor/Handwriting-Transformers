from pathlib import Path
import numpy as np
from math import ceil
from tqdm import tqdm
from PIL import Image
from PIL import PpmImagePlugin

def chunkify(text, max_words):
    """ Chunkify a sentence based on batch_size and maximum number of words

    Returns:

    """
    words = text.split(" ")
    chunks = [words[x:x+max_words] for x in range(0, len(words), max_words)]
    return chunks

def chunkify_with_batch_size(text, batch_size, max_words):
    """ Chunkify a sentence based on batch_size and maximum number of words

    Returns:

    """
    words = text.split(" ")
    words_per_group = max(min(ceil(len(words)/batch_size), max_words),0)
    chunks = [words[x:x+words_per_group] for x in range(0, len(words), words_per_group)]
    return chunks

def list_of_lists_to_array(a):
    """ Once a sentence is converted to a list of lists, this can convert it to an array
            text = "A string of text"
            list_of_list_to_array(split_text_into_batches(text, 7, 12))

    Args:
        a:

    Returns:

    """
    b = np.zeros([len(a), len(max(a, key=lambda x: len(x)))], dtype=object)
    for i, j in enumerate(a):
        b[i][0:len(j)] = j
    return b

def combine_results():
    """

    Returns:

    """
    pass

def show(img):
    from matplotlib import pyplot as plt
    plt.imshow(img, cmap="gray")
    plt.show()

def fix_handwriting_keys():
    for path in Path("/home/taylor/anaconda3/datasets/HANDWRITING_WORD_DATA").rglob("*.npy"):
        dataset = np.load(path, allow_pickle=True).item()
        for _key,_item in dataset.items():
            keys = list(_item.keys())
            for key in tqdm(keys):
                dataset[_key][key.strip()] = _item.pop(key)
        np.save(path, dataset)


def shape(item):
    """
    Args:
        item:
    Returns:
        x, y
    """
    if isinstance(item, np.ndarray):
        return item.shape[1],item.shape[0]
    elif isinstance(item, (PpmImagePlugin.PpmImageFile, Image.Image)):
        return item.size

def ndim(item):
    return len(shape(item))

def channels(item):
    if isinstance(item, np.ndarray):
        if ndim(item)==2:
            return 1
        elif ndim(item)==3:
            return item.shape[-1]
        else:
            raise Exception
    elif isinstance(item, (PpmImagePlugin.PpmImageFile, Image.Image)):
        if item.mode == "L":
            return 1
        elif item.mode == "RGB":
            return 3
        else:
            raise Exception

def display(img, n=1, *args, **kwargs):
    if isinstance(img, list):
        for i,im in enumerate(img):
            if i >= n:
                break
            _display(im, *args, **kwargs)

    else:
        _display(img, *args, **kwargs)

def _display(img, cmap="gray"):
    # if isinstance(img, PpmImagePlugin.PpmImageFile) or isinstance(img, Image.Image):
    #     img.show()
    # else:
    #
    if isinstance(img, ImageDraw.ImageDraw):
        img = img._image
    if isinstance(img,list):
        img = img[0]

    fig, ax = plt.subplots(figsize=(10, 10))

    if channels(img)==3:
        cmap = None
    ax.imshow(img, cmap=cmap)
    plt.show()

if __name__ == "__main__":
    fix_handwriting_keys()
