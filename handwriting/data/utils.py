import numpy as np
from math import ceil

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