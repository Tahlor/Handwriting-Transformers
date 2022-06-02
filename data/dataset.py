# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
#import lmdb
import torchvision.transforms as transforms
import six
import sys
from PIL import Image
import numpy as np
import os
import sys
import pickle
import numpy as np
from params import *
import re
from text_utils import *

def get_transform(grayscale=False, convert=True):

    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


class TextDataset():

    def __init__(self, base_path = DATASET_PATHS,  num_examples = 15, target_transform=None):

        self.NUM_EXAMPLES = num_examples
  
        #base_path = DATASET_PATHS
        file_to_store = open(base_path, "rb")
        self.IMG_DATA = pickle.load(file_to_store)['train']
        self.IMG_DATA  = dict(list( self.IMG_DATA.items()))#[:NUM_WRITERS])
        if 'None' in self.IMG_DATA.keys():
            del self.IMG_DATA['None']
        self.author_id = list(self.IMG_DATA.keys())

        self.transform = get_transform(grayscale=True)
        self.target_transform = target_transform
        
        self.collate_fn = TextCollator()


    def __len__(self):
        return len(self.author_id)

    def __getitem__(self, index):

        

        NUM_SAMPLES = self.NUM_EXAMPLES


        author_id = self.author_id[index]

        self.IMG_DATA_AUTHOR = self.IMG_DATA[author_id]
        random_idxs = np.random.choice(len(self.IMG_DATA_AUTHOR), NUM_SAMPLES, replace = True)

        rand_id_real = np.random.choice(len(self.IMG_DATA_AUTHOR))
        real_img = self.transform(self.IMG_DATA_AUTHOR[rand_id_real]['img'].convert('L'))
        real_labels = self.IMG_DATA_AUTHOR[rand_id_real]['label'].encode()


        imgs = [np.array(self.IMG_DATA_AUTHOR[idx]['img'].convert('L')) for idx in random_idxs]
        labels = [self.IMG_DATA_AUTHOR[idx]['label'].encode() for idx in random_idxs]
       
        max_width = 192 #[img.shape[1] for img in imgs] 
        
        imgs_pad = []
        imgs_wids = []

        for img in imgs:

            img = 255 - img
            img_height, img_width = img.shape[0], img.shape[1]
            outImg = np.zeros(( img_height, max_width), dtype='float32')
            outImg[:, :img_width] = img[:, :max_width]

            img = 255 - outImg

            imgs_pad.append(self.transform((Image.fromarray(img))))
            imgs_wids.append(img_width)

        imgs_pad = torch.cat(imgs_pad, 0)

        item = {
            'imgs_padded': imgs_pad,
            'img_wids': imgs_wids,
            'img': real_img,
            'label': real_labels,
            'img_path': 'img_path',
            'idx': 'indexes',
            'wcl': index,
            'author_id': author_id
        }

        return item



def collate_fn_text(data):
    """
                          alphabet_size=27,
                          sentence_length=32,
                          embedding_dim=512
    Args:
        data (list): list of dicts of target_length (batch)

    Returns:

    """
    keys = data[0].keys()
    output_dict = {}
    padding = {"masked_gt": -100,
                "attention_mask": 0,
                "text": 0,
                "gt_idxs": 0,
                "vgg_logits": 0,
                "embedding":0,
                "image": -0.4242,
                #"text_raw": text_raw,
                "text_encoded": 0, # batch x max_sentence x max_word_length
               }
    padding2d = set(["text_encoded"])
    for data_key in keys:
        batch_data = [b[data_key] for b in data]

        # pad if it is already a tensor and padding value in dict
        if data_key in padding.keys() and batch_data and torch.is_tensor(batch_data[0]):
            if data_key in padding2d:
                dims = [len(batch_data)]
                for i in range(len(batch_data[0].shape)):
                    dims.append(max([x.shape[i] for x in batch_data]))
                _batch_data = torch.zeros(*dims) + padding[data_key]
                for i,d in enumerate(batch_data):
                    slices = [i] + [slice(0,x) for x in d.shape]
                    _batch_data[slices] = d
                batch_data = _batch_data
                # last_dim_max = max([x.shape[-1] for x in batch_data])
                # penultimate_dim_max = max([x.shape[-2] for x in batch_data])
                # batch
                # output = torch.zeros()
                # for i,item in batch_data:
                #     last_dim_pad = last_dim_max - item.shape[-1]
                #     penultimate_dim_pad = penultimate_dim_max - item.shape[-1]
                #
                #     #batch_data[i] = torch.nn.functional.pad(batch_data[i], pad=[0,last_dim_pad,0,penultimate_dim_pad], value=padding[data_key])
            else:
                batch_data = torch.nn.utils.rnn.pad_sequence(batch_data, batch_first=True, padding_value=padding[data_key])
        else:
            output_dict[data_key] = batch_data
    return output_dict


class Wikipedia():
    def __init__(self,
                 dataset,
                 vocabulary,
                 encode_function,
                 min_sentence_length=32,
                 max_sentence_length=64,
                 ):
        self.dataset = dataset
        self.vocabulary = vocabulary
        self.filter_sentence = filter_vocab(vocabulary)
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.encode_function = encode_function
        self.collate_fn = collate_fn_text

    def encode(self, text):
        text_encode = [j.encode() for j in text.split(' ')]
        eval_text_encode, eval_len_text = self.encode_function(text_encode)
        return eval_text_encode, eval_len_text

    @staticmethod
    def chunk_letters(filtered_sentence, length):
        """ Search rightward for next space at beginning/end of line

        Args:
            filtered_sentence:
            length:

        Returns:

        """
        try:
            start = random.randint(0, len(filtered_sentence) - length)
            new_start = filtered_sentence[start:].find(" ")+start+1
            fs = filtered_sentence[new_start:new_start+length]
            new_end = fs.rfind(" ")
            fs = fs[:new_end]
        except Exception as e:
            fs = filtered_sentence[:length]
            new_end = fs.rfind(" ") if fs.rfind(" ") > 0 else None
            fs = fs[:new_end].strip()
        return fs

    @staticmethod
    def chunk_words(filtered_sentence, length, force_new_sentence=1, force_end_sentence=1):
        """ Search rightward for next sentence and end of sentence.
        Args:
            filtered_sentence:
            length:
            force_new_sentence [0,1] probability: 1 - force text to begin at the beginning of a sentence
            force_end_sentence [0,1] probability: 1 - force text to end at end of sentence
        Returns:

        It's possible it returns a blank.
        """
        def find_next_period(sentence, index):
            # Incorrect attempt to search within selection for sentence end
            #end = -next(i for i,word in enumerate(filtered_sentence[end:start:-1]) if word[-1] == ".") + end

            try: # Search rightward for next period
                index += next(i for i, word in enumerate(sentence[index:]) if word and word[-1] == ".") + 1
            except StopIteration: # None found
                index = None
            return index

        force_new_sentence = random.random() < force_new_sentence # 1 means force
        force_end_sentence = random.random() < force_end_sentence # 1 means force

        _filtered_sentence = filtered_sentence.split(" ")
        start = random.randint(0, max(0,len(_filtered_sentence) - length))
        if force_new_sentence:
            start = find_next_period(filtered_sentence, start)
        end = start + length if start else length
        if force_end_sentence:
            end = find_next_period(_filtered_sentence, end)

        new_sentence = " ".join(_filtered_sentence[start:end]).strip()
        return new_sentence


    def _get_text(self, sentence, key="text", target_length=None, unit="words"):
        """ Target length is usually a minimum length if we need to force it to end at a sentence end.
            However, the returned sentence can be shorter than the target length, if the initial start point is
                at the end of a document and we force to start at sentence beginning.
                    [Instead of looking forward to start sentence, we could look backward, then target_length would be minimum length.]
                    [We could reject articles that are too short]
        Args:
            sentence:
            key:
            target_length:
            unit:

        Returns:

        """
        if target_length is None:
            target_length = self.max_sentence_length

        text = sentence[key]

        # Remove section headings
        filtered_sentence = replace_symbols(filter_lines_to_sentences(text))

        # Remove OOV symbols
        filtered_sentence = " " + self.filter_sentence(filtered_sentence) + " "
        if unit == "characters":
            fs = self.chunk_letters(filtered_sentence, target_length)
        elif unit == "words":
            fs = self.chunk_words(filtered_sentence, target_length)
        return fs

    def __len__(self):
        return len(self.dataset)

    def get_text(self, idx, unit="words"):
        """ Choose random length sentence satisfying (min,max). After processing, make sure it is still long enough.

        Args:
            idx:
            unit:

        Returns:

        """
        length = random.randint(self.min_sentence_length, self.max_sentence_length)
        while True:
            sentence = self._get_text(self.dataset[idx], target_length=length)
            s = sentence if unit == "characters" else sentence.split()
            s = s[:self.max_sentence_length]
            if len(s) >= self.min_sentence_length: # make sure article is long enough
                break
            else:
                idx = random.randint(0,len(self)) # try a new random article
        return s if unit == "characters" else " ".join(s)

    def __getitem__(self, index):
        text_raw = self.get_text(index)
        text_encode, text_encode_l = self.encode(text_raw)
        return {
            "text_raw": text_raw,
            "text_encoded": text_encode,
            "text_encoded_l": text_encode_l,
        }

class TextDatasetval():

    def __init__(self, base_path = DATASET_PATHS, num_examples = 15, target_transform=None):
        
        self.NUM_EXAMPLES = num_examples
        #base_path = DATASET_PATHS
        file_to_store = open(base_path, "rb")
        self.IMG_DATA = pickle.load(file_to_store)['test']
        self.IMG_DATA  = dict(list( self.IMG_DATA.items()))#[NUM_WRITERS:])
        if 'None' in self.IMG_DATA.keys():
            del self.IMG_DATA['None']
        self.author_id = list(self.IMG_DATA.keys())

        self.transform = get_transform(grayscale=True)
        self.target_transform = target_transform
        
        self.collate_fn = TextCollator()
    

    def __len__(self):
        return len(self.author_id)

    def __getitem__(self, index):

        NUM_SAMPLES = self.NUM_EXAMPLES

        author_id = self.author_id[index]

        self.IMG_DATA_AUTHOR = self.IMG_DATA[author_id]
        random_idxs = np.random.choice(len(self.IMG_DATA_AUTHOR), NUM_SAMPLES, replace = True)

        rand_id_real = np.random.choice(len(self.IMG_DATA_AUTHOR))
        real_img = self.transform(self.IMG_DATA_AUTHOR[rand_id_real]['img'].convert('L'))
        real_labels = self.IMG_DATA_AUTHOR[rand_id_real]['label'].encode()


        imgs = [np.array(self.IMG_DATA_AUTHOR[idx]['img'].convert('L')) for idx in random_idxs]
        labels = [self.IMG_DATA_AUTHOR[idx]['label'].encode() for idx in random_idxs]
       
        max_width = 192 #[img.shape[1] for img in imgs] 
        
        imgs_pad = []
        imgs_wids = []

        for img in imgs:

            img = 255 - img
            img_height, img_width = img.shape[0], img.shape[1]
            outImg = np.zeros(( img_height, max_width), dtype='float32')
            outImg[:, :img_width] = img[:, :max_width]

            img = 255 - outImg

            imgs_pad.append(self.transform((Image.fromarray(img))))
            imgs_wids.append(img_width)

        imgs_pad = torch.cat(imgs_pad, 0) # batch, height, padded_width

        item = {
            'imgs_padded': imgs_pad,
            'img_wids': imgs_wids,
            'img': real_img,
            'label': real_labels,
            'img_path': 'img_path',
            'idx': 'indexes',
            'wcl': index,
            'author_id': author_id
        }

        return item

class WikiCollator(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        {[item['text_raw'] for item in batch],
         [item['img'].shape[2] for item in batch],
         [item['idx'] for item in batch]}
        return batch


class TextCollator(object):
    def __init__(self):
        self.resolution = resolution

    def __call__(self, batch):

        img_path = [item['img_path'] for item in batch]
        width = [item['img'].shape[2] for item in batch]
        indexes = [item['idx'] for item in batch]
        imgs_paddeds =  torch.stack([item['imgs_padded'] for item in batch], 0)
        wcls =  torch.Tensor([item['wcl'] for item in batch])
        author_ids = [item['author_id'] for item in batch]
        img_wids =  torch.Tensor([item['img_wids'] for item in batch])
        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], max(width)], dtype=torch.float32)
        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print(imgs.shape)
        item = {
            'img': imgs,
            'img_path': img_path,
            'idx': indexes,
            'imgs_padded': imgs_paddeds,
            'img_wids': img_wids,
            'wcl': wcls,
            'author_id': author_ids,
        }
        if 'label' in batch[0].keys():
            labels = [item['label'] for item in batch]
            item['label'] = labels
        if 'z' in batch[0].keys():
            z = torch.stack([item['z'] for item in batch])
            item['z'] = z
        return item

