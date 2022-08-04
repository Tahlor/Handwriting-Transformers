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
from handwriting.params import *
import re
from handwriting.text_utils import *

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
            'author_ids': author_id
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
    padding2d = set(["text_encoded", ]) # "text_encoded_l"
    for data_key in keys:
        batch_data = [b[data_key] for b in data]

        # pad if it is already a tensor and padding value in dict
        if data_key in padding.keys() and batch_data and torch.is_tensor(batch_data[0]):
            if data_key in padding2d:
                dims = [len(batch_data)]
                for i in range(len(batch_data[0].shape)):
                    dims.append(max([x.shape[i] for x in batch_data]))
                _batch_data = torch.zeros(*dims, dtype=torch.int64) + padding[data_key]
                for i,d in enumerate(batch_data):
                    slices = [i] + [slice(0,x) for x in d.shape]
                    _batch_data[slices] = d
                batch_data = _batch_data
            else:
                batch_data = torch.nn.utils.rnn.pad_sequence(batch_data, batch_first=True, padding_value=padding[data_key])
        output_dict[data_key] = batch_data
    return output_dict



class TextDatasetval():
    """ preset_author: ALL items will be for a specific author, must be set to None to iterate through dataset

        WARNING: DO NOT CHANGE preset_author after Dataloader is created!
                 Keep shuffle on--code assumes shuffle is on here OR in Dataloader, otherwise next(iter()) won't work on Dataloader


    """
    def __init__(self, base_path = DATASET_PATHS,
                 num_examples = 15,
                 target_transform=None,
                 shuffle=True,
                 preset_author=None
                 ):
        
        self.NUM_EXAMPLES = num_examples
        #base_path = DATASET_PATHS
        file_to_store = open(base_path, "rb")
        self.IMG_DATA = pickle.load(file_to_store)['test']
        self.IMG_DATA  = dict(list( self.IMG_DATA.items()))#[NUM_WRITERS:])
        if 'None' in self.IMG_DATA.keys():
            del self.IMG_DATA['None']
        self.author_ids = list(self.IMG_DATA.keys())
        self.author_id_to_index = {v: k for k, v in enumerate(self.author_ids)}

        self.transform = get_transform(grayscale=True)
        self.target_transform = target_transform
        
        self.collate_fn = TextCollator()
        self.preset_author = preset_author
        self.shuffle = shuffle

    def __len__(self):
        return len(self.author_ids)

    def random_author(self):
        idx = random.randint(0, len(self.author_ids))
        return self.author_ids[idx]

    def get(self, author_id=None):
        """

        Args:
            index: Author index

        Returns:

        """
        if not self.preset_author is None:
            author_id = self.preset_author
        elif author_id is None and self.shuffle: # always do a random author if shuffle is on
            author_id = self.random_author()

        index = self.author_id_to_index[author_id]

        NUM_SAMPLES = self.NUM_EXAMPLES


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
            'author_ids': author_id
        }

        return item

    def __getitem__(self, item):
        author_id = self.author_ids[item]
        return self.get(author_id)

    def get_one_author(self, n, author_id=None, same_images=False):
        """ Collate/tile a single item

        Args:
            item:
            n:
            same_images (bool): if True, just use the same style images; otherwise, use different images from same author

        Returns:

        """
        if author_id is None:
            author_id = self.random_author()
        if same_images:
            return self.collate_fn([self.get(author_id)]*n)
        else:
            return self.collate_fn([self.get(author_id) for i in range(n)])

    def truncate(self, batch, n):
        """ Make batch smaller

        Args:
            batch:
            n:

        Returns:

        """
        for key,item in batch.items():
            batch[key] = item[:n]
        return batch


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
        author_ids = [item['author_ids'] for item in batch]
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
            'author_ids': author_ids,
        }
        if 'label' in batch[0].keys():
            labels = [item['label'] for item in batch]
            item['label'] = labels
        if 'z' in batch[0].keys():
            z = torch.stack([item['z'] for item in batch])
            item['z'] = z
        return item

