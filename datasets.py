# We heavily borrow code from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.train_hf = h5py.File(data_folder + '/train36.hdf5', 'r')
        self.train_features = self.train_hf['image_features']
        self.val_hf = h5py.File(data_folder + '/val36.hdf5', 'r')
        self.val_features = self.val_hf['image_features']

        
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
#        self.imgs = self.h['images']
        self.imgs_opt = self.h['opticalflow']
        
        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Load bottom up image features distribution
        with open(os.path.join(data_folder, self.split + '_GENOME_DETS_' + data_name + '.json'), 'r') as j:
            self.objdet = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
       # img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)

        objdet = self.objdet[i // self.cpi]

        # Load bottom up image features
        if objdet[0] == "v":
            img_bottomup = torch.FloatTensor(self.val_features[objdet[1]])
        else:
            img_bottomup = torch.FloatTensor(self.train_features[objdet[1]])

        

        img_opt = torch.FloatTensor(self.imgs_opt[i // self.cpi] /255.)

        #data = {'image': img, 'image_opticalflow': img_opt}
        
        if self.transform is not None:
            img_opt = self.transform(img_opt)
        
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])


        if self.split is 'TRAIN':
            return img_opt, img_bottomup, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img_opt, img_bottomup, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
