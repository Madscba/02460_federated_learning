from torch.utils.data import Dataset
import pickle
import os
import numpy as np
from torch import tensor
import torch
from PIL import Image
from dataset_utils import load_n_split, relabel_class 

class FemnistDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, user, root_dir, transform=None, train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.imgs_labs = load_n_split(user, root_dir, train)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.imgs_labs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                  self.imgs_labs[idx, 0])
        img = Image.open(img_name)
        target=relabel_class(self.imgs_labs[idx, 1])
        if self.transform:
            img = self.transform(img)

        return img, tensor(target)



