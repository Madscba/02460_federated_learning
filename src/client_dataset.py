from torch.utils.data import Dataset
import pickle
import os
import numpy as np
from torch import tensor
import torch
from PIL import Image
import wandb
from dataset_utils import load_n_split, relabel_class 

class FemnistDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, user, transform=None, train=True, train_proportion = 0.8, num_classes=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if wandb.config.dataset_path:
            self.root_dir=wandb.config.dataset_path    
        else:
            self.root_dir=os.path.join(os.getcwd(),'dataset/femnist')
        self.imgs_labs = load_n_split(user, self.root_dir, train, train_proportion,num_classes)
        self.transform = transform
    
    def __len__(self):
        return len(self.imgs_labs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                  self.imgs_labs[idx, 0])
        img = Image.open(img_name)
        target=tensor(int(self.imgs_labs[idx, 1]))
        if self.transform:
            img = self.transform(img)

        # image is 3x128x128 but is grayscale so 1x128x128 is more apropriate
        img = img[0, :, :]
        img = img.unsqueeze(0)
        return img, target



