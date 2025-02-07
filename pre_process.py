import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models 
from PIL import Image
import os
import random
import numpy as np
import pandas as pd

total_path = pd.read_csv('RAMA/meta_data.csv')

class ImageDataset(Dataset):
    def __init__(self, mode = "train", transform=None):
        self.path = []
        self.label=[]
        self.transform = transform

        if mode == "train":
            self.path = total_path[total_path['train']==1]['path'].values
            self.label = total_path[total_path['train']==1]['label'].values
        elif mode == 'test':
            self.path = total_path[total_path['train']==0]['path'].values
            self.label = total_path[total_path['train']==0]['label'].values

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        path = self.path[idx]
        label = self.label[idx]
        path = os.path.join('RAMA',path)
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label