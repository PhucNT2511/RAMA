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

total_path = pd.read_csv('/kaggle/working/RAMA/meta_data.csv')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label