import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from data_augmentation import *
import torchvision.transforms.functional as F

class listDataset(Dataset):
    def __init__(self, root,shuffle=True, transform=None, batch_size=20, num_workers=4):
     
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        img_list = []
        hmap_list=[]
        offset_list=[]
        size_list=[]
        for i in range(self.batch_size):
            img, heatmap, offset, size = load_data(img_path)
            img = np.transpose(img ,(2,0,1))
            img_list.append(img)
            hmap_list.append(heatmap)
            offset_list.append(offset)
            size_list.append(size)
        img_list = np.array(img_list)
        hmap_list = np.array(hmap_list)
        offset_list = np.array(offset_list)
        size_list = np.array(size_list)
        assert img.shape[0] == 3

        if self.transform is not None:
            img = self.transform(img)
        return torch.from_numpy(img_list.copy()).float(), torch.from_numpy(hmap_list.copy()).float(),torch.from_numpy(offset_list.copy()).float(), torch.from_numpy(size_list.copy()).float()