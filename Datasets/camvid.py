import os
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from Datasets.seg_transfo import SegTransformCompose, ToTensor


class CamVid(torch.utils.data.Dataset):
    """Class to load the CamVid Dataset"""

    colors = np.array([
                [128, 128, 128],          # sky
                [128,   0,   0],          # building
                [192, 192, 128],          # column_pole
                [128,  64, 128],          # road
                [  0,   0, 192],          # sidewalk
                [128, 128,   0],          # Tree
                [192, 128, 128],          # SignSymbol
                [ 64,  64, 128],          # Fence
                [ 64,   0, 128],          # Car
                [ 64,  64,   0],          # Pedestrian
                [  0, 128, 192],          # Bicyclist
                [  0,   0,   0],          # Void
                [ 60, 250, 240]])         # Anomaly
    
    cmap = dict(zip(range(len(colors)), colors))

    class_name = ["Sky", "Building", "Pole", "Road", "Sidewalk", "Tree", "SignSymbol",
                  "Fence", "Car", "Pedestrain", "Bicyclist", "Void", "Anomaly"]

    def __init__(self, imgFolder, split, transforms=SegTransformCompose(ToTensor())):
        super(CamVid, self).__init__()
        self.imgFolder = imgFolder
        self.split = split
        with open(os.path.join(imgFolder, split+".txt")) as f:
            self.img_set = np.array([l.split() for l in f.readlines()])
        self.x = self.img_set[:, 0]
        self.y = self.img_set[:, 1]
        self.transforms = transforms

    def __len__(self):
        return len(self.img_set)

    def __getitem__(self, id):
        filex = os.path.join(self.imgFolder, self.x[id])
        filey = os.path.join(self.imgFolder, self.y[id])

        imgx = Image.open(filex)
        imgy = Image.open(filey)

        imgx, imgy = self.transforms(imgx, imgy)

        return imgx, imgy
