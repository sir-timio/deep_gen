import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import torch.nn as nn
from PIL import Image
class AnomalyDataset(Dataset):
    def __init__(self, paths: list, transforms: transforms.Compose):
        self.paths = paths
        self.transforms = transforms
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('L')
        tensor = self.transforms(img)
        return tensor


class TestAnomalyDataset(Dataset):
    def __init__(self, paths: list, labels: list, transforms: transforms.Compose):
        self.paths = paths
        self.labels = labels
        self.transforms = transforms
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('L')
        label = self.labels[i]
        tensor = self.transforms(img)
        return tensor, label