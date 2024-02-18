import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class FaceDataset(Dataset):
    def __init__(self, paths: list, transforms: transforms.Compose):
        self.paths = paths
        self.transforms = transforms
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        tensor = self.transforms(img)
        return tensor