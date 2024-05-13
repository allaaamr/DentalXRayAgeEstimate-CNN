import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class DentalXRayDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, shuffle=None, seed = 7):
        """
		Args:
			csv_file (string): Path to the csv file with annotations.
            root_dir (string): Path to the root directory of images.
            transform (callable, optional): A function/transform that takes in an PIL image
                and returns a transformed version. E.g, ``transforms.ToTensor()``.
		"""
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name)
        age = self.data.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(age, dtype=torch.float32)
