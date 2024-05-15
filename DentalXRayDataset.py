import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class DentalXRayDataset(Dataset):
    def __init__(self, csv_file:str, root_dir:str, transform=None):
        """
		Args:
			csv_file (string): Path to the csv file with age labels.
            root_dir (string): Path to the root directory of images.
            transform (callable): A function/transform that takes in an PIL image
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

    # def load_image(self, index:int) -> Image.Image:
    #     #Opens an image via a path and returns it
    #    
