from __future__ import print_function
import numpy as np
import torch_geometric
from timeit import default_timer as timer

## Internal Imports
from DentalXRayDataset import DentalXRayDataset

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, sampler, random_split

def main():
    pass

# Define the transformations you want to apply to your images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    # Add more transformations as needed
])

# Create an instance of custom dataset
dataset = DentalXRayDataset(csv_path = '/Users/alaaabdelazeem/Desktop/Masters/Thesis/Data/filtered_data.csv.zip',
							data_dir= '/Users/alaaabdelazeem/Desktop/Masters/Thesis/Data/features/',
                            transform=transform)

# Calculate the sizes of each split
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

# Use random_split to split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

#DataLoader objects for each split
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

if __name__ == "__main__":
	start = timer()
	results = main()
	end = timer()
	print("finished!")
	print("end script")
	print('Script Time: %f seconds' % (end - start))