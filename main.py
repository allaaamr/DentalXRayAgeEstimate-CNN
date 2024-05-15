from __future__ import print_function
import numpy as np
import torch_geometric
from timeit import default_timer as timer

## Internal Imports
from DentalXRayDataset import DentalXRayDataset
from model import CustomCNN

### PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, sampler, random_split

def main():
    #hyperparameters for the grid search
    pretrained_models = ['DenseNet201', 'InceptionResNetV2', 'ResNet50', 'VGG16', 'VGG19', 'Xception']
    num_channels_range = range(5, 1001)
    fc_size_range = range(1, 2049)
    learning_rate = 3.24e-4

    best_model = None
    best_loss = float('inf')

    for pretrained_model_name in pretrained_models:
        for num_channels in num_channels_range:
            for fc_size in fc_size_range:
                # Create the model
                model = CustomCNN(pretrained_model_name, num_channels, use_attention=False, fc_size=fc_size)

                # Define the loss function and optimizer
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # Train the model
                model.train()
                for epoch in range(100):
                    # Iterate over the training dataset
                    for inputs, labels in train_loader:
                        # optimizer.zero_grad()
                        # outputs = model(inputs)
                        # loss = criterion(outputs, labels)
                        # loss.backward()
                        # optimizer.step()

                        #1 Forward Pass
                        output = model(inputs)
                        #2. Calculate Loss
                        loss = criterion(output, labels)
                        #3 Optimizer Zero Grad
                        optimizer.zero_grad()
                        #4 Perform back propagation on the loss 
                        loss.backward()
                        #5 Step the optimizer
                        optimizer.step()

                #Testing
                # Evaluate the model
                model.eval()
                with torch.no_grad():
                    test_outputs = model(test_dataset)
                    test_loss = criterion(test_outputs, test_labels)

                # Update the best model if this model has the lowest loss
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_model = model

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