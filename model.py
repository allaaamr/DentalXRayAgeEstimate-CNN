import torch
import torch.nn as nn
import torchvision.models as models

class CustomCNN(nn.Module):
    def __init__(self, pretrained_model_name, num_channels, use_attention, fc_size):
        super(CustomCNN, self).__init__()

        # Loading the pretrained model
        # NEEDS to be updated to NEWW wayy
        if pretrained_model_name == 'DenseNet201':
            self.pretrained_model = models.densenet201(pretrained=True)
        elif pretrained_model_name == 'InceptionResNetV2':
            self.pretrained_model = models.inception_resnet_v2(pretrained=True)
        elif pretrained_model_name == 'ResNet50':
            self.pretrained_model = models.resnet50(pretrained=True)
        elif pretrained_model_name == 'VGG16':
            self.pretrained_model = models.vgg16(pretrained=True)
        elif pretrained_model_name == 'VGG19':
            self.pretrained_model = models.vgg19(pretrained=True)
        elif pretrained_model_name == 'Xception':
            self.pretrained_model = models.xception(pretrained=True)
        else:
            raise ValueError("Invalid pretrained model name")

        # Freeze the pretrained model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # 1x1 convolutional layer
        self.conv1x1 = nn.Conv2d(in_channels=self.pretrained_model.classifier.in_features,
                                  out_channels=num_channels,
                                  kernel_size=1)
        self.flatten = nn.Flatten()
        # Optional attention mechanism
        if use_attention:
            pass

        # Fully connected layers
        self.fc1 = nn.Linear(num_channels, fc_size)
        self.fc2 = nn.Linear(fc_size, 1)

        # Activation function
        self.relu = nn.ReLU()

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(fc_size)


    def forward(self, x):
        features = self.pretrained_model.features(x)
        out = self.conv1x1(out)
        out = self.flatten(out)
        # out = out.view(out.size(0), -1) #flattens the tensor, converting it from a 4D tensor to a 2D tensor with shape (batch_size, num_features) 
        #batch normalization is used after the first layer to make network more stable
        # through normalization of the layer's inputs by recentering and rescaling
        out = self.relu(self.fc1(self.batch_norm(out)))
        out = self.fc2(out)
        return out