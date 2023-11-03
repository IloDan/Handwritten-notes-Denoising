import torch
import torch.nn.init as init
import torch.nn as nn

from model import UNet


# Create and initialize three different instances of the UNet model
for i in range(3):
    model = UNet(input_channels=3, num_classes=2)

    # Initialize weights using Kaiming initialization
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    # Apply weight initialization to the model
    model.apply(initialize_weights)

    # Save the weights with different filenames
    filename = f'kaiming_unet_{i}.pth'
    torch.save(model.state_dict(), filename)