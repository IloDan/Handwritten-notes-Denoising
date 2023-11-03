import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    """
    (3x3 conv -> BN -> ReLU) ** 2

    Attributes:
        in_channels (int): number of input channels.
        out_channels (int): number of the output channels.
        kernel_size (int): size of the convolutional kernel.
        padding (int): padding to be applied to the input.
    
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):

        super(DoubleConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)            
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)
    

class Down(nn.Module):
    """
    2x2 MaxPool -> doubleConv

    Attributes:
        in_channels (int): number of input channels.
        out_channels (int): number of the output channels.
        maxpool_kernel_size (int): size of the MaxPooling kernel.
                
    """
    def __init__(self, in_channels: int, out_channels: int):

        super(Down, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.maxpool_kernel_size = 2

        self.double_conv = DoubleConv(in_channels=self.in_channels, out_channels=self.out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=self.maxpool_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_doubleconv = self.double_conv(x)            # out_doubleconv: save the convolution output for the skip connection
        out_maxpool = self.maxpool(out_doubleconv)
        return (out_doubleconv, out_maxpool)


class Up(nn.Module):
    """
    2x2 upconv -> concatenate --> doubleConv
    """

    def __init__(self, in_channels: int, out_channels: int):
        
        super(Up, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upconv_kernel_size = 2

        self.upconv = nn.ConvTranspose2d(in_channels=self.in_channels-self.out_channels, out_channels=self.in_channels-self.out_channels, kernel_size=self.upconv_kernel_size, stride=2)
        self.double_conv = DoubleConv(in_channels=self.in_channels, out_channels=self.out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)

        # resize x if is not the same size as skip
        if x.shape != skip.shape:
            x = TF.resize(x, size=skip.shape[2:], antialias=True)
        
        x = torch.cat([x, skip], axis = 1)
        return self.double_conv(x)


class UNet(nn.Module):
    """
    UNet model for segmentation of images with num_classes classes     
    """
    def __init__(self, input_channels: int = 3, num_classes: int = 2):

        super(UNet, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        # downsampling
        self.down_conv1 = Down(in_channels=self.input_channels, out_channels=64)         
        self.down_conv2 = Down(in_channels=64, out_channels=128)          
        self.down_conv3 = Down(in_channels=128, out_channels=256)            
        self.down_conv4 = Down(in_channels=256, out_channels=512)

        # bottleneck           
        self.double_conv = DoubleConv(in_channels=512, out_channels=1024)       

        # upsampling
        self.up_conv1 = Up(in_channels=(512 + 1024), out_channels=512)
        self.up_conv2 = Up(in_channels=(256 + 512), out_channels=256)
        self.up_conv3 = Up(in_channels=(128 + 256), out_channels=128)
        self.up_conv4 = Up(in_channels=(64 + 128), out_channels=64)

        # final
        self.up_conv5 = nn.Conv2d(in_channels=64, out_channels=self.num_classes, kernel_size=1)   
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        downsampling -> bottleneck -> upsampling
        """
        
        skip_1, x = self.down_conv1(x)  # skip: output of the double convolution, x: output of the maxpool
        skip_2, x = self.down_conv2(x)
        skip_3, x = self.down_conv3(x)
        skip_4, x = self.down_conv4(x)

        x = self.double_conv(x)

        x = self.up_conv1(x, skip_4)
        x = self.up_conv2(x, skip_3)
        x = self.up_conv3(x, skip_2)
        x = self.up_conv4(x, skip_1)
        x = self.up_conv5(x)
        x = torch.sigmoid(x)
        
        return x


