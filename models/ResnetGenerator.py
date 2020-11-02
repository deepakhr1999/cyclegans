import torch
from torch import nn

class ResnetBlock(nn.Module):
    """Residual block"""
    
    def __init__(self, dim):
        """Initializes a resnet block
        
        Parameters:
            dim (int) : number channels in the convolution layer
            
        Returns:
            Block of two 3x3 refectionpad-conv-instancenorm layers.
        
        This block learns the residual function.
        Thus the input must have the same channels as the arg dim passed here
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1),
            nn.InstanceNorm2d(dim)
        )
        
    def forward(self, x):
        return self.conv_block(x) + x
    

def ConvBlock(channels_out):
    channels_in = channels_out // 2
    return (
        nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm2d(channels_out),
        nn.ReLU(inplace=True)
    )

def ConvTranposeBlock(channels_out):
    channels_in = channels_out * 2
    return (
        nn.ConvTranspose2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.InstanceNorm2d(channels_out),
        nn.ReLU(True)            
    )


def get_generator():
    model = nn.Sequential(
        # first block uses reflection padding and instance norm
        nn.ReflectionPad2d(3),
        nn.Conv2d(3, 64, kernel_size=7, stride=1),
        nn.InstanceNorm2d(64),
        nn.ReLU(True),
        
        *ConvBlock(128),
        *ConvBlock(256),
        
        # nine residual blocks
        *[ResnetBlock(256) for i in range(9)],
        
        *ConvTranposeBlock(128),
        *ConvTranposeBlock(64),
        
        # last block uses reflection padding but no normalization and tanh
        nn.ReflectionPad2d(3),
        nn.Conv2d(64, 3, kernel_size=7, stride=1),
        nn.Tanh()
    )
    return model