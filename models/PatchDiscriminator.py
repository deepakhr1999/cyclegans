import torch
from torch import nn

class DiscConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out, stride=2, is_first=False):
        super(DiscConvBlock, self).__init__()
        block = (
            nn.Conv2d(channels_in, channels_out, kernel_size=4, stride=stride, padding=1),
            nn.InstanceNorm2d(channels_out),
            nn.LeakyReLU(0.2, True),
        )
        if is_first: # remove the second element
            block = block[0], block[2]
        self.block = nn.Sequential(*block)
        
    def forward(self, x):
        return self.block(x)
    
def get_model():
    """
        Note that there is instance norm for first block
        and the stride is 2 for first 3 blocks and 1 for the last block.
        This is followed by a huge conv layer with again stride=1
    """
    model = nn.Sequential(
        DiscConvBlock(3, 64, is_first=True),
        DiscConvBlock(64, 128),
        DiscConvBlock(128, 256),
        DiscConvBlock(256, 512, stride=1),
        # last block uses 1 channel conv
        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
    )
    return model