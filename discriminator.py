import torch 
import numpy as np
import torch.nn as nn

import config 

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    channels, height, width = 3, config.IMAGE_SIZE, config.IMAGE_SIZE

    def discriminator_block(in_filters, out_filters, normalize=True):
        """Returns downsampling layers of each
        discriminator block"""
        layers = [nn.Conv2d(in_filters, out_filters,4, stride=2, padding=1)]
        if normalize:
          layers.append(nn.InstanceNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    self.model = nn.Sequential(
                      *discriminator_block(channels,64,normalize=False),
                      *discriminator_block(64, 128),
                      *discriminator_block(128, 256),
                      *discriminator_block(256, 512),
                      nn.ZeroPad2d((1, 0, 1, 0)),
                      nn.Conv2d(512, 1, 4, padding=1)
                      )
    self.apply(config.weights_init_normal)
  
  def forward(self, img):
    return self.model(img)