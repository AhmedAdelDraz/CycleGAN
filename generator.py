
import torch 
import torch.nn as nn

import config


class ResidualBlock(nn.Module):
  def __init__(self, in_features):
    super(ResidualBlock, self).__init__()
    self.block = nn.Sequential(
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      )
  def forward(self, x):
    return x + self.block(x)


class GeneratorResNet(nn.Module):
  def __init__(self,num_residual_blocks=9):
    super(GeneratorResNet, self).__init__()
    out_features = 64
    channels = 3
    model = [nn.ReflectionPad2d(3),
             nn.Conv2d(channels, out_features, 7),
             nn.InstanceNorm2d(out_features),
             nn.ReLU(inplace=True),]
    
    in_features = out_features
    # Downsampling
    for _ in range(2):
      out_features *= 2
      model += [ nn.Conv2d(in_features, out_features, 3,stride=2, padding=1),
                 nn.InstanceNorm2d(out_features),
                 nn.ReLU(inplace=True),]
      in_features = out_features
    
    # Residual blocks
    for _ in range(num_residual_blocks):
      model += [ResidualBlock(out_features)]
    
    # Upsampling
    for _ in range(2):
      out_features //= 2
      model += [nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),]
      in_features = out_features

    # Output layer
    model += [nn.ReflectionPad2d(channels), 
              nn.Conv2d(out_features, channels, 7),
              nn.Tanh()]
    self.model = nn.Sequential(*model)
    self.apply(config.weights_init_normal)       ##initialize the model weights

  def forward(self, x):
    return self.model(x)