import glob
import numpy as np
from PIL import Image

import torch 
from torch.utils.data import Dataset

import config

class Data(Dataset):
  def __init__(self,data_A,data_B):
    self.data_A = glob.glob(data_A+"*.jpg")
    self.data_B = glob.glob(data_B+"*.jpg")
  
  def __getitem__(self, index):
      apple = self.data_A[index%len(self.data_A)]
      orange = self.data_B[np.random.randint(len(self.data_B))]
      apple = Image.open(apple).convert('RGB')
      orange = Image.open(orange).convert('RGB')
      return apple,orange

  def __len__(self):
    return max(len(self.data_A),len(self.data_B))

  def collate_fn(self, batch):
    srcs, trgs = list(zip(*batch))
    srcs=torch.cat([config.transform(img)[None] for img in srcs], 0).to(config.device).float()
    trgs=torch.cat([config.transform(img)[None] for img in trgs], 0).to(config.device).float()
    return srcs.to(config.device), trgs.to(config.device)