import itertools
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid


from dataset import Data
from generator import GeneratorResNet
from discriminator import Discriminator

from config import device,test_transforms


plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torch.squeeze(img).cpu().numpy()
        img = np.moveaxis(img,0,2)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])




@torch.no_grad()
def generate_sample(real_A,real_B):
    G_AB.eval()
    G_BA.eval()
    fake_B = G_AB(real_A)
    fake_A = G_BA(real_B)
    return fake_A,fake_B

def load_image(image_path):
    image = Image.open(image_path)
    image = test_transforms(image)
    image = torch.unsqueeze(image,0)
    return image.to(device)



if __name__ == "__main__":
    G_AB = GeneratorResNet().to(device)
    G_BA = GeneratorResNet().to(device)
    D_A  = Discriminator().to(device)
    D_B  = Discriminator().to(device)
    
    meta = torch.load("./checkpoints/cycle_gan.pth")
    
    G_AB.load_state_dict(meta["G_AB_state_dict"])
    G_BA.load_state_dict(meta["G_BA_state_dict"])
    D_A.load_state_dict(meta["D_A_state_dict"])
    D_B.load_state_dict(meta["D_B_state_dict"])

    # real_A = load_image("path")
    # real_B = load_image("path")

    # fake_A,fake_B = generate_sample(real_A,real_B)
    # grid = make_grid([real_A, fake_A, real_B, fake_B],nrow=4)
    # show([real_A, fake_A, real_B, fake_B])