import itertools
import glob
import numpy as np
from tqdm import tqdm

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataset import Data
from generator import GeneratorResNet
from discriminator import Discriminator

from config import device
import argparse


def generator_train_step(Gs, optimizer, real_A, real_B):
  G_AB, G_BA = Gs
  optimizer.zero_grad()
  # Identity loss
  loss_id_A = criterion_identity(G_BA(real_A), real_A)
  loss_id_B = criterion_identity(G_AB(real_B), real_B)
  loss_identity = (loss_id_A + loss_id_B) / 2
  # GAN loss
  fake_B = G_AB(real_A)
  loss_GAN_AB = criterion_GAN(D_B(fake_B), torch.Tensor(np.ones((len(real_A), 1, 16, 16))).to(device))
  fake_A = G_BA(real_B) 
  loss_GAN_BA = criterion_GAN(D_A(fake_A), torch.Tensor(np.ones((len(real_A), 1, 16, 16))).to(device))
  loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
  # cycle loss
  recov_A = G_BA(fake_B)
  loss_cycle_A = criterion_cycle(recov_A, real_A)
  recov_B = G_AB(fake_A)
  loss_cycle_B = criterion_cycle(recov_B, real_B)
  loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

  loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity
  loss_G.backward()
  optimizer.step()
  return loss_G, loss_identity, loss_GAN, loss_cycle,loss_G, fake_A, fake_B


def discriminator_train_step(D, real_data, fake_data, optimizer):
  optimizer.zero_grad()
  loss_real = criterion_GAN(D(real_data), torch.Tensor(np.ones((len(real_data), 1, 16, 16))).to(device))
  loss_fake = criterion_GAN(D(fake_data.detach()), torch.Tensor(np.zeros((len(real_data), 1, 16, 16))).to(device))
  loss_D = (loss_real + loss_fake) / 2
  loss_D.backward()
  optimizer.step()
  return loss_D

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # positionl Arguments
    parser.add_argument('-e',"--epochs",type=int,required=True, help="number of training epoch")
    parser.add_argument('-tr',"--trdata",type=str,required=True, help="training dataset folder")
    parser.add_argument('-ts',"--tsdata",type=str, help = "testing dataset folder") 
    args = parser.parse_args()

    train_folders = glob.glob(args.trdata+"/**")
    trn_ds = Data(train_folders[0]+'/', train_folders[1]+'/')
    trn_dl = DataLoader(trn_ds, batch_size=1, shuffle=True,collate_fn=trn_ds.collate_fn)

    if args.tsdata !=None:
      test_folders = glob.glob(args.tsdata+"/**")
      val_ds = Data(test_folders[0]+'/',test_folders[1]+'/')
      val_dl = DataLoader(val_ds, batch_size=5, shuffle=True,collate_fn=val_ds.collate_fn)

    G_AB = GeneratorResNet().to(device)
    G_BA = GeneratorResNet().to(device)
    D_A  = Discriminator().to(device)
    D_B  = Discriminator().to(device)

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    
    # configurations
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
    lambda_cyc, lambda_id = 10.0, 5.0

    epochs = args.epochs
    for epoch in range(epochs):
        discriminators_loss = [] 
        generators_loss=[]
        for batch in tqdm(trn_dl):
            real_A, real_B = batch
            loss_G, loss_identity, loss_GAN, loss_cycle, loss_G, fake_A, fake_B = generator_train_step((G_AB,G_BA), optimizer_G, real_A, real_B)
            loss_D_A = discriminator_train_step(D_A, real_A, fake_A, optimizer_D_A)
            loss_D_B = discriminator_train_step(D_B, real_B, fake_B, optimizer_D_B)
            loss_D = (loss_D_A + loss_D_B) / 2
            
            generators_loss.append(loss_G)
            discriminators_loss.append(loss_D)

        print(f"Epoch: {epoch+1}/{epochs}...\t"
                f"Discriminators Loss: {sum(discriminators_loss)/len(discriminators_loss):.3f}...\t"
                f"Generators Loss: {sum(generators_loss)/len(generators_loss):.3f}...\t"
            )
        
        torch.save({"epoch":epoch,
                    "G_AB_state_dict":G_AB.state_dict(),
                    "G_BA_state_dict":G_BA.state_dict(),
                    "D_A_state_dict":D_A.state_dict(),
                    "D_B_state_dict":D_B.state_dict(),
                    "optimizer_G_state_dict":optimizer_G.state_dict(),
                    "optimizer_D_A_state_dict":optimizer_D_A.state_dict(),
                    "optimizer_D_B_state_dict":optimizer_D_B.state_dict(),
                },"./checkpoints/cycle_gan.pth")
