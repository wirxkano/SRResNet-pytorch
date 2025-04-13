import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from constants import *
from Utils import Utils

class SRTrainer(nn.Module):
  def __init__(self, network, train_loader, test_loader):
    super().__init__()
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.train_length = len(train_loader)
    self.test_length = len(test_loader)

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.network = network
    self.network.to(self.device)

    self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
    self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=5)
    self.loss_fn = nn.L1Loss()

    self.best_psnr = 0.0

  def train(self, ckpt_dir):
    for epoch in tqdm(range(EPOCHS)):
      running_loss = 0.0

      self.network.train()
      for lr_img, hr_img in self.train_loader:
        lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)

        self.optimizer.zero_grad()
        output = self.network(lr_img)
        loss = self.loss_fn(output, hr_img)
        loss.backward()
        self.optimizer.step()

        running_loss += loss.item()

      train_loss = running_loss / self.train_length
      self.scheduler.step(train_loss)
      print(f'Epoch: {epoch+1}, Train Loss: {train_loss}')

      if (epoch + 1) % 10 == 0:
        current_psnr, current_ssim = self.test()
        print(f'Epoch: {epoch+1}, PSNR: {current_psnr}, SSIM: {current_ssim}')
        if current_psnr > self.best_psnr:
          self.best_psnr = current_psnr
          if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
          torch.save(self.network.state_dict(), os.path.join(ckpt_dir, 'best_model.pth'))

  def test(self):
    psnr_vals = []
    ssim_vals = []

    self.network.eval()
    with torch.no_grad():
      for lr_img, hr_img in self.test_loader:
        lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
        output = self.network(lr_img)
        psnr = Utils.psnr(hr_img, output, test_y_channel=True)
        ssim = Utils.ssim(hr_img, output)
        psnr_vals.append(psnr)
        ssim_vals.append(ssim)

    return np.mean(psnr_vals), np.mean(ssim_vals)

  def restore(self, lr_img):
    self.network.eval()
    with torch.no_grad():
      lr_img = lr_img.to(self.device)
      output = self.network(lr_img)

    return output