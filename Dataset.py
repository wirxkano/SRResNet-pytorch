import os
from constants import *
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

train_preprocess = transforms.Compose([
  transforms.RandomCrop(PATCH_SIZE),
  transforms.RandomHorizontalFlip(),
  transforms.RandomVerticalFlip()
])

class ImageDataset(Dataset):
  def __init__(self, hr_dir, train=True):
    self.hr_dir = hr_dir
    self.images_length = len(os.listdir(self.hr_dir))
    self.train = train

  def __len__(self):
    return self.images_length

  def __getitem__(self, index):
    hr_path = os.path.join(self.hr_dir, os.listdir(self.hr_dir)[index])
    hr_img = Image.open(hr_path).convert('RGB')

    if self.train:
      hr_img = hr_img.resize((520, 520), Image.BICUBIC)
      hr_img = train_preprocess(hr_img)
    else:
      hr_img = transforms.Resize((PATCH_SIZE, PATCH_SIZE), Image.BICUBIC)(hr_img)

    lr_img = hr_img.resize((hr_img.width // 4, hr_img.height // 4), Image.BICUBIC)

    lr_img = transforms.ToTensor()(lr_img)
    hr_img = transforms.ToTensor()(hr_img)

    return lr_img, hr_img
