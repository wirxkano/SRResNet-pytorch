import numpy as np
import cv2
from skimage.metrics import structural_similarity as skl_ssim

class Utils:
  @staticmethod
  def show_img(img):
    img = img.permute(1, 2, 0)
    img = img.cpu().numpy()
    return img
  
  @staticmethod
  def psnr(img1, img2, crop_border=4, test_y_channel=False):
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()

    psnr_values = []
    batch = img1.shape[0] if img1.shape[0] > 1 else 1
    for i in range(batch):
      img1_i = img1[i].transpose(1, 2, 0)
      img2_i = img2[i].transpose(1, 2, 0)

      img1_i = img1_i.astype(np.float32)
      img2_i = img2_i.astype(np.float32)

      if test_y_channel:
        img1_i = cv2.cvtColor(img1_i, cv2.COLOR_RGB2YCrCb)[:, :, 0]
        img2_i = cv2.cvtColor(img2_i, cv2.COLOR_RGB2YCrCb)[:, :, 0]

      if crop_border != 0:
          img1_i = img1_i[crop_border:-crop_border, crop_border:-crop_border, ...]
          img2_i = img2_i[crop_border:-crop_border, crop_border:-crop_border, ...]

      mse = np.mean((img1_i - img2_i)**2)
      if mse == 0:
          psnr_values.append(float('inf'))
      else:
          psnr_values.append(20. * np.log10(1. / np.sqrt(mse)))

    return np.mean(psnr_values)

  @staticmethod
  def ssim(img1, img2):
    img1_np = img1.detach().cpu().numpy()  # (32, 3, 512, 512)
    img2_np = img2.detach().cpu().numpy()

    ssim_values = []
    batch = img1_np.shape[0] if img1_np.shape[0] > 1 else 1
    for i in range(batch):
        img1_np_i = np.transpose(img1_np[i], (1, 2, 0))
        img2_np_i = np.transpose(img2_np[i], (1, 2, 0))
        ssim_values.append(skl_ssim(img1_np_i, img2_np_i, data_range=img2_np_i.max() - img2_np_i.min(), channel_axis=2))
    return np.mean(ssim_values)
