import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from Utils import Utils

def inference(model, image_path):
  img = Image.open(image_path).convert("RGB")

  transform = transforms.Compose([
        transforms.ToTensor()
  ])
  
  img_lr = transform(img).unsqueeze(0)

  img_bi = F.interpolate(img_lr, scale_factor=4, mode='bicubic', align_corners=False)
  
  img_sr = model.restore(img_lr)

  f, axarr = plt.subplots(1,2,figsize=(20, 8))
  axarr[0].title.set_text('Bicubic Image')
  axarr[1].title.set_text('SR Image')
  axarr[0].imshow(Utils.show_img(img_bi.squeeze(0)))
  axarr[1].imshow(Utils.show_img(img_sr.squeeze(0)))
  axarr[0].axis('off')
  axarr[1].axis('off')
  