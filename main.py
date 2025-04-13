import argparse
import os
import torch
from torch.utils.data import DataLoader
from constants import *
from Dataset import ImageDataset
from Train import SRTrainer
from Models import SRResNet
from Inference import inference

def main():
  parser = argparse.ArgumentParser(description="SRResNet: Train or Inference Mode")
  parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True,
                      help="Mode to run: 'train' or 'inference'")

  parser.add_argument('--ckpt_dir', type=str, default=None, help="Directory storing checkpoint (.pth file)")
  parser.add_argument('--img_path', type=str, default=None, help="Path to the image to process")
  
  args = parser.parse_args()
  
  if args.mode == 'train':
    if not args.ckpt_dir:
      parser.error("--ckpt_dir is required for train mode")
    
    hr_train_path = os.path.join(FOLDER_PATH, 'DIV2K_train_HR/DIV2K_train_HR')
    hr_test_path = os.path.join(FOLDER_PATH, 'DIV2K_valid_HR/DIV2K_valid_HR')
    
    train_dataset = ImageDataset(hr_train_path)
    test_dataset = ImageDataset(hr_test_path, train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)
    
    model = SRTrainer(SRResNet(num_res_blocks=16), train_loader, test_loader)
    model.train(args.ckpt_dir)
    
  elif args.mode == 'inference':
    if not args.ckpt_dir or not args.img_path:
      parser.error("--ckpt_dir and --img_path are required for inference mode")
    
    ckpt_path = os.path.join(args.ckpt_dir, 'best_model.pth')
    if not os.path.exists(ckpt_path):
      raise FileNotFoundError(f"Checkpoint file (best_model.pth) not found at {ckpt_path}")
    
    model = SRTrainer(SRResNet(num_res_blocks=16), None, None)
    model.network.load_state_dict(torch.load(ckpt_path, weights_only=True))
    inference(model, args.img_path)
  
if __name__ == '__main__':
  main()
