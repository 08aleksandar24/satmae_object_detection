from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm
from util.datasets import DIORDataset
from DINOv2_features import DINOv2
from models_vit import vit_large_patch16_frcnn
dataset = DIORDataset(root="/storage/local/ssd/dior", split="trainval", transform=None)
loader = DataLoader(dataset, batch_size=1)
import torch
mean = []
std = []

for img, _ in tqdm(loader):
    img = img[0]  # Already a tensor: shape (3, H, W)
    mean.append(torch.mean(img, dim=(1, 2)).numpy())
    std.append(torch.std(img, dim=(1, 2)).numpy())

mean = np.mean(mean, axis=0)
std = np.mean(std, axis=0)

print("DIOR mean:", mean)
print("DIOR std :", std)
