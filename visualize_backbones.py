import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from torchvision.utils import save_image
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.datasets import DIORDataset
from DINOv2_features import DINOv2
from models_vit import vit_large_patch16_frcnn

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def visualize_feature_map(feature_map, save_path=None, title=''):
    """Visualize feature map by averaging over channels (activation heatmap)."""
    feature_map = feature_map.squeeze(0)  # (C, H, W)
    activation = feature_map.mean(dim=0).detach().cpu().numpy()

    # Normalize for better contrast
    activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-5)

    plt.figure(figsize=(6, 6))
    plt.imshow(activation)
    plt.title(title)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def load_backbone(model_name, device='cuda'):
    if model_name == "vit_large_patch16":
        model = vit_large_patch16_frcnn(pretrained=True)
    elif model_name == "DINOv2_base_0":
        model = DINOv2(device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    model.to(device)
    model.eval()
    return model


def run_visualization(model_name, save_dir="vis_out", num_images=10, device="cuda"):
    os.makedirs(save_dir, exist_ok=True)

    dataset = DIORDataset(root="/storage/local/ssd/dior", split="test", transform=None)
    indices = random.sample(range(len(dataset)), num_images)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    model = load_backbone(model_name, device)

    for idx, (image, _) in enumerate(tqdm(loader, desc=f"{model_name} Features")):
        image = image.to(device)

        # Resize to ViT-compatible size (multiple of 14)
        if model_name == "DINOv2_base_0":
            h, w = 224, 224
            new_h = (h // 14) * 14
            new_w = (w // 14) * 14
            if new_h != h or new_w != w:
                image = F.resize(image, [new_h, new_w])
        else:
            image = F.resize(image, [224,224])
        image = image.unsqueeze(0)  # add batch

        # Save unnormalized input image
        img_to_save = image.squeeze(0).clone()

        mean = torch.tensor([0.38913488, 0.40021667, 0.36280048], device=img_to_save.device).view(3, 1, 1)
        std = torch.tensor([0.15591773, 0.14450581, 0.14117402], device=img_to_save.device).view(3, 1, 1)
        img_to_save = img_to_save * std + mean
        img_to_save = torch.clamp(img_to_save, 0, 1)
        save_image(img_to_save.cpu(), os.path.join(save_dir, f"img_{idx}.png"))

        # Extract features
        with torch.no_grad():
            image_input = image.squeeze(0) 
            feats = model(image_input)  # {"0": tensor (1, C, H, W)}

        for fmap_i in feats:
            fmap = feats[fmap_i]
            title = f"{model_name} - Img {idx} - Layer {fmap_i}"
            save_path = os.path.join(save_dir, f"{model_name}_img{idx}_layer{fmap_i}.png")
            visualize_feature_map(fmap, save_path=save_path, title=title)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DINOv2_base_0', choices=['DINOv2_base_0', 'vit_large_patch16'])
    parser.add_argument('--save_dir', type=str, default='vis_backbone')
    parser.add_argument('--num_images', type=int, default=10)
    args = parser.parse_args()

    run_visualization(args.model, args.save_dir, args.num_images)
