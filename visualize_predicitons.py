import torch
import torchvision
import random
import os
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import functional as F
from DINOv2_features import DINOv2
from util.datasets import DIORDataset
from models_vit import vit_large_patch16_frcnn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from PIL import ImageFont
import matplotlib.font_manager as fm
import albumentations as A
from albumentations.pytorch import ToTensorV2
font_path = fm.findfont(fm.FontProperties(family='DejaVu Sans'))
font = ImageFont.truetype(font_path, size=16)

# 🔧 Settings
NUM_SAMPLES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "/home/aleksandar/satmaeoutputdinofeats/checkpoint-50.pth"
DATASET_ROOT = "/storage/local/ssd/dior"
SAVE_DIR = "/home/aleksandar/visuals/satmaedinofeats"
os.makedirs(SAVE_DIR, exist_ok=True)
model_name = 'DINO'
DIOR_CLASSES = [
    "airplane", "airport", "baseballfield", "basketballcourt", "bridge",
    "chimney", "dam", "Expressway-Service-area", "Expressway-toll-station", "golffield",
    "groundtrackfield", "harbor", "overpass", "ship", "stadium",
    "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill"
]

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.backbone.patch_embed.num_patches
        num_extra_tokens = model.backbone.pos_embed.shape[-2] - num_patches
        # height and width
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)
        # class token and distillation token
        if orig_size != new_size:
            print("Position embedding interpolation from {} to {}".format(orig_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


# ====== Model Load ======
def load_frcnn_model(checkpoint_path, num_classes, model_name):
    if model_name == "DINO":
        backbone = DINOv2("cuda")
    else:
        backbone = vit_large_patch16_frcnn(pretrained=False)
        backbone.out_channels = 1024

    anchor_generator = AnchorGenerator(
        sizes=((2, 4, 8, 16, 32, 64, 128, 256),),  # Multiple scales
        aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),)
    )
    image_mean=[0.485, 0.456, 0.406]
    image_std=[0.229, 0.224, 0.225]
    transform = GeneralizedRCNNTransform(
        min_size=(224,),
        max_size=224,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )
    cropping_transform = A.Compose([
        A.RandomCrop(width=224, height=224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=image_mean, std=image_std),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        transform=cropping_transform,
        min_size=224
    )
    # ✅ Load your checkpoint (which already has trained head)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_model = checkpoint["model"]
    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print("✅ Model loaded. Missing keys:", msg.missing_keys)
    return model.to(DEVICE).eval()

# ====== Dataset & Model ======
dataset = DIORDataset(root=DATASET_ROOT, split="test")
model = load_frcnn_model(CHECKPOINT_PATH, len(DIOR_CLASSES) + 1, model_name)

# ====== Visualization ======
indices = random.sample(range(len(dataset)), NUM_SAMPLES)

for i, idx in enumerate(indices):
    image, target = dataset[idx]
    image_tensor = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(image_tensor)[0]
    print(f"[{idx}] Predicted {len(pred['boxes'])} boxes, top scores: {pred['scores'][:5].cpu().tolist()}")

    keep = pred["scores"] > 0.3
    pred_boxes = pred["boxes"][keep].cpu()
    pred_labels = pred["labels"][keep].cpu()
    pred_classes = [DIOR_CLASSES[i] for i in pred_labels]
    print(f"predicted {[i for i in pred_labels]}")
    gt_boxes = target["boxes"]
    gt_labels = [DIOR_CLASSES[i] for i in target["labels"]]
    print(f'true {[i for i in target["labels"]]}')

    image_pred = draw_bounding_boxes((image * 255).to(torch.uint8), pred_boxes, pred_classes, width=10, colors="red",font=font_path, font_size=30)
    image_gt = draw_bounding_boxes((image * 255).to(torch.uint8), gt_boxes, gt_labels, width=10, colors="green", font=font_path, font_size=30)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(F.to_pil_image(image_gt))
    ax[0].set_title("Ground Truth")
    ax[1].imshow(F.to_pil_image(image_pred))
    ax[1].set_title("Prediction")
    for a in ax:
        a.axis("off")
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, f"sample_{i:04d}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"[✓] Saved: {save_path}")