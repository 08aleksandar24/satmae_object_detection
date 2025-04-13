# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
import torch.nn.functional as F
import timm.models.vision_transformer
from util.pos_embed import get_2d_sincos_pos_embed
from util.pos_embed import interpolate_pos_embed
class ViTBackbone(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_channels = 1024  # Important for FasterRCNN to work

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, C, H, W) → Flattened
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (B, 197, C)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Remove CLS token
        x = x[:, 1:, :]  # Shape: (B, 196, 1024)

        # 🔁 Reshape to (B, C, H, W)
        h = w = int(x.shape[1] ** 0.5)
        x = x.permute(0, 2, 1).contiguous().view(B, -1, h, w)  # → (B, 1024, 14, 14)

        return {"0": x}  # ✅ Return as dict for FasterRCNN

def vit_large_patch16_frcnn(pretrained=True):
    model = ViTBackbone(
        img_size=224,
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, 
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)
    )
    
    if pretrained:
        checkpoint_path = "/home/aleksandar/pre_train/last_vit_l_rvsa_ss_is_rd_pretrn_model_encoder.pth" 
        #checkpoint_path = "/home/aleksandar/satmaepretrain/fmow_pretrain.pth" 
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print("[DEBUG X1]")
        print("Checkpoint keys:", checkpoint.keys())  # Debugging step

        # Interpolate position embeddings
        if "pos_embed" in checkpoint["state_dict"]:
            print("[DEBUG X2]", "pos_embed")
            old_pos_embed = checkpoint["state_dict"]["pos_embed"]  # Shape: [1, 784, 1024]
            new_pos_embed = model.pos_embed  # Expected shape: [1, 197, 1024]

            num_tokens_old = old_pos_embed.shape[1]  # 784 in checkpoint
            num_tokens_new = new_pos_embed.shape[1]  # 197 in model

            if num_tokens_old != num_tokens_new:
                print(f"Resizing pos_embed from {num_tokens_old} to {num_tokens_new}")
                cls_token_embed = old_pos_embed[:, 0, :].unsqueeze(1)  # Keep the CLS token
                old_grid_embed = old_pos_embed[:, 1:, :]  # Remove CLS token

                # **Fix: Compute closest valid grid size**
                gs_old_h = int((num_tokens_old - 1) ** 0.5)  # Approximate height
                gs_old_w = (num_tokens_old - 1) // gs_old_h  # Compute width to avoid floating point issues

                gs_new = int((num_tokens_new - 1) ** 0.5)  # Target new size

                print(f"Old grid size: {gs_old_h}x{gs_old_w}, New grid size: {gs_new}x{gs_new}")

                # Reshape for interpolation
                old_grid_embed = old_grid_embed.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)  # (1, C, H, W)
                new_grid_embed = F.interpolate(old_grid_embed, size=(gs_new, gs_new), mode='bicubic', align_corners=False)
                new_grid_embed = new_grid_embed.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)

                # Reconstruct pos_embed with CLS token
                checkpoint["state_dict"]["pos_embed"] = torch.cat([cls_token_embed, new_grid_embed], dim=1)

        # Load weights
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    return model




class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        # Added by Samar, need default pos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, 
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model