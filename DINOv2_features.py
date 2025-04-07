import re
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

class DINOv2(nn.Module):

    def __init__(self, device) -> None:
        super().__init__()
        # self.model_size = model_args["model_size"]
        self.model_size = 'base'

        if self.model_size == "small":
            self.feat_extr = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        if self.model_size == "small_reg":
            self.feat_extr = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14_reg"
            )
        if self.model_size == "base":
            self.feat_extr = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        # if self.model_size == "base":
        #     self.feat_extr = torch.hub.load(
        #         "/home/filip/pretrained_weights/", "vitl16_reg4_SimDINOv2_100ep.pth"
        #     )
        if self.model_size == "base_reg":
            self.feat_extr = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14_reg"
            )
        if self.model_size == "large":

            # def revert_block_chunk_weight(state_dict):
            #     # convert blocks.chunkid.id.* to blocks.id.*: blocks.3.22. to blocks.22.
            #     return {
            #         re.sub(r"blocks\.(\d+)\.(\d+)\.", r"blocks.\2.", k): v
            #         for k, v in state_dict.items()
            #     }

            # ckpt = torch.load(
            #     "/home/filip/pretrained_weights/vitl16_reg4_SimDINOv2_100ep.pth",
            #     map_location="cpu",
            # )["teacher"]

            # ckpt = {
            #     k.removeprefix("backbone."): v
            #     for k, v in ckpt.items()
            #     if k.startswith("backbone")
            # }
            # ckpt = revert_block_chunk_weight(ckpt)
            # # ckpt = timm.models.vision_transformer.checkpoint_filter_fn(ckpt, model)

            # print(timm.list_models(pretrained=True))

            # self.feat_extr = timm.models.vision_transformer.VisionTransformer(
            #     embed_dim=1024,
            #     depth=24,
            #     num_heads=16,
            #     mlp_ratio=4,
            #     qkv_bias=True,
            #     norm_layer=partial(nn.LayerNorm, eps=1e-6),
            # )
            # self.feat_extr.load_state_dict(ckpt)
            self.feat_extr = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
            # self.feat_extr.load_state_dict(
            #     torch.load(
            #         "/home/filip/pretrained_weights/vitl16_reg4_SimDINOv2_100ep.pth"
            #     )
            # )
        # f self.model_size == "large":
        #     self.feat_extr = torch.hub.load(
        #         "/home/filip/pretrained_weights/",
        #         "vitl16_reg4_SimDINOv2_100ep.pth",
        #         source="local",
        #     )i
        self.embed_dim = self.feat_extr.embed_dim
        self.model = self.feat_extr
        self.return_interm_layers = False
        self.feat_extr.eval()

    def forward(self, x):
        #if x.shape[-1] != 518:
         #   x = F.interpolate(x, size=(518, 518), mode='bilinear', align_corners=False)
        # Get current height/width (assuming square input)
        h, w = x.shape[-2], x.shape[-1]

        # Compute nearest lower multiples of 14
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14

        # Only interpolate if the size is not already valid
        if new_h != h or new_w != w:
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            out = self.model.forward_features(x)
            tokens = out['x_norm_patchtokens']  # (B, N, C)
            B, N, C = tokens.shape
            H = W = int(N ** 0.5)
            tokens = tokens.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W)
        return {"0": tokens}
    @property
    def out_channels(self):
        return self.embed_dim