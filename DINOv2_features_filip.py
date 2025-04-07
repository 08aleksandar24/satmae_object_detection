import re
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from UPerNet.UPerNetHead import UperNetHead
from functools import partial

from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerForUniversalSegmentation,
)


class DINOv2(nn.Module):

    def __init__(self, args, device) -> None:
        super().__init__()
        # self.model_size = model_args["model_size"]
        self.model_size = args.model.split("_")[0]
        self.conv_size = int(args.model.split("_")[1])

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
        self.feat_extr.eval()  # type: ignore
        self.feat_extr.to(device)  # type: ignore
        self.device = device
        self.patch_size = 14

        # upernet stuff
        if self.model_size == "small":
            self.embed_dim = 384
            feature_channels = [
                self.embed_dim + self.conv_size,
                self.embed_dim,
            ]
        elif self.model_size == "base":
            self.embed_dim = 768
            feature_channels = [
                self.embed_dim + self.conv_size,
                self.embed_dim,
                self.embed_dim,
                self.embed_dim,
            ]
        else:
            self.embed_dim = 1024
            feature_channels = [
                self.embed_dim + self.conv_size,
                self.embed_dim,
                self.embed_dim,
                self.embed_dim,
            ]

        fpn_out = self.embed_dim + self.conv_size
        self.input_size = (args.input_size, args.input_size)
        self.num_patches = int(self.input_size[0] / self.patch_size)

        self.do_interpolation = False

        if args.input_size % 14 != 0:
            self.do_interpolation = True

        if args.dataset_type == "euro_sat" or args.dataset_type == "rgb":
            self.task = "classification"
            # self.classifier = LinearClassifier(
            #     self.embed_dim, self.num_patches, self.num_patches, args.nb_classes
            # )
            self.classification_head = nn.Linear(self.embed_dim, args.nb_classes)
        else:
            self.task = "segmentation"

        # self.PPN = PSPModule(feature_channels[-1])
        # self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        # self.head = nn.Conv2d(fpn_out, args.nb_classes, kernel_size=3, padding=1)
        self.up_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        config = {
            "pool_scales": [1, 2, 3, 6],
            "hidden_size": 512,
            "num_labels": args.nb_classes,
            "initializer_range": 0.02,
        }

        self.upernet_head = UperNetHead(config, feature_channels)

        if self.conv_size > 0:
            if args.dataset_type == "spacenet":
                self.up = nn.Upsample(
                    size=(64, 64), mode="bilinear", align_corners=True
                )
            elif (
                args.dataset_type == "sen1floods11"
                or args.dataset_type == "vaihingen"
                or args.dataset_type == "potsdam"
            ):
                self.up = nn.Upsample(
                    size=(144, 144), mode="bilinear", align_corners=True
                )
            elif args.dataset_type == "isaid":
                self.up = nn.Upsample(
                    size=(256, 256), mode="bilinear", align_corners=True
                )
            elif args.dataset_type == "mass_roads":
                self.up = nn.Upsample(
                    size=(428, 428), mode="bilinear", align_corners=True
                )
            # elif  args.dataset_type == "rgb":

        # self.conv = nn.Conv2d(
        #     in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1
        # )  # 3x3 kernel, stride 2, padding 1
        # self.bn = nn.BatchNorm2d(256)
        # self.relu = nn.ReLU()

        # self.conv = nn.Conv2d(
        #     in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1
        # )
        # self.bn = nn.BatchNorm2d(32)
        # self.relu = nn.ReLU()

        # self.classifier = LinearClassifier(
        #     self.embed_dim, self.num_patches, self.num_patches, args.nb_classes
        # )

        self.swin_encoder = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-large-cityscapes-semantic"
        ).model.pixel_level_module.encoder

        self.swin_encoder.to(device)

        self.linear_layer = nn.Linear(7, 3)

        self.linear_layer.to(device)

        if self.conv_size == 32:
            self.conv_layers = nn.Sequential(
                # Conv1: Input [B, 3, 224, 224] -> Output [B, 64, 112, 112]
                nn.Conv2d(
                    in_channels=7, out_channels=16, kernel_size=7, stride=2, padding=3
                ),  # Kernel size 7x7, stride 2, padding 3
                nn.BatchNorm2d(16),
                nn.ReLU(),
                # Conv2: Input [B, 64, 112, 112] -> Output [B, 128, 56, 56]
                nn.Conv2d(
                    in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
                ),  # Kernel size 3x3, stride 2, padding 1
                nn.BatchNorm2d(32),
                nn.ReLU(),
                # nn.Conv2d(
                #     in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
                # ),  # Kernel size 3x3, stride 2, padding 1
                # nn.BatchNorm2d(64),
                # nn.ReLU(),
                # nn.Upsample(size=(144, 144), mode="bilinear", align_corners=False),
            )
        elif self.conv_size == 256:
            self.conv_layers = nn.Sequential(
                # Conv1: Input [B, 3, 224, 224] -> Output [B, 64, 112, 112]
                nn.Conv2d(
                    in_channels=7, out_channels=64, kernel_size=7, stride=2, padding=3
                ),  # Kernel size 7x7, stride 2, padding 3
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # Conv2: Input [B, 64, 112, 112] -> Output [B, 128, 56, 56]
                nn.Conv2d(
                    in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
                ),  # Kernel size 3x3, stride 2, padding 1
                nn.BatchNorm2d(128),
                nn.ReLU(),
                # Conv3: Input [B, 128, 56, 56] -> Output [B, 256, 56, 56]
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # Kernel size 3x3, stride 1, padding 1
                nn.BatchNorm2d(256),
                nn.ReLU(),
                # Conv4: Input [B, 256, 56, 56] -> Output [B, 512, 56, 56]
                # nn.Conv2d(
                #     in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
                # ),  # Kernel size 3x3, stride 1, padding 1
                # nn.BatchNorm2d(512),
                # nn.ReLU(),
                # Conv5: Input [B, 512, 56, 56] -> Output [B, 1024, 56, 56]
                # nn.Conv2d(
                #     in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1
                # ),  # Kernel size 3x3, stride 1, padding 1
                # nn.BatchNorm2d(1024),
                # nn.ReLU(),
            )

    def get_features(self, imgs):
        # layer = self.layer_num[0] # TODO: make it a list
        # layers = []
        if self.do_interpolation:
            if imgs.shape[-1] == 512:
                imgs = F.interpolate(
                    imgs, size=504, mode="bilinear", align_corners=True
                )
            elif imgs.shape[-1] == 64:
                imgs = F.interpolate(imgs, size=56, mode="bilinear", align_corners=True)
            elif imgs.shape[-1] == 256:
                imgs = F.interpolate(
                    imgs, size=252, mode="bilinear", align_corners=True
                )
            elif imgs.shape[-1] == 1500:
                imgs = F.interpolate(
                    imgs, size=1498, mode="bilinear", align_corners=True
                )

        with torch.no_grad():
            if self.task == "classification":
                out = self.feat_extr.forward_features(imgs)  # type: ignore
                cls = out["x_norm_clstoken"]
                out = cls
            else:
                # if self.layer_num == "last":
                if self.model_size == "base" or self.model_size == "small":
                    patch = self.feat_extr.get_intermediate_layers(imgs, (3, 5, 8, 11))  # type: ignore
                else:
                    patch = self.feat_extr.get_intermediate_layers(imgs, (3, 9, 17, 23))  # type: ignore
                out = patch

            # layers.append(patch)
            # layers.append(patch)
            # cls = out["x_norm_clstoken"]
            # elif self.layer_num == "first":

        # elif self.layer_num == "avg":
        #     pass
        return out

    def encoder_conv(self, x):

        conv_embeds = self.conv_layers(x)
        conv_embeds = self.up(conv_embeds)

        return conv_embeds

    def decoder_upernet(self, features, conv_embeds):

        # conv_1 = self.relu(self.bn(self.conv(conv_embeds)))
        # conv_2 = self.relu(self.bn(self.conv(conv_1)))
        # conv_3 = self.relu(self.bn(self.conv(conv_2)))
        new_features = []

        new_features.append(
            features[0].reshape(-1, self.num_patches, self.num_patches, self.embed_dim)
        )
        new_features.append(
            features[1].reshape(-1, self.num_patches, self.num_patches, self.embed_dim)
        )
        new_features.append(
            features[2].reshape(-1, self.num_patches, self.num_patches, self.embed_dim)
        )
        new_features.append(
            features[3].reshape(-1, self.num_patches, self.num_patches, self.embed_dim)
        )

        new_features[0] = torch.permute(new_features[0], (0, 3, 1, 2))
        new_features[1] = torch.permute(new_features[1], (0, 3, 1, 2))
        new_features[2] = torch.permute(new_features[2], (0, 3, 1, 2))
        new_features[3] = torch.permute(new_features[3], (0, 3, 1, 2))

        new_features[-1] = F.interpolate(
            new_features[-1], scale_factor=0.5, mode="bilinear", align_corners=True
        )
        # features[2] = self.up_1(features[2])
        new_features[1] = self.up_1(new_features[1])
        new_features[0] = self.up_2(new_features[0])
        if self.conv_size > 0:
            new_features[0] = torch.cat((new_features[0], conv_embeds), 1)
            # new_features[1] = torch.cat((new_features[1], conv_1), 1)
            # new_features[2] = torch.cat((new_features[2], conv_2), 1)
            # new_features[3] = torch.cat((new_features[3], conv_3), 1)

        # features[0] = features[0] + conv_embeds

        # new_features[-1] = self.PPN(new_features[-1])
        # x = self.head(features[-1])
        # x = self.head(self.FPN(new_features))

        x = self.upernet_head(new_features)

        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return x

    def decoder_linear(self, x, conv_embeds):
        x = self.classifier(x)
        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return x

    def forward(self, x):

        chunks = torch.split(x, [3, 7], dim=1)

        swin_embeds = self.swin_encoder(self.linear_layer(chunks[1]))

        # conv_embeds = 0
        # if self.conv_size > 0:
        #     conv_embeds = self.encoder_conv(chunks[1])
        # x = self.encoder_forward(x)
        # x = self.decoder_upernet(x, conv_embeds)
        features = self.get_features(chunks[0])
        # x = self.encoder_forward(x)
        x = self.decoder_upernet(features, swin_embeds)
        # new_features = []

        # new_features.append(
        #     features[0].reshape(-1, self.num_patches, self.num_patches, self.embed_dim)
        # )
        # new_features.append(
        #     features[1].reshape(-1, self.num_patches, self.num_patches, self.embed_dim)
        # )
        # new_features.append(
        #     features[2].reshape(-1, self.num_patches, self.num_patches, self.embed_dim)
        # )
        # new_features.append(
        #     features[3].reshape(-1, self.num_patches, self.num_patches, self.embed_dim)
        # )

        # new_features[0] = torch.permute(new_features[0], (0, 3, 1, 2))
        # new_features[1] = torch.permute(new_features[1], (0, 3, 1, 2))
        # new_features[2] = torch.permute(new_features[2], (0, 3, 1, 2))
        # new_features[3] = torch.permute(new_features[3], (0, 3, 1, 2))
        # x = self.upernet_head(new_features)
        # x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)

        # x = self.classification_head(features)
        # x = self.decoder_linear(features[-1], conv_embeds)

        # x = self.decoder_upernet(x[1])

        return x, (conv_embeds, features[-1])