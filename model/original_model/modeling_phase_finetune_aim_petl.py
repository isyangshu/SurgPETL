import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange
from collections import OrderedDict
import clip


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 7,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        **kwargs,
    }


class Adapter(nn.Module):
    def __init__(
        self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True
    ):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: torch.Tensor = None,
        scale=1.0,
        num_tadapter=1,
        num_frames=8,
        drop_path=0.0,
    ):
        super().__init__()
        self.num_tadapter = num_tadapter
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head

        self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        self.S_Adapter = Adapter(d_model)
        self.scale = scale
        self.T_Adapter = Adapter(d_model, skip_connect=False)
        if num_tadapter == 2:
            self.T_Adapter_in = Adapter(d_model)
        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        ## x shape [HW+1, BT, D]
        n, bt, d = x.shape

        ## temporal adaptation
        xt = rearrange(x, "n (b t) d -> t (b n) d", t=self.num_frames)
        if self.num_tadapter == 2:
            xt = self.T_Adapter(self.attention(self.T_Adapter_in(self.ln_1(xt))))
        else:
            xt = self.T_Adapter(self.attention(self.ln_1(xt)))
        xt = rearrange(xt, "t (b n) d -> n (b t) d", n=n)
        x = x + self.drop_path(xt)
        ## spatial adaptation
        x = x + self.S_Adapter(self.attention(self.ln_1(x)))
        ## joint adaptation
        xn = self.ln_2(x)
        x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        num_frames,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor = None,
        num_tadapter=1,
        scale=1.0,
        drop_path=0.1,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlock(
                    width, heads, attn_mask, scale, num_tadapter, num_frames, dpr[i]
                )
                for i in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class I3DHead(nn.Module):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        spatial_type="avg",
        dropout_ratio=0.5,
        init_std=0.01,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == "avg":
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

        self.init_weights()

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for p in self.fc_cls.parameters():
            nn.init.normal_(p, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score


class ViT_CLIP(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(
        self,
        img_size=224,
        all_frames=8,
        patch_size=16,
        embed_dim=768,
        in_chans=3,
        depth=12,
        num_heads=12,
        num_classes=7,
        adapter_scale=0.5,
        drop_path_rate=0.2,
        pretrained=True,
    ):
        super().__init__()
        self.input_resolution = img_size
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = embed_dim**-0.5
        self.layers = depth
        self.class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((img_size // patch_size) ** 2 + 1, embed_dim)
        )
        self.ln_pre = LayerNorm(embed_dim)

        self.num_frames = all_frames
        self.temporal_embedding = nn.Parameter(torch.zeros(1, all_frames, embed_dim))

        self.transformer = Transformer(
            all_frames,
            embed_dim,
            depth,
            num_heads,
            num_tadapter=1,
            scale=adapter_scale,
            drop_path=drop_path_rate,
        )

        self.ln_post = LayerNorm(embed_dim)
        self.head = I3DHead(num_classes=num_classes, in_channels=embed_dim)

        self.init_weights(pretrained=self.pretrained)

    def get_num_layers(self):
        return self.layers

    def init_weights(self, pretrained):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.apply(_init_weights)
            ## Load OpenAI CLIP pretrained weights
            if self.layers == 12:
                clip_model, preprocess = clip.load("ViT-B/16", device="cpu")
            else:
                clip_model, preprocess = clip.load("ViT-L/14", device="cpu")
            pretrain_dict = clip_model.visual.state_dict()
            del clip_model
            del pretrain_dict["proj"]
            msg = self.load_state_dict(pretrain_dict, strict=False)
            print("Missing keys: {}".format(msg.missing_keys))
            print("Unexpected keys: {}".format(msg.unexpected_keys))
            torch.cuda.empty_cache()
        else:
            self.apply(_init_weights)

        ## initialize S_Adapter
        for n, m in self.transformer.named_modules():
            if "S_Adapter" in n:
                for n2, m2 in m.named_modules():
                    if "D_fc2" in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize T_Adapter
        for n, m in self.transformer.named_modules():
            if "T_Adapter" in n:
                for n2, m2 in m.named_modules():
                    if "D_fc2" in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize MLP_Adapter
        for n, m in self.transformer.named_modules():
            if "MLP_Adapter" in n:
                for n2, m2 in m.named_modules():
                    if "D_fc2" in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        for n, m in self.head.named_modules():
            if "fc_cls" in n:
                if isinstance(m, nn.Linear):
                    for p in m.parameters():
                        nn.init.normal_(p, std=0.01)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "class_embedding",
            "positional_embedding",
            "ln_1",
            "ln_2",
            "ln_pre",
            "ln_post",
        }

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table", "temporal_position_bias_table"}

    def forward(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.positional_embedding.to(x.dtype)  # BT, K+1, C

        n = x.shape[1]
        x = rearrange(x, "(b t) n d -> (b n) t d", t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, "(b n) t d -> (b t) n d", n=n)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x = x[:, 0]
        x = rearrange(x, "(b t) d -> b d t", b=B, t=T)

        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        x = self.head(x)

        return x


@register_model
def vit_base_patch16_224_aim(full_finetune=False, **kwargs):
    model = ViT_CLIP(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs,
    )

    model.default_cfg = _cfg()
    if full_finetune:
        print("Tuned all the parameters of the model.")
    else:
        tuned_list = []
        untuned_list = []
        for name, param in model.named_parameters():
            if (
                "temporal_embedding" not in name
                and "ln_post" not in name
                and "head" not in name
                and "Adapter" not in name
            ):
                param.requires_grad = False
                untuned_list.append(name)
            else:
                tuned_list.append(name)

        print("Tuned parameters of total:", ", ".join(tuned_list))
        print("Untuned parameters of total:", ", ".join(untuned_list))

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())
    print(
        "Number of total parameters: {}, tunable parameters: {}".format(
            num_total_param, num_param
        )
    )

    return model


@register_model
def vit_large_patch14_224_aim(full_finetune=False, **kwargs):
    model = ViT_CLIP(
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        **kwargs,
    )

    model.default_cfg = _cfg()
    if full_finetune:
        print("Tuned all the parameters of the model.")
    else:
        tuned_list = []
        untuned_list = []
        for name, param in model.named_parameters():
            if (
                "temporal_embedding" not in name
                and "ln_post" not in name
                and "head" not in name
                and "Adapter" not in name
            ):
                param.requires_grad = False
                untuned_list.append(name)
            else:
                tuned_list.append(name)

        print("Tuned parameters of total:", ", ".join(tuned_list))
        print("Untuned parameters of total:", ", ".join(untuned_list))

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())
    print(
        "Number of total parameters: {}, tunable parameters: {}".format(
            num_total_param, num_param
        )
    )

    return model


if __name__ == "__main__":
    from collections import OrderedDict
    import utils
    from datasets.args import get_args_finetuning
    from datasets.transforms.optim_factory import LayerDecayValueAssigner

    args = get_args_finetuning()[0]
    model = vit_large_patch14_224_aim(args.full_finetune)
    # model = vit_base_patch16_224_aim(args.full_finetune)
    x = torch.rand((2, 3, 8, 224, 224))
    y = model(x)
    print(y.shape)

    # import json

    # def get_parameter_groups(
    #     model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None
    # ):
    #     parameter_group_names = {}
    #     parameter_group_vars = {}
    #     for name, param in model.named_parameters():
    #         if not param.requires_grad:
    #             continue  # frozen weights

    #         if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
    #             group_name = "no_decay"
    #             this_weight_decay = 0.0
    #         else:
    #             group_name = "decay"
    #             this_weight_decay = weight_decay
    #         if get_num_layer is not None:
    #             layer_id = get_num_layer(name)
    #             group_name = "layer_%d_%s" % (layer_id, group_name)
    #         else:
    #             layer_id = None

    #         if group_name not in parameter_group_names:
    #             if get_layer_scale is not None:
    #                 scale = get_layer_scale(layer_id)
    #             else:
    #                 scale = 1.0

    #             parameter_group_names[group_name] = {
    #                 "weight_decay": this_weight_decay,
    #                 "params": [],
    #                 "lr_scale": scale,
    #             }
    #             parameter_group_vars[group_name] = {
    #                 "weight_decay": this_weight_decay,
    #                 "params": [],
    #                 "lr_scale": scale,
    #             }

    #         parameter_group_vars[group_name]["params"].append(param)
    #         parameter_group_names[group_name]["params"].append(name)
    #     print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    #     return list(parameter_group_vars.values())
    
    # skip_weight_decay_list = model.no_weight_decay()
    # num_layers = model.get_num_layers()
    # assigner = LayerDecayValueAssigner(
    #     list(0.75 ** (num_layers + 1 - i) for i in range(num_layers + 2))
    # )
    # assigner = LayerDecayValueAssigner([0.1] * (num_layers + 1) + [1.0])
    # optimizer_params = get_parameter_groups(
    #     model,
    #     args.weight_decay,
    #     skip_weight_decay_list,
    #     assigner.get_layer_id if assigner is not None else None,
    #     assigner.get_scale if assigner is not None else None,
    # )
