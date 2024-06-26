import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange
from collections import OrderedDict
from typing import Tuple
import functools
import utils
import math


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


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qkv_divided=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv_divided = qkv_divided
        if not self.qkv_divided:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=True)
            self.k = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, dim, bias=True)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        if not self.qkv_divided:
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q = (
                self.q(x)
                .reshape(B, N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )
            k = (
                self.k(x)
                .reshape(B, N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )
            v = (
                self.v(x)
                .reshape(B, N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        return x


class Adapter(nn.Module):
    def __init__(self, in_channels, adapter_channels, kernel_size):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels,
            adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        nn.init.constant_(self.conv.weight, 0.0)
        nn.init.constant_(self.conv.bias, 0.0)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x, T):
        BT, L, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W
        x_id = x
        x = x[:, 1:, :]
        x = self.fc1(x)
        x = x.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous()

        x = self.conv(x)

        x = x.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca)
        x = self.fc2(x)
        x_id[:, 1:, :] += x
        return x_id


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        drop_path=0.0,
        adapter_width=384,
        adapter_kernel_size=(3, 1, 1),
        adapter_pre_attn=True,
        adapter_pre_mlp=True,
        qkv_divided=False,
        qkv_bias=False,
    ):
        super().__init__()
        self.qkv_divided = qkv_divided
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_divided=qkv_divided,
        )
        self.norm1 = LayerNorm(dim)

        ## drop path
        mlp_hidden_dim = int(dim * 4)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=QuickGELU,
        )

        self.norm2 = LayerNorm(dim)

        adapter_class = functools.partial(
            Adapter,
            in_channels=dim,
            adapter_channels=adapter_width,
            kernel_size=adapter_kernel_size,
        )
        self.adapter_pre_attn = adapter_class() if adapter_pre_attn else None
        self.adapter_pre_mlp = adapter_class() if adapter_pre_mlp else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, num_frames):
        ## x shape [BT, N+1, D]
        bt, n, d = x.shape

        if self.adapter_pre_attn is not None:
            x = self.adapter_pre_attn(x, num_frames)

        ## spatial adaptation
        x = x + self.attn(self.norm1(x))

        if self.adapter_pre_mlp is not None:
            x = self.adapter_pre_mlp(x, num_frames)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        num_frames=8,
        surgery=False,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (num_frames)
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if surgery:
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=True,
            )
        else:
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W


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


class VisionTransformer(nn.Module):
    """Vision Transformer"""

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
        adapter_width=384,
        adapter_layers=12,
        adapter_kernel_size=(3, 1, 1),
        adapter_pre_attn=True,
        adapter_pre_mlp=True,
        qkv_divided=False,
        pre_norm=False,
        surgery=False,
        qkv_bias=False,
    ):
        super().__init__()
        self.depth = depth
        self.num_classes = num_classes
        self.pre_norm = pre_norm
        self.surgery = surgery
        self.qkv_bias = qkv_bias
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=all_frames,
            surgery=self.surgery,
        )
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.surgery:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches // all_frames, embed_dim)
            )
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches // all_frames + 1, embed_dim)
            )

        ## Attention Blocks
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, self.depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    drop_path=dpr[i],
                    qkv_divided=qkv_divided,
                    qkv_bias=self.qkv_bias,
                    adapter_width=adapter_width,
                    adapter_kernel_size=adapter_kernel_size,
                    adapter_pre_attn=adapter_pre_attn and i >= depth - adapter_layers,
                    adapter_pre_mlp=adapter_pre_mlp and i >= depth - adapter_layers,
                )
                for i in range(self.depth)
            ]
        )
        self.norm = LayerNorm(embed_dim)

        if self.pre_norm:
            self.norm_pre = LayerNorm(embed_dim)

        # Classifier head
        self.head = I3DHead(num_classes=num_classes, in_channels=embed_dim)

        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)
        for n, m in self.blocks.named_modules():
            if "adapter" in n:
                for n2, m2 in m.named_modules():
                    if "fc1" in n2 or "fc2" in n2 or "conv" in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.bias, 0.0)
                            nn.init.constant_(m2.bias, 0.0)
                        elif isinstance(m2, nn.Conv3d):
                            nn.init.constant_(m2.weight, 0.0)
                            nn.init.constant_(m2.bias, 0.0)

        for n, m in self.head.named_modules():
            if "fc_cls" in n:
                if isinstance(m, nn.Linear):
                    for p in m.parameters():
                        nn.init.normal_(p, std=0.01)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
            "pos_embed",
            "norm1",
            "norm2",
            "norm",
        }

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table", "temporal_position_bias_table"}

    def forward_features(self, x):
        B, C, T, H, W = x.shape
        x, T, W = self.patch_embed(x)  # BT, HW, C

        if self.surgery:
            x = x + self.pos_embed.to(x.dtype)  # BT, HW, C  ---> 2*8, 196, 768
            x = torch.cat(
                [
                    self.cls_token.to(x.dtype)
                    + torch.zeros(
                        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                    ),
                    x,
                ],
                dim=1,
            )
        else:
            x = torch.cat(
                [
                    self.cls_token.to(x.dtype)
                    + torch.zeros(
                        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                    ),
                    x,
                ],
                dim=1,
            )
            x = x + self.pos_embed.to(x.dtype)  # BT, HW+1, C  ---> 2*8, 196+1, 768

        if self.pre_norm:
            x = self.norm_pre(x)

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, T)

        x = self.norm(x)
        x = x[:, 0]
        x = rearrange(x, "(b t) d -> b d t", b=B, t=T)

        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def vit_base_224_stadapter24x384(pretrained=True, pretrain_path=None, **kwargs):
    model = VisionTransformer(
        img_size=224,
        embed_dim=768,
        depth=12,
        num_heads=12,
        adapter_width=384,
        adapter_layers=12,
        adapter_kernel_size=(3, 1, 1),
        adapter_pre_attn=True,
        adapter_pre_mlp=True,
        **kwargs,
    )
    model.default_cfg = _cfg()

    if isinstance(pretrain_path, str):
        print("Load ckpt from %s" % pretrain_path)
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        state_dict = model.state_dict()

        remove_list = []
        for k in state_dict.keys():
            if (
                ("head" in k or "patch_embed" in k)
                and k in checkpoint
                and k in state_dict
                and checkpoint[k].shape != state_dict[k].shape
            ):
                remove_list.append(k)
                del checkpoint[k]
        print(f"Removing keys from pretrained checkpoint:", ", ".join(remove_list))
        utils.load_state_dict(model, checkpoint)

    tuned_list = []
    untuned_list = []
    for name, param in model.named_parameters():
        if (
            ("norm" not in name and "head" not in name and "adapter" not in name)
            or "norm1" in name
            or "norm2" in name
            or "norm_pre" in name
        ):
            param.requires_grad = False
            untuned_list.append(name)
        else:
            tuned_list.append(name)
    print("=" * 20)
    print("Tuned parameters of total:", ", ".join(tuned_list))
    print("=" * 20)
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
def vit_large_224_stadapter24x384(pretrained=True, pretrain_path=None, **kwargs):
    model = VisionTransformer(
        img_size=224,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        adapter_width=384,
        adapter_layers=24,
        adapter_kernel_size=(3, 1, 1),
        adapter_pre_attn=True,
        adapter_pre_mlp=True,
        **kwargs,
    )
    model.default_cfg = _cfg()

    if isinstance(pretrain_path, str):
        print("Load ckpt from %s" % pretrain_path)
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        state_dict = model.state_dict()

        remove_list = []
        for k in state_dict.keys():
            if (
                ("head" in k or "patch_embed" in k)
                and k in checkpoint
                and k in state_dict
                and checkpoint[k].shape != state_dict[k].shape
            ):
                remove_list.append(k)
                del checkpoint[k]
        print(f"Removing keys from pretrained checkpoint:", ", ".join(remove_list))
        utils.load_state_dict(model, checkpoint)

    tuned_list = []
    untuned_list = []
    for name, param in model.named_parameters():
        if (
            ("norm" not in name and "head" not in name and "adapter" not in name)
            or "norm1" in name
            or "norm2" in name
            or "norm_pre" in name
        ):
            param.requires_grad = False
            untuned_list.append(name)
        else:
            tuned_list.append(name)
    print("=" * 20)
    print("Tuned parameters of total:", ", ".join(tuned_list))
    print("=" * 20)
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
def vit_base_224_stadapter12x384(pretrained=True, pretrain_path=None, **kwargs):
    model = VisionTransformer(
        img_size=224,
        embed_dim=768,
        depth=12,
        num_heads=12,
        adapter_width=384,
        adapter_layers=12,
        adapter_kernel_size=(3, 1, 1),
        adapter_pre_attn=False,
        adapter_pre_mlp=True,
        **kwargs,
    )
    model.default_cfg = _cfg()

    if isinstance(pretrain_path, str):
        print("Load ckpt from %s" % pretrain_path)
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        state_dict = model.state_dict()

        remove_list = []
        for k in state_dict.keys():
            if (
                ("head" in k or "patch_embed" in k)
                and k in checkpoint
                and k in state_dict
                and checkpoint[k].shape != state_dict[k].shape
            ):
                remove_list.append(k)
                del checkpoint[k]
        print(f"Removing keys from pretrained checkpoint:", ", ".join(remove_list))
        utils.load_state_dict(model, checkpoint)

    tuned_list = []
    untuned_list = []
    for name, param in model.named_parameters():
        if (
            ("norm" not in name and "head" not in name and "adapter" not in name)
            or "norm1" in name
            or "norm2" in name
            or "norm_pre" in name
        ):
            param.requires_grad = False
            untuned_list.append(name)
        else:
            tuned_list.append(name)
    print("=" * 20)
    print("Tuned parameters of total:", ", ".join(tuned_list))
    print("=" * 20)
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
def vit_large_224_stadapter12x384(pretrained=True, pretrain_path=None, **kwargs):
    model = VisionTransformer(
        img_size=224,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        adapter_width=384,
        adapter_layers=24,
        adapter_kernel_size=(3, 1, 1),
        adapter_pre_attn=False,
        adapter_pre_mlp=True,
        **kwargs,
    )
    model.default_cfg = _cfg()

    if isinstance(pretrain_path, str):
        print("Load ckpt from %s" % pretrain_path)
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        state_dict = model.state_dict()

        remove_list = []
        for k in state_dict.keys():
            if (
                ("head" in k or "patch_embed" in k)
                and k in checkpoint
                and k in state_dict
                and checkpoint[k].shape != state_dict[k].shape
            ):
                remove_list.append(k)
                del checkpoint[k]
        print(f"Removing keys from pretrained checkpoint:", ", ".join(remove_list))
        utils.load_state_dict(model, checkpoint)

    tuned_list = []
    untuned_list = []
    for name, param in model.named_parameters():
        if (
            ("norm" not in name and "head" not in name and "adapter" not in name)
            or "norm1" in name
            or "norm2" in name
            or "norm_pre" in name
        ):
            param.requires_grad = False
            untuned_list.append(name)
        else:
            tuned_list.append(name)
    print("=" * 20)
    print("Tuned parameters of total:", ", ".join(tuned_list))
    print("=" * 20)
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
    model = vit_base_224_stadapter24x384(
        pretrain_path="/Users/yangshu/Documents/SurgPETL/pretrain_params/vit_base_patch16_224_laion2b.pth",
        pre_norm=True,
        qkv_bias=True,
    )
    x = torch.rand((2, 3, 8, 224, 224))
    y = model(x)
