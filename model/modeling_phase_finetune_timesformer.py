import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import numpy as np
import utils
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange
from collections import OrderedDict
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


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


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


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        drop_path=0.2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        qkv_divided=False,
        qkv_bias=False,
    ):
        super().__init__()
        self.qkv_divided = qkv_divided
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_divided=qkv_divided,
        )

        ## Temporal Attention Parameters
        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_divided=qkv_divided,
        )
        self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * 4)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer
        )

    def forward(self, x, B, T, K):
        # 如果alpha以及beta初始化为0，则xs、xt初始化为0, 在训练过程中降低了学习难度；
        # 仿照其余模型可以使用alpha.sigmoid()以及beta.sigmoid()；
        B, M, C = x.shape
        assert T * K + 1 == M

        # Temporal_Self_Attention
        xt = x[:, 1:, :]
        xt = rearrange(xt, "b (k t) c -> (b k) t c", t=T)
        res_temporal = self.drop_path(
            self.temporal_attn.forward(self.temporal_norm1(xt))
        )

        res_temporal = rearrange(
            res_temporal, "(b k) t c -> b (k t) c", b=B
        )  # 通过FC时需要将时空tokens合并，再通过残差连接连接输入特征
        xt = self.temporal_fc(res_temporal) + x[:, 1:, :]

        # Spatial_Self_Attention
        init_cls_token = x[:, 0, :].unsqueeze(1)  # B, 1, C
        cls_token = init_cls_token.repeat(1, T, 1)  # B, T, C
        cls_token = rearrange(cls_token, "b t c -> (b t) c", b=B, t=T).unsqueeze(1)
        xs = xt
        xs = rearrange(xs, "b (k t) c -> (b t) k c", t=T)

        xs = torch.cat((cls_token, xs), 1)  # BT, K+1, C
        res_spatial = self.drop_path(self.attn.forward(self.norm1(xs)))

        ### Taking care of CLS token
        cls_token = res_spatial[:, 0, :]  # BT, C 表示了在每帧单独学习的class token
        cls_token = rearrange(cls_token, "(b t) c -> b t c", b=B, t=T)
        cls_token = torch.mean(cls_token, 1, True)  # 通过在全局帧上平均来建立时序关联（适用于视频分类任务）
        res_spatial = res_spatial[
            :,
            1:,
        ]  # BT, xK, C
        res_spatial = rearrange(res_spatial, "(b t) k c -> b (k t) c", b=B)
        res = res_spatial
        x = xt

        ## Mlp
        x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 通过MLP学习时序对应的cls_token?

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
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2)
        x = rearrange(x, "(b t) c k -> b t k c", b=B)

        return x


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
        drop_path_rate=0.1,
        fc_drop_rate=0.5,
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
        self.qkv_divided = qkv_divided
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
        self.time_embed = nn.Parameter(torch.zeros(1, all_frames, embed_dim))

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
                )
                for i in range(self.depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        if self.pre_norm:
            self.norm_pre = nn.LayerNorm(embed_dim)

        # Classifier head
        self.fc_dropout = (
            nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        )
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        i = 0
        for m in self.blocks.modules():
            m_str = str(m)
            if "Block" in m_str:
                if i > 0:
                    nn.init.constant_(m.temporal_fc.weight, 0)
                    nn.init.constant_(m.temporal_fc.bias, 0)
                i += 1

    def get_num_layers(self):
        return len(self.blocks)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "time_embed"}

    def forward_features(self, x):
        B, C, T, H, W = x.shape
        # B, C, T, H, W
        x = self.patch_embed(x)
        # B, T, K, C
        B, T, K, C = x.size()
        W = int(math.sqrt(K))
        x = rearrange(x, "b t k c -> (b t) k c")

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

        # 添加Temporal Position Embedding
        cls_tokens = x[:B, 0, :].unsqueeze(1)
        x = x[:, 1:]  # 过滤掉cls_tokens
        x = rearrange(x, "(b t) k c -> (b k) t c", b=B)
        x = x + self.time_embed  # BK, T, C  ---> 2*196, 8, 768

        # 添加Cls token
        x = rearrange(x, "(b k) t c -> b (k t) c", b=B)  # Spatial-Temporal tokens
        x = torch.cat((cls_tokens, x), dim=1)  # 时空tokens对应的class token的添加；

        if self.pre_norm:
            x = self.norm_pre(x)

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, K)
        x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(self.fc_dropout(x))
        return x


@register_model
def vit_base_224_timm(pretrained=True, pretrain_path=None, **kwargs):
    model = VisionTransformer(
        img_size=224,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs,
    )
    model.default_cfg = _cfg()

    if isinstance(pretrain_path, str):
        print("Load ckpt from %s" % pretrain_path)
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        state_dict = model.state_dict()

        add_list = []
        for k in state_dict.keys():
            if "blocks" in k and "temporal_attn" in k:
                k_init = k.replace("temporal_attn", "attn")
                if k_init in checkpoint:
                    checkpoint[k] = checkpoint[k_init]
                    add_list.append(k)
            if "blocks" in k and "temporal_norm1" in k:
                k_init = k.replace("temporal_norm1", "norm1")
                if k_init in checkpoint:
                    checkpoint[k] = checkpoint[k_init]
                    add_list.append(k)

        print("Adding keys from pretrained checkpoint:", ", ".join(add_list))

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

    print("Tuned all the parameters of the model.")

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())
    print(
        "Number of total parameters: {}, tunable parameters: {}".format(
            num_total_param, num_param
        )
    )

    return model


@register_model
def vit_large_224_timm(pretrained=True, pretrain_path=None, **kwargs):
    model = VisionTransformer(
        img_size=224,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        **kwargs,
    )
    model.default_cfg = _cfg()

    if isinstance(pretrain_path, str):
        print("Load ckpt from %s" % pretrain_path)
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        state_dict = model.state_dict() 
        
        add_list = []
        for k in state_dict.keys():
            if "blocks" in k and "temporal_attn" in k:
                k_init = k.replace("temporal_attn", "attn")
                if k_init in checkpoint:
                    checkpoint[k] = checkpoint[k_init]
                    add_list.append(k)
            if "blocks" in k and "temporal_norm1" in k:
                k_init = k.replace("temporal_norm1", "norm1")
                if k_init in checkpoint:
                    checkpoint[k] = checkpoint[k_init]
                    add_list.append(k)

        print("Adding keys from pretrained checkpoint:", ", ".join(add_list))

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

    print("Tuned all the parameters of the model.")

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
    model = vit_base_224_timm(
        pretrain_path="/Users/yangshu/Documents/PETL4SurgVideo/pretrain_params/vit_base_patch16_224_wit400m.pth",
    )
    # model = vit_large_224_timm(
    #     pretrain_path="/Users/yangshu/Documents/PETL4SurgVideo/pretrain_params/vit_large_patch14_224_wit400m.pth",
    # )
    x = torch.rand((2, 3, 8, 224, 224))
    y = model(x)
    print(y.shape)
