import torch
import torch.nn as nn
from functools import partial, reduce
import torch.nn.functional as F
import numpy as np
import utils
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange
from collections import OrderedDict
from typing import Tuple, Union
from torchvision import transforms
from operator import mul
import math
import matplotlib.pyplot as plt


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


class TemporalPositionalEmbedding(nn.Module):
    def __init__(self, channels):
        super(TemporalPositionalEmbedding, self).__init__()
        self.channels = channels
        self.inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2).float() / channels)
        )

    def forward(self, tensor):
        if len(tensor.shape) == 3:
            batch_size, N, channels = tensor.shape
            resize_shape = int(math.sqrt(N - 1))
            num_frame_per_width = 4
            width = int(16 / num_frame_per_width)  # resized frame size
            temp = torch.zeros(batch_size, 16, width, width).to(tensor.device)
            for i in range(num_frame_per_width**2):
                temp[:, i, :, :] = i + 1
            temp = temp.reshape(
                batch_size, num_frame_per_width, num_frame_per_width, width, width
            )
            temp = temp.permute(0, 1, 3, 2, 4).reshape(batch_size, 16, 16)
            resize = transforms.Resize((resize_shape, resize_shape))
            temp = resize(temp)
            emb = temp.view(batch_size, -1)[0]
            emb = torch.cat([torch.tensor([0.0]).view(1).to(tensor.device), emb])
            emb = torch.einsum(
                "i,j->ij", emb, self.inv_freq.to(tensor.device)
            )  # [N, D]
            emb = torch.stack((emb.sin(), emb.cos()), dim=-1)
            emb = torch.flatten(emb, -2, -1)
            return emb.repeat(batch_size, 1, 1)
        else:
            batch_size, Tt, N, channels = tensor.shape
            resize_shape = int(math.sqrt(N - 1))
            # emb = torch.zeros(batch_size, N, channels)
            num_frame_per_width = 4
            width = int(16 / num_frame_per_width)  # resized frame size
            temp = torch.zeros(batch_size, 16 * Tt, width, width).to(tensor.device)
            for i in range((num_frame_per_width**2) * Tt):
                temp[:, i, :, :] = i + 1
            temp = temp.reshape(
                batch_size, Tt, num_frame_per_width, num_frame_per_width, width, width
            )
            temp = temp.permute(0, 1, 2, 4, 3, 5).reshape(batch_size, Tt, 16, 16)
            resize = transforms.Resize((resize_shape, resize_shape))
            temp = resize(temp)  # [B, Tt, root(N), root(N)]
            emb = temp.view(batch_size, Tt, -1)[0]  # [B, Tt, N]
            emb = torch.cat(
                [torch.tensor([[0.0]] * Tt).to(tensor.device), emb], dim=1
            )  # [Tt, N]
            emb = emb.view(-1)  # [TtxN]
            emb = torch.einsum(
                "i,j->ij", emb, self.inv_freq.to(tensor.device)
            )  # [N, D/2]
            emb = torch.stack((emb.sin(), emb.cos()), dim=-1)
            emb = torch.flatten(emb, -2, -1).reshape(Tt, N, -1)
            return emb.repeat(batch_size, 1, 1, 1)


class Adapter(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        bottleneck: int = 128,
        dropout=0.0,
        init_option="lora",
        adapter_scalar="1.0",
        adapter_layernorm_option="out",
    ):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        # _before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.GELU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=False, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == "in":
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


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
        num_frames,
        attn_mask=None,
        adapter=None,
        qkv_divided=False,
        attn_divided=False,
        qkv_bias=False,
    ):
        super().__init__()
        # Config
        self.qkv_divided = qkv_divided
        self.attn_divided = attn_divided

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_divided=qkv_divided,
        )
        self.norm1 = LayerNorm(dim)

        mlp_hidden_dim = int(dim * 4)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=QuickGELU,
        )
        self.norm2 = LayerNorm(dim)
        self.attn_mask = attn_mask
        self.num_frames = num_frames

        self.adapter = adapter
        if self.adapter == "w-adapter":
            self.t_adapter_attn_b = Adapter(
                d_model=dim,
                bottleneck=int(0.25 * dim),
                dropout=0.1,
                adapter_layernorm_option=None,
            )
            self.s_adapter_attn = Adapter(
                d_model=dim,
                bottleneck=int(0.25 * dim),
                dropout=0.1,
                adapter_layernorm_option=None,
            )
            self.t_adapter_attn = Adapter(
                d_model=dim,
                bottleneck=int(0.25 * dim),
                dropout=0.1,
                adapter_layernorm_option=None,
            )
            self.s_adapter_mlp = Adapter(
                d_model=dim,
                bottleneck=int(0.25 * dim),
                dropout=0.1,
                adapter_layernorm_option=None,
            )
            self.t_adapter_mlp = Adapter(
                d_model=dim,
                bottleneck=int(0.25 * dim),
                dropout=0.1,
                adapter_layernorm_option=None,
            )

    def forward(self, x):
        T = self.num_frames
        Ts = 8
        Tt = int(T - Ts)
        B, N, D = int(x.shape[0] / (T)), x.shape[1], x.shape[2]
        x_t_residual = x.reshape(B, T, N, D)[:, 0:Tt, :, :].reshape(-1, N, D)
        x_s_residual = x.reshape(B, T, N, D)[:, Tt:, :, :].reshape(-1, N, D)
        x = self.norm1(x)
        x = x.reshape(B, T, N, D)
        x_s = x[:, Tt:, :, :].reshape(-1, N, D)  # [B*Ts, N+1, D]
        x_t = x[:, 0:Tt, :, :].reshape(-1, N, D)  # [B*Tt, N+1, D]
        x_t = self.t_adapter_attn_b(x_t, add_residual=False)
        x_t = self.attn(x_t)
        x_t = self.t_adapter_attn(x_t)
        x_t = x_t + x_t_residual
        x_t_residual2 = x_t
        x_t = self.norm2(x_t)
        x_t = self.mlp(x_t)
        x_t = self.t_adapter_mlp(x_t) + x_t_residual2

        x_s_adapt = self.s_adapter_attn(x_s)
        x_s = self.attn(x_s) + x_s_adapt + x_s_residual
        x_s_residual2 = x_s
        x_s = self.norm2(x_s)
        x_s_mlp = self.s_adapter_mlp(x_s)
        x_s = self.mlp(x_s) + x_s_mlp + x_s_residual2
        x = torch.cat(
            [x_t.reshape(B, -1, N, D), x_s.reshape(B, -1, N, D)], dim=1
        ).reshape(-1, N, D)
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
        all_frames=31,
        patch_size=16,
        embed_dim=768,
        in_chans=3,
        depth=12,
        num_heads=12,
        num_classes=7,
        adapter="w-adapter",
        qkv_divided=False,
        attn_divided=False,
        pre_norm=False,
        surgery=False,
        qkv_bias=False,
    ):
        super().__init__()
        self.input_resolution = img_size
        self.depth = depth
        self.num_classes = num_classes
        self.pre_norm = pre_norm
        self.surgery = surgery
        self.qkv_bias = qkv_bias
        self.adapter = adapter
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
        scale = embed_dim**-0.5
        self.cls_token = nn.Parameter(scale * torch.randn(1, 1, embed_dim))
        if self.surgery:
            self.pos_embed = nn.Parameter(
                scale * torch.randn(1, num_patches // all_frames, embed_dim)
            )
        else:
            self.pos_embed = nn.Parameter(
                scale * torch.randn(1, num_patches // all_frames + 1, embed_dim)
            )
        if self.pre_norm:
            self.norm_pre = LayerNorm(embed_dim)

        ## Attention Blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    num_frames=9,
                    adapter=self.adapter,
                    qkv_divided=qkv_divided,
                    attn_divided=attn_divided,
                    qkv_bias=self.qkv_bias,
                )
                for i in range(self.depth)
            ]
        )
        self.norm = LayerNorm(embed_dim)

        # Classifier head
        self.head = I3DHead(num_classes=num_classes, in_channels=embed_dim)

        self.temporal_positional_embedding = TemporalPositionalEmbedding(embed_dim)
        self.spatial_positional_embedding = nn.Parameter(
            scale * torch.randn(num_patches // all_frames + 1, embed_dim)
        )
        self.num_frames = 9
        self.init_weights()

    def get_num_layers(self):
        return len(self.blocks)

    def init_weights(self):
        for n, m in self.head.named_modules():
            if "fc_cls" in n:
                if isinstance(m, nn.Linear):
                    for p in m.parameters():
                        nn.init.normal_(p, std=0.01)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
            "pos_embed",
            "norm1",
            "temporal_norm1",
            "norm2",
            "norm",
        }

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table", "temporal_position_bias_table"}

    def forward_features(self, x):
        t = [i for i in range(31)]
        t_t = t[::-2][::-1]
        t_s = t[::-4][::-1]
        B, C, T, H, W = x.shape
        x_T = x[:, :, t_t, :, :]
        x_S = x[:, :, t_s, :, :]
        x_T = rearrange(x_T, "b c (th tw) h w -> b c (th h) (tw w)", th=4)
        x_T = F.interpolate(x_T, size=(H, W), mode="bilinear", align_corners=False)
        # x_np = x_T.numpy()
        # plt.imshow(x_np[0, :, :, :].transpose(1, 2, 0))
        # plt.axis("off")
        # plt.show()
        x_T = x_T.unsqueeze(2)
        x = torch.cat((x_T, x_S), dim=2)

        x, T, W = self.patch_embed(x)  # BT, HW, C
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

        if self.adapter == "w-adapter":
            GB, N, D = x.shape
            Ts = len(t_s)  # 8
            Tt = len(t_t) // 16
            x = x.reshape(int(GB / self.num_frames), self.num_frames, N, D)
            x[:, 0:Tt, :, :] = (
                x[:, 0:Tt, :, :]
                + self.temporal_positional_embedding(x[:, 0:Tt, :, :])
                + self.spatial_positional_embedding.expand(
                    int(GB / self.num_frames), Tt, -1, -1
                )
            )
            if self.surgery:
                x[:, Tt:, 1:, :] = (
                    x[:, Tt:, 1:, :].reshape(-1, N, D) + self.pos_embed.to(x.dtype)
                ).reshape(int(GB / self.num_frames), 8, N, D)
            else:
                x[:, Tt:, :, :] = (
                    x[:, Tt:, :, :].reshape(-1, N, D) + self.pos_embed.to(x.dtype)
                ).reshape(int(GB / self.num_frames), 8, N, D)
            x = x.reshape(-1, N, D)

        if self.pre_norm:
            x = self.norm_pre(x)

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x)

        # 在这里没有follow dual-path的设计，为了公平使用了一致的I3D Head
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
def vit_base_224_dual_path(pretrained=True, pretrain_path=None, **kwargs):
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

        if "model_state" in checkpoint.keys():
            checkpoint = checkpoint["model_state"]
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                # strip `model.` prefix
                name = k[6:] if k.startswith("model") else k
                new_state_dict[name] = v
            checkpoint = new_state_dict

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
        else:
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

    tuned_list = []
    untuned_list = []
    for name, param in model.named_parameters():
        if name in checkpoint and name != "cls_token" and name != "pos_embed" and name != "spatial_positional_embedding":
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
def vit_large_224_dual_path(pretrained=True, pretrain_path=None, **kwargs):
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

        if "model_state" in checkpoint.keys():
            checkpoint = checkpoint["model_state"]
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                # strip `model.` prefix
                name = k[6:] if k.startswith("model") else k
                new_state_dict[name] = v
            checkpoint = new_state_dict

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

        else:
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

    tuned_list = []
    untuned_list = []
    for name, param in model.named_parameters():
        if name in checkpoint and name != "cls_token" and name != "pos_embed" and name != "spatial_positional_embedding":
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


import os
from datasets.phase.AutoLaparo_phase import PhaseDataset_AutoLaparo


def build_dataset(is_train, test_mode, fps, args):
    if args.data_set == "AutoLaparo":
        mode = None
        anno_path = None
        if is_train is True:
            mode = "train"
            anno_path = os.path.join(
                args.data_path, "labels_pkl", mode, fps + "train.pickle"
            )
        elif test_mode is True:
            mode = "test"
            anno_path = os.path.join(
                args.data_path, "labels_pkl", mode, fps + "test.pickle"
            )
        else:
            mode = "val"
            anno_path = os.path.join(
                args.data_path, "labels_pkl", mode, fps + "val.pickle"
            )

        dataset = PhaseDataset_AutoLaparo(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            data_strategy="online",
            output_mode="key_frame",
            cut_black=False,
            clip_len=31,
            frame_sample_rate=4,  # 0表示指数级间隔，-1表示随机间隔设置, -2表示递增间隔
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 7
    assert nb_classes == args.nb_classes
    print(
        "%s %s - %s : Number of the class = %d"
        % ("AutoLaparo", mode, fps, args.nb_classes)
    )

    return dataset, nb_classes


if __name__ == "__main__":
    from collections import OrderedDict
    import utils
    from datasets.args import get_args_finetuning
    from datasets.transforms.optim_factory import LayerDecayValueAssigner

    args = get_args_finetuning()[0]
    # model = vit_base_224_dual_path(
    #     pretrain_path="/Users/yangshu/Documents/SurgVideoMAE/pretrain_params/vit_base_patch16_224_augreg_in21k.bin",
    #     # pretrain_path="/Users/yangshu/Documents/SurgVideoMAE/pretrain_params/vit_base_patch16_clip_224_laion2b_ft_in12k_in1k.bin",
    #     # pretrain_path="pretrain_params/TimeSformer_divST_8x32_224_HowTo100M.pyth",
    # )
    model = vit_large_224_dual_path(
        pretrain_path="/Users/yangshu/Documents/SurgVideoMAE/pretrain_params/vit_large_patch16_224_augreg_in21k.bin",
        # pretrain_path="/Users/yangshu/Documents/SurgVideoMAE/pretrain_params/vit_base_patch16_clip_224_laion2b_ft_in12k_in1k.bin",
        # pretrain_path="pretrain_params/TimeSformer_divST_8x32_224_HowTo100M.pyth",
    )
    # dataset, nb_class = build_dataset(
    #     is_train=True, test_mode=False, fps="1fps", args=args
    # )
    # data_loader_train = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    # for k in data_loader_train:
    #     images, gt, names, flags = k
    #     # print(flags)
    #     print(gt)
    #     output = model(images)
    #     print(output)
    #     break

    x = torch.rand((2, 3, 31, 224, 224))
    y = model(x)
    print(y.shape)
