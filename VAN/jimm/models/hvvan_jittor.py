import jittor as jt
from jittor import Module, nn
from jittor import init
import numpy as np
import math
from cc_attention.functions import HorizontalVerticalAttention

from .registry import register_model
from .layers import DropPath, to_2tuple, trunc_normal_
from ..data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
BatchNorm2d = nn.BatchNorm2d

class Dropout2d(Module):
    def __init__(self, p=0.5, is_train=False):
        '''
        Randomly zero out entire channels, from "Efficient Object Localization Using Convolutional Networks"
        input:
            x: [N,C,H,W] or [N,C,L]
        output:
            y: same shape as x
        '''
        assert p >= 0 and p <= 1, "dropout probability has to be between 0 and 1, but got {}".format(p)
        self.p = p
        self.is_train = is_train

    def execute(self, input):
        output = input
        shape = input.shape[:-2]
        if self.p > 0 and self.is_train:
            if self.p == 1:
                output = jt.zeros(input.shape)
            else:
                noise = jt.random(shape)
                noise = (noise > self.p).int()
                output = output * noise.broadcast(input.shape, dims=[-2,-1]) / (1.0 - self.p) # div keep prob
        return output


class RHVAModule(Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RHVAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels))
        self.hva = HorizontalVerticalAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(out_channels),
            Dropout2d(0.1),
            nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)  # TODO: why magic number 512?
            )

    def execute(self, x, recurrence=1):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.hva(output)
        output = self.convb(output)
        output = self.bottleneck(jt.concat([x, output], 1))
        return output


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            init.gauss_(m.weight, 0.0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                # m.bias.data.zero_()
                init.constant_(m.bias)

    def execute(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def execute(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def execute(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = jt.Var(
            layer_scale_init_value * jt.ones((dim))).start_grad()
        self.layer_scale_2 = jt.Var(
            layer_scale_init_value * jt.ones((dim))).start_grad()

        self.apply(self._init_weights)

    # def start_grad(x):
    #     return x._update(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            init.gauss_(m.weight, 0.0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                # m.bias.data.zero_()
                init.constant_(m.bias)

    def execute(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            init.gauss_(m.weight, 0.0, math.sqrt(2.0 / fan_out))

            if m.bias is not None:
                # m.bias.data.zero_()
                init.constant_(m.bias)

    def execute(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W


class VAN(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], num_stages=4, flag=False, recurrence=2):
        super().__init__()
        if flag == False:
            self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.head = RHVAModule(512, 128, num_classes)  # * edition here

        # * edition here
        self.dsn = nn.Sequential(
            nn.Conv2d(320, 128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(128),
            Dropout2d(0.1),
            nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        self.recurrence = recurrence

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            init.gauss_(m.weight, 0.0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                # m.bias.data.zero_()
                init.constant_(m.bias)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad_ = False

    # @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = RHVAModule(512, 128, num_classes) if num_classes > 0 else nn.Identity()

    def execute(self, x, label=None):
        B = x.shape[0]
        # print(x.shape)
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            # x = x.flatten(2).transpose(1, 2)
            x = jt.transpose(x.flatten(2),0,2,1)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            # x size [B, ]
            outs.append(x)
            
        # print([out.shape for out in outs])
        
        # print(outs[-2].shape, outs[-1].shape)
        # exit()
        x_dsn = self.dsn(outs[-2])
        # print('x:', x.shape)
        x = self.head(x, self.recurrence)
        
        output = [x, x_dsn]
        
        return output


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def execute(self, x):
        x = self.dwconv(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register_model
def van_tiny(num_classes=19, img_size = 512, pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4], 
        norm_layer=nn.LayerNorm, depths=[3, 3, 5, 2], num_classes=num_classes, img_size=img_size,
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def van_small(num_classes=19, img_size = 512, pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=nn.LayerNorm, depths=[2, 2, 4, 2], num_classes=num_classes, img_size=img_size,
        **kwargs)
    model.default_cfg = _cfg()

    return model

@register_model
def van_base(num_classes=19, img_size = 512, pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4], 
        norm_layer=nn.LayerNorm, depths=[3, 3, 12, 3], num_classes=num_classes, img_size=img_size,
        **kwargs)
    model.default_cfg = _cfg()

    return model

@register_model
def van_large(num_classes=19, img_size = 512, pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4], 
        norm_layer=nn.LayerNorm, depths=[3, 5, 27, 3], num_classes=num_classes, img_size=img_size,
        **kwargs)
    model.default_cfg = _cfg()

    return model

# if __name__ == '__main__':
    # model = van_tiny()
    # print(model)
