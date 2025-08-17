import torch
import math
import torch.nn as nn
from einops import rearrange
from timm.layers import trunc_normal_, DropPath
from torch.nn import functional as F

act_layer = nn.ReLU(inplace=True)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True):
        super(BasicConv2d, self).__init__()
        self.relu = relu
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.relu:
            self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x
class MulBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.conv357 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, padding=0)


    def forward(self, x):
        x_residual = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x_357 = torch.cat((x3, x5, x7), dim=1)
        x_prime = self.conv357(x_357)
        xx = x_residual + x_prime
        return xx

class MCFE(nn.Module):
    def __init__(self, inplanes, outplanes, block_index, layer_num=2, bn_d=0.1):
        super(MCFE, self).__init__()
        self.layer_num = layer_num
        self.block_index = block_index

        self.ms_conv = MulBlock(inplanes, outplanes)
        dim_list = [outplanes]
        for i in range(self.layer_num):
            conv_name = 'downblock'+str(self.block_index)+'_conv%s'%i
            relu_name = 'downblock'+str(self.block_index)+'_relu%s'%i

            in_dim = sum(dim_list)
            conv = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, bias=False))

            setattr(self, conv_name, conv)
            setattr(self, relu_name, nn.LeakyReLU())

            dim_list.append(in_dim)
            in_dim = sum(dim_list)
            self.local_conv = nn.Sequential(nn.Conv2d(in_dim, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.LeakyReLU())

    def forward(self, x):
        x = self.ms_conv(x)

        global_out = x

        # local branch
        li = [x]
        out = x
        for i in range(self.layer_num):
            conv_name = 'downblock' + str(self.block_index) + '_conv%s' % i
            relu_name = 'downblock' + str(self.block_index) + '_relu%s' % i
            conv = getattr(self, conv_name)
            relu = getattr(self, relu_name)
            out = conv(out)
            out = relu(out)

            li.append(out)
            out = torch.cat(li, 1)

        local_out = self.local_conv(out)

        # residual learning
        out = global_out + local_out
        return out

class M2A2(nn.Module):
    def __init__(self, dim):
        super(M2A2, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.project_out = nn.Conv2d(dim, dim // 8, kernel_size=1)
        self.conv3 = nn.Conv2d(dim // 4, dim // 8, kernel_size=1)
        self.conv4 = nn.Conv2d(dim // 8, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv(x)
        x1 = self.conv0_1(x)
        x1 = self.conv0_2(x1)
        x2 = self.conv1_1(x)
        x2 = self.conv1_2(x2)
        x3 = self.conv2_1(x)
        x3 = self.conv2_2(x3)
        out1 = x1 + x2 + x3 + x

        out1 = self.project_out(out1)

        q1 = rearrange(out1, head=1)
        k1 = rearrange(out1, head=1)
        v1 = rearrange(out1, head=1)
        q2 = rearrange(out1, head=1)
        k2 = rearrange(out1, head=1)
        v2 = rearrange(out1, head=1)

        q1 = F.normalize(q1, dim=-1)
        q2 = F.normalize(q2, dim=-1)
        k1 = F.normalize(k1, dim=-1)
        k2 = F.normalize(k2, dim=-1)

        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out2 = attn1 @ v1

        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out3 = attn2 @ v2
        out2 = rearrange(out2, head=1, h=h, w=w)
        out3 = rearrange(out3, head=1, h=h, w=w)
        out4 = torch.cat((out2, out3), dim=1)

        out5 = self.conv3(out4) + out1
        out = self.conv4(out5)
        return out
class SA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.LeakyReLU()  #
        self.spatial_gating_unit = M2A2(dim)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.act(x)
        x = self.spatial_gating_unit(x)
        x = x + shortcut
        return x
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, if_BN=None):
        super(BasicBlock, self).__init__()
        self.if_BN = if_BN
        if self.if_BN:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.downsample = downsample

        self.conv = BasicConv2d(planes, planes, 3, padding=1)
        self.attn = SA(planes)
        self.dropout = nn.Dropout2d(p=0.4)
        self.stride = stride

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        out = self.conv(x)
        out = out + self.dropout(self.attn(out))
        out += x
        return out

class Cv(nn.Module):
    def __init__(self, dim=256):
        super(Cv, self).__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.conv = Cv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
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
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.conv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SelfA(nn.Module):
    def __init__(self, dim1, dim2, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 pool_ratio=16):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        self.dim2 = dim2
        self.num_heads = num_heads
        head_dim = dim1 // num_heads
        self.pool_ratio = pool_ratio
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.kv = nn.Linear(dim2, dim1 * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.pool_ratio >= 0:
            self.pool = nn.AvgPool2d(self.pool_ratio, self.pool_ratio)
            self.sr = nn.Conv2d(dim2, dim2, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim2)
        self.act = nn.GELU()
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
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y, H2, W2):
        B1, N1, C1 = x.shape
        B2, N2, C2 = y.shape
        q = self.q(x).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)

        if self.pool_ratio >= 0:
            # x_ = y.permute(0, 2, 1).reshape(B2, C2, H2, W2)
            # x_ = self.sr(self.pool(x_)).reshape(B2, C2, -1).permute(0, 2, 1)
            x_ = self.norm(y)
            x_ = self.act(y)
        else:
            x_ = y

        kv = self.kv(x_).reshape(B1, -1, 2, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1,4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
class Block(nn.Module):

    def __init__(self, dim1, dim2, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratio=16):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.norm2 = norm_layer(dim2)
        self.norm3 = norm_layer(dim1)
        self.attn = SelfA(dim1=dim1, dim2=dim2, num_heads=num_heads, pool_ratio=pool_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
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
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x, y, H2, W2, H1, W1):
        x = self.norm1(x)
        y = self.norm2(y)
        x = x + self.drop_path(self.attn(x, y, H2, W2))
        x = self.norm3(x)
        x = x + self.drop_path(self.mlp(x, H1, W1))
        return x
class AEUF(nn.Module):
    def __init__(self):
        super(AEUF,self).__init__()
        self.attn_c1 = Block(dim1=128, dim2=128, num_heads=1, mlp_ratio=2,
                                drop_path=0.1, pool_ratio=1)

    def forward(self,x,x_1):
        c, c1 = x, x_1
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c.shape
        _, _, h1, w1 = c1.shape
        c = c.flatten(2).transpose(1, 2)
        c1 = c1.flatten(2).transpose(1, 2)
        _c1 = self.attn_c1(c, c1, h, w, h1, w1)
        _c1 = _c1.permute(0,2,1).reshape(n, -1, h1, w1)
        return _c1
class ResNet_34(nn.Module):
    def __init__(self, nclasses, params, block=BasicBlock, layers=[3, 4, 6, 3], if_BN=True, zero_init_residual=False,
                 norm_layer=None, groups=1, width_per_group=64):
        super(ResNet_34, self).__init__()
        self.nclasses = nclasses

        self.use_range = params["train"]["input_depth"]["range"]
        self.use_xyz = params["train"]["input_depth"]["xyz"]
        self.use_remission = params["train"]["input_depth"]["remission"]
        self.input_depth = 0
        self.input_idxs = []
        if self.use_range:
            self.input_depth += 1
            self.input_idxs.append(0)
        if self.use_xyz:
            self.input_depth += 3
            self.input_idxs.extend([1, 2, 3])
        if self.use_remission:
            self.input_depth += 1
            self.input_idxs.append(4)
        dim = 32
        self.inc_range = MCFE(1, dim, block_index='range')
        self.inc_zxy = MCFE(3, dim, block_index='zxy')
        self.inc_remission = MCFE(1, dim, block_index='remission')
        self.merge = nn.Sequential(nn.Conv2d(dim * 3, dim, kernel_size=1, padding=0),
                                   nn.BatchNorm2d(dim),
                                   nn.LeakyReLU())

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.if_BN = if_BN
        self.dilation = 1
        self.aux = params["train"]["aux_loss"]["use"]

        self.groups = groups
        self.base_width = width_per_group


        self.conv1 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(128, 128, kernel_size=3, padding=1)

        self.inplanes = 128

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        self.decoder41 = AEUF()
        self.decoder31 = AEUF()
        self.decoder21 = AEUF()
        self.decoder42 = nn.Sequential(BasicConv2d(256, 128, 3, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU()
                                      )
        self.decoder32 = nn.Sequential(BasicConv2d(256, 128, 3, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU()
                                      )
        self.decoder22 = nn.Sequential(BasicConv2d(256, 128, 3, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU()
                                      )
        self.decoder12 = nn.Sequential(BasicConv2d(256, 128, 3, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU()
                                      )

        self.dropout = nn.Dropout2d(p=0.2)

        self.fusion_conv = BasicConv2d(128 * 4, 128, kernel_size=1)
        self.semantic_output = nn.Conv2d(128, nclasses, 1)

        if self.aux:
            self.aux_head4 = nn.Conv2d(128, nclasses, 1)
            self.aux_head3 = nn.Conv2d(128, nclasses, 1)
            self.aux_head2 = nn.Conv2d(128, nclasses, 1)
            self.aux_head1 = nn.Conv2d(128, nclasses, 1)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.if_BN:
                downsample = nn.Sequential(
                    # conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.AvgPool2d(kernel_size=(3, 3), stride=2, padding=1),
                    # SoftPool2d(kernel_size=(2, 2), stride=(2, 2)),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    # conv1x1(self.inplanes, planes * block.expansion, stride)
                    # SoftPool2d(kernel_size=(2, 2), stride=(2, 2))
                    # nn.AvgPool2d(kernel_size=(3, 3), stride=2, padding=1),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, if_BN=self.if_BN))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(planes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                if_BN=self.if_BN))

        return nn.Sequential(*layers)

    def forward(self, x):

        range = x[:, 0, :, :].unsqueeze(1)
        zxy = x[:, 1:4, :, :]
        remission = x[:, -1, :, :].unsqueeze(1)

        range = self.inc_range(range)
        zxy = self.inc_zxy(zxy)
        remission = self.inc_remission(remission)
        x = torch.cat((range, zxy, remission), dim=1)
        x = self.merge(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_1 = self.layer1(x)  # 1
        x_2 = self.layer2(x_1)  # 1/2
        x_3 = self.layer3(x_2)  # 1/4
        x_4 = self.layer4(x_3)  # 1/8

        x4 = self.decoder41(x_4, x_4)
        res_4 = F.interpolate(x4, size=x_1.size()[2:], mode='bilinear', align_corners=True)
        res__4 = self.dropout(self.decoder42(torch.cat((res_4, res_4), dim=1)))

        x3 = self.decoder31(x_3, x_3)
        res_3 = F.interpolate(x3, size=x_1.size()[2:], mode='bilinear', align_corners=True)
        res__3 = self.dropout(self.decoder32(torch.cat((res_3, res__4), dim=1)))

        x2 = self.decoder21(x_2, x_2)
        res_2 = F.interpolate(x2, size=x_1.size()[2:], mode='bilinear', align_corners=True)
        res__2 = self.dropout(self.decoder22(torch.cat((res_2, res__3), dim=1)))

        x1 = x_1
        res_1 = F.interpolate(x1, size=x.size()[2:], mode='bilinear', align_corners=True)
        res__1 = self.dropout(self.decoder12(torch.cat((res_1, res__2), dim=1)))

        res = [res__4, res__3, res__2, res__1]

        out = torch.cat(res, dim=1)
        out = self.fusion_conv(out)
        out = self.semantic_output(out)
        logits = F.softmax(out, dim=1)

        if self.aux:
            res__4 = self.aux_head4(res__4)
            res__4 = F.softmax(res__4, dim=1)

            res__3 = self.aux_head3(res__3)
            res__3 = F.softmax(res__3, dim=1)

            res__2 = self.aux_head2(res__2)
            res__2 = F.softmax(res__2, dim=1)

            res__1 = self.aux_head1(res__1)
            res__1 = F.softmax(res__1, dim=1)

        if self.aux:
            return [logits, res__4, res__3, res__2, res__1]
        else:
            return logits, out