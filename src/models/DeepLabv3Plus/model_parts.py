import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet


class BasicBlockWithDilation(torch.nn.Module):
    """Workaround for prohibited dilation in BasicBlock in 0.4.0"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockWithDilation, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = resnet.conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU()
        self.conv2 = resnet.conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


_basic_block_layers = {
    'resnet18': (2, 2, 2, 2),
    'resnet34': (3, 4, 6, 3),
}


def get_encoder_channel_counts(encoder_name):
    is_basic_block = encoder_name in _basic_block_layers
    ch_out_encoder_bottleneck = 512 if is_basic_block else 2048
    ch_out_encoder_4x = 64 if is_basic_block else 256
    return ch_out_encoder_bottleneck, ch_out_encoder_4x


class Encoder(torch.nn.Module):
    def __init__(self, name, **encoder_kwargs):
        super().__init__()
        encoder = self._create(name, **encoder_kwargs)
        del encoder.avgpool
        del encoder.fc
        self.encoder = encoder

    def _create(self, name, **encoder_kwargs):
        if name not in _basic_block_layers.keys():
            fn_name = getattr(resnet, name)
            model = fn_name(**encoder_kwargs)
        else:
            # special case due to prohibited dilation in the original BasicBlock
            pretrained = encoder_kwargs.pop('pretrained', False)
            progress = encoder_kwargs.pop('progress', True)
            model = resnet._resnet(
                name, BasicBlockWithDilation, _basic_block_layers[name], pretrained, progress, **encoder_kwargs
            )

        replace_stride_with_dilation = encoder_kwargs.get('replace_stride_with_dilation', (False, False, False))
        assert len(replace_stride_with_dilation) == 3
        if replace_stride_with_dilation[0]:
            model.layer2[0].conv2.padding = (2, 2)
            model.layer2[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[1]:
            model.layer3[0].conv2.padding = (2, 2)
            model.layer3[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[2]:
            model.layer4[0].conv2.padding = (2, 2)
            model.layer4[0].conv2.dilation = (2, 2)
        return model

    def update_skip_dict(self, skips, x, sz_in):
        rem, scale = sz_in % x.shape[3], sz_in // x.shape[3]
        assert rem == 0
        skips[scale] = x

    def forward(self, x):
        """
        DeepLabV3+ style encoder
        :param x: RGB input of reference scale (1x)
        :return: dict(int->Tensor) feature pyramid mapping downscale factor to a tensor of features
        """
        out = {1: x}
        sz_in = x.shape[3]

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.maxpool(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer1(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer2(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer3(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer4(x)
        self.update_skip_dict(out, x, sz_in)

        return out


class DecoderDeeplabV3p(torch.nn.Module):

    def __init__(self, bottleneck_ch, skip_4x_ch, num_out_ch):
        """
        :param bottleneck_ch: num aspp out ch
        :param skip_4x_ch: num encoder low level out ch
        :param num_out_ch: out ch
        """
        super(DecoderDeeplabV3p, self).__init__()
        # TODO: Implement a proper decoder with skip connections instead of the following
        self.num_conv1x1_out_ch = 48 # as in the original paper
        self.features_to_predictions = torch.nn.Conv2d(bottleneck_ch + self.num_conv1x1_out_ch,
                                                       num_out_ch,
                                                       kernel_size=1,
                                                       stride=1)
        self.conv1x1 = ASPPpart(skip_4x_ch, self.num_conv1x1_out_ch, kernel_size=1, stride=1, padding=0, dilation=1)
        self.up_conv = nn.Sequential(*ASPPpart(bottleneck_ch + self.num_conv1x1_out_ch,
                                              bottleneck_ch,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              dilation=1),
                                     ASPPpart(bottleneck_ch,
                                              bottleneck_ch,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              dilation=1) # padding?
        )

    def forward(self, features_bottleneck, features_skip_4x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4, from aspp
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.

        # upsample ASPP features by 4
        features_4x = F.interpolate(
            features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
        )

        # reduce channels of backbone features
        features_skip_4x_conv = self.conv1x1(features_skip_4x)

        # concat high- and low-level features
        out = torch.cat([features_4x, features_skip_4x_conv], dim=1)
        print("out shape after concat: ", out.shape)

        # up conv
        out = self.up_conv(out)

        # upsample conv features by 4
        predictions_4x = F.interpolate(
            out, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
        )

        return predictions_4x, features_4x


class ASPPpart(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

class ASPPPool(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        
    def forward(self, x):
        size = x.shape[-2:] 
        print(size)
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)  


class ASPP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, rates=(3, 6, 9)):
        super().__init__()
        # TODO: Implement ASPP properly instead of the following

        # standard 1x1 convolution
        self.aspp_conv1x1 = ASPPpart(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)

        # 3 dilation conv using different dilation rates
        self.aspp_dilation_convs = nn.ModuleList([ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rate, dilation=rate)
                                    for rate in rates])
        
            
        # image pooling
        self.aspp_pool = nn.Sequential(
            ASPPPool(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1),
        )

        # aspp feature aggregation
        self.aspp_aggr = nn.Sequential(ASPPpart(out_channels*5,
                                        out_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        dilation=1), 
                                      nn.Dropout2d(0.5))
            

        # make a list of conv modules
        self.convs = [
            self.aspp_conv1x1,
            self.aspp_dilation_convs,
            self.aspp_pool
        ]

    def forward(self, x):
        # TODO: Implement ASPP properly instead of the following
        out = []
        for m in self.convs:
            if isinstance(m, nn.ModuleList):
                out += [m_(x) for m_ in m]
            else:
                out += [m(x)]
        out = torch.cat(out, dim=1)
        out = self.aspp_aggr(out)

        return out


class MLP(torch.nn.Module):
    def __init__(self, dim, expansion):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj1 = nn.Linear(dim, int(dim * expansion))
        self.act = nn.GELU()
        self.proj2 = nn.Linear(int(dim * expansion), dim)

    def forward(self, x):
        """
        MLP with pre-normalization with one hidden layer
        :param x: batched tokens with dimesion dim (B,N,C)
        :return: tensor (B,N,C)
        """
        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        return x


class SelfAttention(torch.nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # TODO: Implement self attention, you need a projection
        # and the normalization layer

        self.dim_per_head = dim // num_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.temperature = torch.sqrt(torch.tensor(self.dim_per_head, dtype=torch.float32))
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, pos_embed):
        """
        Pre-normalziation style self-attetion
        :param x: batched tokens with dimesion dim (B,N,C)
        :param pos_embed: batched positional with shape (B, N, C)
        :return: tensor processed with output shape same as input
        """
        B, N, C = x.shape

        # TODO: Implement self attention, you need:
        # 1. obtain query, key, value
        # 2. if positional embed is not none add to the query tensor
        # 3. compute the similairty between query and key (dot product)
        # 4. obtain the convex combination of the values based on softmax of similarity
        # Remember that the num_heads speciifc the number of indepdendt heads to unpack
        # the operation in. Namely 2 heads, means that the channels are unpacked into 
        # 2 independent: something.reshape(B, N, self.num_heads, C // self.num_heads)
        # and rearrange accordingly to perform the operation such that you work only with
        # N and self.dim // self.num_heads
        # Remember that also the positional embedding needs to follow a similar rearrangement
        # to be consistent with the shapes of the query
        # Remember to rearrange the output tensor such that the output shape is B N C again

        x_norm = self.norm(x)

        Q = self.query(x_norm)
        K = self.key(x_norm)
        V = self.value(x_norm)

        Q = Q.reshape(B, N, self.num_heads, self.dim_per_head).transpose(1,2)
        K = K.reshape(B, N, self.num_heads, self.dim_per_head).transpose(1,2)
        V = V.reshape(B, N, self.num_heads, self.dim_per_head).transpose(1,2)

        if pos_embed is not None:
            Q += pos_embed.reshape(B, N, self.num_heads, self.dim_per_head).transpose(1,2)

        S = torch.matmul(Q, K.transpose(-2, -1))/self.temperature
        A = F.softmax(S, -1)
        A = self.dropout(A)

        values = torch.matmul(A,V)
        values = values.transpose(1,2).reshape(B,N,C)

        output = self.out(values) + x
        return output


class TransformerBlock(torch.nn.Module):
    def __init__(self, dim, num_heads=4, expansion=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = dim
        self.mlp = MLP(dim, expansion=expansion)
        self.attn = SelfAttention(dim=dim, num_heads=num_heads)

    def forward(self, x, pos_embed=None):
        x = self.attn(x, pos_embed) + x
        x = self.mlp(x) + x
        return x
    

class LatentsExtractor(torch.nn.Module):
    def __init__(self, num_latents: int = 256):
        super().__init__()
        edge = num_latents ** 0.5
        assert edge.is_integer(), "Remeber to give num_bins a number whose sqrt is still an integere, e.g., 256"
        self.edge = int(edge)
    
    def forward(self, x):
        B, C = x.shape[:2]
        x_down = F.interpolate(x, size=(self.edge, self.edge), mode="bilinear", align_corners=False)
        x_down_flat = x_down.permute(0, 2, 3, 1).view(B, self.edge*self.edge, C)
        return x_down_flat
    
    
def print_module_device(module):
    for name, param in module.named_parameters():
        device = param.device
        print(f"Module: {name}, Device: {device}")

    for name, sub_module in module.named_children():
        print_module_device(sub_module)