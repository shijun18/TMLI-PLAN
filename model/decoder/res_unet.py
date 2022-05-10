import torch
import torch.nn as nn
import torch.nn.functional as F

from model.module import attention as attention



class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
            norm_layer=None,
            activation=True
    ):

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = norm_layer
        

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = norm_layer(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu if activation else nn.Identity())



class ResDecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            norm_layer=None,
            attention_type=None,
    ):
        super().__init__()
        self.downsample = (in_channels + skip_channels) != out_channels
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
            norm_layer=norm_layer
        )
        self.attention1 = attention.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
            norm_layer=norm_layer,
            activation=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=1,
            use_batchnorm=use_batchnorm,
            norm_layer=norm_layer,
            activation=False
        )
        self.attention2 = attention.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear",align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            res = self.conv3(res)
        out += res
        out = self.relu(out)
        out = self.attention2(out)
        return out


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, norm_layer=None):
        conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
            norm_layer=norm_layer
        )
        conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
            norm_layer=norm_layer
        )
        super().__init__(conv1, conv2)



class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_chans, out_chans, sizes=(1, 2, 3, 6), norm_layer=None):
        super(PSPModule, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        len_size = len(sizes)
        self.stages = []
        self.stages = nn.ModuleList(
                [self._make_stage(in_chans, out_chans, size, norm_layer) for size in sizes]
            )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(int(in_chans + len_size*out_chans), out_chans, kernel_size=1, padding=0, dilation=1, bias=False),
            norm_layer(out_chans),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle



class ResUnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=4,
            use_batchnorm=True,
            norm_layer=None,
            attention_type=None,
            center=False,
            aux_deepvision=False
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )
        self.aux_deepvision = aux_deepvision
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:])
        out_channels = decoder_channels

        if center:
            # self.center = CenterBlock(
            #     head_channels, head_channels, use_batchnorm=use_batchnorm,norm_layer=norm_layer
            # )
            self.center = PSPModule(
                head_channels, head_channels,norm_layer=norm_layer
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, norm_layer=norm_layer, attention_type=attention_type)
        blocks = [
            ResDecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        if self.aux_deepvision:
            fpn_out = [
                Conv2dReLU(in_ch, out_channels[-1], kernel_size=3,padding=1,stride=1,use_batchnorm=use_batchnorm,norm_layer=norm_layer)
                for in_ch in out_channels[:-1]
            ]
            self.fpn_out = nn.ModuleList(fpn_out)

    def forward(self, *features):

        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        mid_out = []
        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

            if self.aux_deepvision:
                mid_out.append(
                    self.fpn_out[i](x) if i < len(self.fpn_out) else x
                )

        if self.aux_deepvision:
            mid_out.reverse() 
            output_size = mid_out[0].size()[2:]
            fusion_out = [mid_out[0]]
            for i in range(1,len(mid_out)):
                fusion_out.append(
                    F.interpolate(mid_out[i],output_size,mode='bilinear',align_corners=False)
                )
            return torch.cat(fusion_out, 1)

        else:
            return x
