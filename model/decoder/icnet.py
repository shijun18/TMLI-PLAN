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




class CascadeFeatureFusion(nn.Module):
    """CFF Unit"""

    def __init__(self, low_channels, high_channels, out_channels, nclass, norm_layer=nn.BatchNorm2d):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            norm_layer(out_channels)
        )
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )
        self.conv_low_cls = nn.Conv2d(out_channels, nclass, 1, bias=False)

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)
        x_low_cls = self.conv_low_cls(x_low)

        return x, x_low_cls



class PyramidPoolingModule(nn.Module):
	def __init__(self, pyramids=[1,2,3,6]):
		super(PyramidPoolingModule, self).__init__()
		self.pyramids = pyramids

	def forward(self, input):
		feat = input
		height, width = input.shape[2:]
		for bin_size in self.pyramids:
			x = F.adaptive_avg_pool2d(input, output_size=bin_size)
			x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)
			feat  = feat + x
		return feat



class ICHead(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, nclass, norm_layer=nn.BatchNorm2d):
        super(ICHead, self).__init__()
        self.cff_24 = CascadeFeatureFusion(encoder_channels[1], encoder_channels[0], decoder_channels[-1], nclass, norm_layer)
        self.cff_12 = CascadeFeatureFusion(decoder_channels[-1], decoder_channels[-2], decoder_channels[-1], nclass, norm_layer)
        

        self.conv_cls = nn.Conv2d(decoder_channels[-1], nclass, 1, bias=False)

    def forward(self, x_sub1, x_sub2, x_sub4):
        outputs = list()
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outputs.append(x_24_cls)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)
        outputs.append(x_12_cls)

        up_x2 = F.interpolate(x_cff_12, scale_factor=2, mode='bilinear', align_corners=True)
        up_x2 = self.conv_cls(up_x2)
        outputs.append(up_x2)
        # 1/4 -> 1/8 -> 1/16
        outputs.reverse()

        return outputs




class ICnetDecoder(nn.Module):
    def __init__(
            self,
            in_channels,
            encoder_channels,
            encoder_outindice,
            decoder_channels,
            use_batchnorm=True,
            norm_layer=None,
            classes=2
    ):
        super().__init__()

        encoder_channels = [encoder_channels[i] for i in encoder_outindice]
        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, norm_layer=norm_layer)

        self.conv_sub1 = nn.Sequential(
            Conv2dReLU(in_channels, decoder_channels[0], 3, stride=2, padding=1,**kwargs),
            Conv2dReLU(decoder_channels[0], decoder_channels[0], 3, stride=2, padding=1,**kwargs),
            Conv2dReLU(decoder_channels[0], decoder_channels[1], 3, stride=2, padding=1,**kwargs)
        )

        self.ppm = PyramidPoolingModule()

        self.head = ICHead(encoder_channels,decoder_channels,classes,norm_layer)

    def forward(self, x, *features):
        x_sub1 = self.conv_sub1(x)
        x_sub2, x_sub4 = features

        x_sub4 = self.ppm(x_sub4)

        x,_,_ = self.head(x_sub1,x_sub2,x_sub4)

        return x
