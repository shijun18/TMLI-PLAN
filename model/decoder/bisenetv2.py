import torch
import torch.nn as nn
import torch.nn.functional as F



class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1, 
            groups=1,
            use_batchnorm=True,
            norm_layer=None,
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
            dilation=dilation,
            groups=groups,
            bias=not (use_batchnorm),
        )

        if use_batchnorm:
            bn = norm_layer(out_channels)

        else:
            bn = nn.Identity()

        relu = nn.ReLU(inplace=True)

        super(Conv2dReLU, self).__init__(conv, bn, relu)



class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)




class DetailBranch(nn.Module):

    def __init__(self,in_channels,out_channels,**kwargs):
        super(DetailBranch, self).__init__()
        assert len(out_channels) == 2
        self.S1 = nn.Sequential(
            Conv2dReLU(in_channels, out_channels[0], 3, stride=2, padding=1, **kwargs),
            Conv2dReLU(out_channels[0], out_channels[0], 3, stride=1, padding=1,**kwargs),
        )
        self.S2 = nn.Sequential(
            Conv2dReLU(out_channels[0], out_channels[0], 3, stride=2, padding=1,**kwargs),
            Conv2dReLU(out_channels[0], out_channels[0], 3, stride=1, padding=1,**kwargs),
            Conv2dReLU(out_channels[0], out_channels[0], 3, stride=1, padding=1,**kwargs),
        )
        self.S3 = nn.Sequential(
            Conv2dReLU(out_channels[0], out_channels[1], 3, stride=2, padding=1, **kwargs),
            Conv2dReLU(out_channels[1], out_channels[1], 3, stride=1, padding=1,**kwargs),
            Conv2dReLU(out_channels[1], out_channels[1], 3, stride=1, padding=1,**kwargs),
        )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat


class CEBlock(nn.Module):

    def __init__(self,in_channels,out_channels,**kwargs):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv_gap = Conv2dReLU(in_channels, in_channels, 1, stride=1, padding=0,**kwargs)
        self.conv_last = Conv2dReLU(in_channels, out_channels, 3, stride=1, padding=1,**kwargs)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat


'''
class StemBlock(nn.Module):

    def __init__(self,in_channels,out_channels,**kwargs):
        super(StemBlock, self).__init__()
        self.conv = Conv2dReLU(in_channels, out_channels, 3, stride=2)
        self.left = nn.Sequential(
            Conv2dReLU(out_channels, out_channels//2, 1, stride=1, padding=0,**kwargs),
            Conv2dReLU(out_channels//2, out_channels, 3, stride=2, **kwargs),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = Conv2dReLU(int(2*out_channels), out_channels, 3, stride=1,**kwargs)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return 



class GELayerS1(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6,**kwargs):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = Conv2dReLU(in_chan, in_chan, 3, stride=1,**kwargs)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat



class GELayerS2(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6,**kwargs):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = Conv2dReLU(in_chan, in_chan, 3, stride=1,**kwargs)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chan, in_chan, kernel_size=3, stride=2,
                    padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(
                    in_chan, out_chan, kernel_size=1, stride=1,
                    padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat

'''


class BGALayer(nn.Module):

    def __init__(self,in_chan):
        super(BGALayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                in_chan, in_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.Conv2d(
                in_chan, in_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                in_chan, in_chan, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                in_chan, in_chan, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(in_chan),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                in_chan, in_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.Conv2d(
                in_chan, in_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=4)
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_chan, in_chan, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up1(right1)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.up2(right)
        out = self.conv(left + right)
        return out




class BiSenetv2Decoder(nn.Module):
    def __init__(
            self,
            in_channels,
            encoder_channels,
            encoder_outindice,
            decoder_channels,
            use_batchnorm=True,
            norm_layer=None
    ):
        super().__init__()
        kwargs = dict(use_batchnorm=use_batchnorm, norm_layer=norm_layer)

        assert len(encoder_outindice) == 1
        encoder_channels = encoder_channels[encoder_outindice[0]]
        mid_chan = int(2*decoder_channels[-1])
        self.encoder_outindice = encoder_outindice[0]
        
        self.ce = CEBlock(encoder_channels,decoder_channels[-1],**kwargs)
        self.detail = DetailBranch(in_channels, decoder_channels,**kwargs)
        self.bga = BGALayer(decoder_channels[-1])
        self.conv_out = Conv2dReLU(decoder_channels[-1],mid_chan,3,stride=1,padding=1,**kwargs)

        self.init_weight()

    def forward(self, x, *features):
        feat_d = self.detail(x)
        feat_s = features[self.encoder_outindice] #x32
        feat_s = self.ce(feat_s)
        feat_head = self.bga(feat_d, feat_s)
        feat_out = self.conv_out(feat_head)

        return feat_out


    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)