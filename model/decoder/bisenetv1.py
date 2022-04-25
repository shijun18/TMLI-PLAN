import torch
import torch.nn as nn
import torch.nn.functional as F



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



class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = Conv2dReLU(in_chan, out_chan, 3, stride=1, padding=1,**kwargs)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = atten.sigmoid()
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)



class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = Conv2dReLU(in_chan, out_chan, 1, stride=1, padding=0,**kwargs)
        self.conv1 = nn.Conv2d(out_chan, out_chan//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_chan//4, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = atten.sigmoid()
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)




class SpatialPath(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(SpatialPath,self).__init__()

        blocks = []
        for i, out_ch in enumerate(out_channels):
            if i == 0:
                block = Conv2dReLU(in_channels,out_ch,7,stride=2,padding=3,**kwargs)
            elif i == len(out_channels) - 1:
                block = Conv2dReLU(out_channels[i-1],out_ch,1,stride=1,padding=0,**kwargs)
            else:
                block = Conv2dReLU(out_channels[i-1],out_ch,3,stride=2,padding=1,**kwargs)
            blocks.append(block)

        self.blks = nn.ModuleList(blocks)
        self.init_weight()

    def forward(self, x):   
        for blk in self.blks:
            x = blk(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)



class ContextPath(nn.Module):
    def __init__(self, encoder_channels, out_channels, **kwargs):
        super(ContextPath, self).__init__()
        assert len(encoder_channels) == 2
        self.conv_avg = Conv2dReLU(encoder_channels[1], out_channels, 1, stride=1, padding=0, **kwargs)

        self.arm16 = AttentionRefinementModule(encoder_channels[0], out_channels, **kwargs)
        self.arm32 = AttentionRefinementModule(encoder_channels[1], out_channels, **kwargs)
        self.conv_head16 = Conv2dReLU(out_channels, out_channels, 3, stride=1, padding=1, **kwargs)
        self.conv_head32 = Conv2dReLU(out_channels, out_channels, 3, stride=1, padding=1, **kwargs)

        self.up16 = nn.Upsample(scale_factor=2.)
        self.up32 = nn.Upsample(scale_factor=2.)
        
        self.init_weight()
    
    def forward(self, feat16, feat32):
        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_up = self.up32(feat32_sum)
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = self.up16(feat16_sum)
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up, feat32_up # x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)



class BiSenetv1Decoder(nn.Module):
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

        assert len(encoder_outindice) == 2
        encoder_channels = [encoder_channels[i] for i in encoder_outindice]
        self.encoder_outindice = encoder_outindice
        cat_chan = int(2*decoder_channels[-1])
        
        self.cp = ContextPath(encoder_channels, decoder_channels[-1],**kwargs)
        self.sp = SpatialPath(in_channels, decoder_channels,**kwargs)
        self.ffm = FeatureFusionModule(cat_chan,cat_chan,**kwargs)
        self.conv_out = Conv2dReLU(cat_chan,cat_chan,3,stride=1,padding=1,**kwargs)

        self.init_weight()

    def forward(self, x, *features):

        feat16 = features[self.encoder_outindice[0]] #x16
        feat32 = features[self.encoder_outindice[1]] #x32

        feat_cp8, feat_cp16 = self.cp(feat16,feat32) #x8, x16
        feat_sp = self.sp(x)
        feat_fuse = self.ffm(feat_sp,feat_cp8)
        faet_out = self.conv_out(feat_fuse)

        return faet_out


    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)