import torch
import torch.nn as nn
import torch.nn.functional as F
from . import resnet,swin_transformer


moco_weight_path = {
    'resnet18':None
}

def build_encoder(arch='resnet18', weights=None, **kwargs):
        
    arch = arch.lower()
    # print(arch)
    if arch.startswith('resnet'):
        backbone = resnet.__dict__[arch](classification=False,**kwargs)
    elif arch.startswith('swin_transformer'):
        backbone = swin_transformer.__dict__[arch](classification=False,**kwargs)
    else:
        raise Exception('Architecture undefined!')
        
    if weights is not None and isinstance(moco_weight_path[arch], str):
        print('Loading weights for backbone')
        msg = backbone.load_state_dict(
            torch.load(moco_weight_path[arch], map_location=lambda storage, loc: storage)['state_dict'], strict=False)
        print(msg)
    
    return backbone


def conv1x1(in_planes, out_planes):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class TransPlusConv(nn.Module):
    def __init__(self, trans_encoder, conv_encoder, conv_num_features, n_channels=1,classification=False, num_classes=2):
        super(TransPlusConv, self).__init__()
        self.classification = classification
        self.trans_encoder = build_encoder(trans_encoder,
                                        n_channels=n_channels,
                                        embed_dim=96,
                                        depths=[2, 2, 6, 2],
                                        num_heads=[3, 6, 12, 24],
                                        window_size=7,
                                        mlp_ratio=4.)
        self.trans_num_features = self.trans_encoder.num_features

        self.conv_encoder = build_encoder(conv_encoder,
                                        n_channels=n_channels)
        self.conv_num_features = conv_num_features
        
        assert len(self.conv_num_features) >= len(self.trans_num_features)
        self.depth = len(self.trans_num_features)
        self.offset = len(self.conv_num_features) - len(self.trans_num_features)
        self.conv_num_features = self.conv_num_features[self.offset:]
        fusion_layer = []
        for (n_trans,n_conv) in zip(self.trans_num_features,self.conv_num_features):
            layer = nn.Sequential(
                conv1x1(n_trans+n_conv,n_conv),
                nn.BatchNorm2d(n_conv),
                nn.ReLU(inplace=True)
            )
            fusion_layer.append(layer)
        self.fusion_layer = nn.ModuleList(fusion_layer)
        if self.classification:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.conv_num_features[-1], num_classes)

    def forward(self, x):
        trans_out = self.trans_encoder(x)
        conv_out = self.conv_encoder(x)
        out_x = []
        out_x += conv_out[:self.offset]
        conv_out = conv_out[self.offset:]

        for i,layer in enumerate(self.fusion_layer):
            cat_x = torch.cat([trans_out[i], conv_out[i]], dim=1)
            fusion_x = layer(cat_x)
            out_x.append(fusion_x)
        if self.classification:
            x = self.avgpool(out_x[-1])
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
        else:
            return out_x

    def get_stages(self):
        stages = []
        for i in range(self.offset):
            stages.append(nn.Identity())
        for i in range(self.depth):
            stages.append(self.fusion_layer[i])
        return stages


def swinplusr18(**kwargs):
    model = TransPlusConv(trans_encoder='swin_transformer',
                        conv_encoder='resnet18',
                        conv_num_features=[64,64,128,256,512],
                        **kwargs)
    return model


def swinplusr34(**kwargs):
    model = TransPlusConv(trans_encoder='swin_transformer',
                        conv_encoder='resnet34',
                        conv_num_features=[64,64,128,256,512],
                        **kwargs)
    return model



def swinplusr50(**kwargs):
    model = TransPlusConv(trans_encoder='swin_transformer',
                        conv_encoder='resnet50',
                        conv_num_features=[64,256,512,1024,2048],
                        **kwargs)
    return model