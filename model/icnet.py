import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
from .model_config import MODEL_CONFIG
from .decoder.icnet import ICnetDecoder
from .get_encoder import build_encoder
from .base_model import SegmentationModel



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ICnet(SegmentationModel):
    """
    Args:
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_channels: List of integers which specify **out_channels** parameter for convolutions used in encoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNormalization layer between Conv2D and Activation layers is used.
            Available options are **True, False**.
        decoder_attention_type: Attention module used in decoder of the model. Available options are **None** and **scse**.
            SCSE paper - https://arxiv.org/abs/1808.08127
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        upsampling: Int number of upsampling factor for segmentation head, default=1 
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        aux_classifier: If **True**, add a classification branch based the last feature of the encoder.
            Available options are **True, False**.
    Returns:
        ``torch.nn.Module``: ICnet
    """

    def __init__(
        self,
        in_channels: int = 3,
        encoder_name: str = "resnet18",
        encoder_weights: Optional[str] = None,
        encoder_depth: int = 5,
        encoder_channels: List[int] = [32,64,128,256,512],
        encoder_outindice: List[int] = [2,4],
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = [32,64,128],
        upsampling: int = 1,
        classes: int = 1,
        aux_classifier: bool = False,
    ):
        super().__init__()

        assert len(encoder_outindice) == 2

        self.encoder_depth = encoder_depth
        self.encoder_channels = encoder_channels
        self.encoder_outindice = encoder_outindice

        self.encoder = build_encoder(
            encoder_name,
            weights=encoder_weights,
            n_channels=in_channels
        )

        if encoder_name.startswith('resnet'):
            self.make_dilated(
                    stage_list=[3, 4],
                    dilation_list=[2, 4]
                )

        self.decoder = ICnetDecoder(
            in_channels=in_channels,
            encoder_channels=self.encoder_channels,
            encoder_outindice=encoder_outindice,
            decoder_channels=decoder_channels,
            use_batchnorm=decoder_use_batchnorm,
            norm_layer=nn.BatchNorm2d,
            classes=classes
        )

        self.segmentation_head = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        )

        if aux_classifier:
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(self.encoder_channels[encoder_outindice[-1]], classes - 1, bias=True)
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()


    def make_dilated(self, stage_list, dilation_list):
        stages = self.encoder.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            self.replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )
    
    def replace_strides_with_dilation(self, module, dilation_rate):
        """Patch Conv2d modules replacing strides with dilation"""
        for mod in module.modules():
            if isinstance(mod, nn.Conv2d):
                mod.stride = (1, 1)
                mod.dilation = (dilation_rate, dilation_rate)
                kh, kw = mod.kernel_size
                mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)


    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = list()
        x_sub2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        feat_sub2 = self.encoder(x_sub2)[self.encoder_outindice[0]]
        features.append(feat_sub2)

        x_sub4 =  F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        feat_sub4 = self.encoder(x_sub4)[self.encoder_outindice[1]]
        features.append(feat_sub4)

        decoder_output = self.decoder(x,*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks



def icnet(model_name,encoder_name,**kwargs):
    params = MODEL_CONFIG[model_name][encoder_name]
    dynamic_params = kwargs
    for key in dynamic_params:
        if key in params:
            params[key] = dynamic_params[key]

    net = ICnet(**params)
    return net
