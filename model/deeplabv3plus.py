from model.encoder.resnet import conv1x1
import torch.nn as nn
from typing import Optional, Union, List
from .model_config import MODEL_CONFIG
from .decoder.deeplabv3plus import DeepLabV3PlusDecoder
from .get_encoder import build_encoder
from .base_model import SegmentationModel



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class DeepLabV3Plus(SegmentationModel):
    """DeepLabV3+ implementation from "Encoder-Decoder with Atrous Separable
    Convolution for Semantic Image Segmentation"
    
    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        encoder_output_stride: Downsampling factor for last encoder features (see original paper for explanation)
        decoder_atrous_rates: Dilation rates for ASPP module (should be a tuple of 3 integer values)
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV3Plus**
    
    Reference:
        https://arxiv.org/abs/1802.02611v3
    """
    def __init__(
            self, 
            in_channels: int = 3,
            encoder_name: str = "resnet34",
            encoder_weights: Optional[str] = None,
            encoder_depth: int = 5,
            encoder_channels: List[int] = [32,64,128,256,512],
            encoder_output_stride: int = 16,
            decoder_channels: int = 256,
            decoder_atrous_rates: tuple = (12, 24, 36),
            upsampling: int = 4,
            classes: int = 1,
            aux_classifier: bool = False,
    ):
        super().__init__()

        self.encoder_channels = encoder_channels

        self.encoder = build_encoder(
            encoder_name, 
            weights=encoder_weights,
            n_channels=in_channels,
        )

        if encoder_output_stride == 8:
            self.make_dilated(
                stage_list=[3, 4],
                dilation_list=[2, 4]
            )

        elif encoder_output_stride == 16:
            self.make_dilated(
                stage_list=[4],
                dilation_list=[2]
            )
        elif encoder_output_stride == 32:
            pass
        else:
            raise ValueError(
                "Encoder output stride should be 8 or 16 or 32, got {}".format(encoder_output_stride)
            )

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )

        self.segmentation_head = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity(),
            nn.Conv2d(decoder_channels, classes, kernel_size=3, padding=1)
        )

        if aux_classifier:
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(self.encoder_channels[-1], classes - 1, bias=True)
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

                # Kostyl for EfficientNet
                if hasattr(mod, "static_padding"):
                    mod.static_padding = nn.Identity()



def deeplabv3plus(model_name,encoder_name,**kwargs):
    params = MODEL_CONFIG[model_name][encoder_name]
    dynamic_params = kwargs
    for key in dynamic_params:
        if key in params:
            params[key] = dynamic_params[key]

    net = DeepLabV3Plus(**params)
    return net