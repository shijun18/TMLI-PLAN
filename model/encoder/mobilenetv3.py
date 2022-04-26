import timm
import numpy as np
import torch.nn as nn


def _make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1.0 / divisible_by) * divisible_by)


class MobileNetV3(nn.Module):
    def __init__(self, model_name, width_mult, depth=5, n_channels=1,num_classes=2):
        super().__init__()
        if "large" not in model_name and "small" not in model_name:
            raise ValueError("MobileNetV3 wrong model name {}".format(model_name))

        self.mode = "small" if "small" in model_name else "large"
        self.depth = depth
        self.out_channels = self._get_channels(self.mode, width_mult)
        self.n_channels = n_channels
        # minimal models replace hardswish with relu
        self.model = timm.create_model(
            model_name=model_name,
            scriptable=True,  # torch.jit scriptable
            exportable=True,  # onnx export
            features_only=True,
            in_chans=n_channels
        )

    def _get_channels(self, mode, width_mult):
        if mode == "small":
            channels = [16, 16, 24, 48, 576]
        else:
            channels = [16, 24, 40, 112, 960]
        channels =  [_make_divisible(x * width_mult) for x in channels]
        return tuple(channels)

    def get_stages(self):
        if self.mode == "small":
            return [
                nn.Identity(),
                nn.Sequential(
                    self.model.conv_stem,
                    self.model.bn1,
                    self.model.act1,
                ),
                self.model.blocks[0],
                self.model.blocks[1],
                self.model.blocks[2:4],
                self.model.blocks[4:],
            ]
        elif self.mode == "large":
            return [
                nn.Identity(),
                nn.Sequential(
                    self.model.conv_stem,
                    self.model.bn1,
                    self.model.act1,
                    self.model.blocks[0],
                ),
                self.model.blocks[1],
                self.model.blocks[2],
                self.model.blocks[3:5],
                self.model.blocks[5:],
            ]
        else:
            ValueError("MobileNetV3 mode should be small or large, got {}".format(self._mode))

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self.depth + 1):
            x = stages[i](x)
            if i != 0:
                features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("conv_head.weight", None)
        state_dict.pop("conv_head.bias", None)
        state_dict.pop("classifier.weight", None)
        state_dict.pop("classifier.bias", None)
        self.model.load_state_dict(state_dict, **kwargs)



def mobilenetv3_large_075(**kwargs):
    """ MobileNet V3 """
    model = MobileNetV3('tf_mobilenetv3_large_075', 0.75, **kwargs)
    return model


def mobilenetv3_large_100(**kwargs):
    """ MobileNet V3 """
    model = MobileNetV3('tf_mobilenetv3_large_100', 1.0, **kwargs)
    return model


def mobilenetv3_large_minimal_100(**kwargs):
    """ MobileNet V3 """
    model = MobileNetV3('tf_mobilenetv3_large_minimal_100', 1.0, **kwargs)
    return model


def mobilenetv3_small_075(**kwargs):
    """ MobileNet V3 """
    model = MobileNetV3('tf_mobilenetv3_small_075', 0.75, **kwargs)
    return model


def mobilenetv3_small_100(**kwargs):
    """ MobileNet V3 """
    model = MobileNetV3('tf_mobilenetv3_small_100', 1.0, **kwargs)
    return model


def mobilenetv3_small_minimal_100(**kwargs):
    """ MobileNet V3 """
    model = MobileNetV3('tf_mobilenetv3_small_minimal_100', 1.0, **kwargs)
    return model
