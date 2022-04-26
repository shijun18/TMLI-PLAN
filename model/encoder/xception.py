import re
import torch.nn as nn
import timm
import numpy as np



class Xception(nn.Module):
    def __init__(self, model_name, depth=5, n_channels=1,num_classes=2):
        super().__init__()

        self.depth = depth
        self.in_channels = n_channels
        self.model = timm.create_model(
            model_name=model_name,
            scriptable=True,  # torch.jit scriptable
            exportable=True,  # onnx export
            in_chans=n_channels
        )

        # modify padding to maintain output shape
        self.model.conv1.padding = (1, 1)
        self.model.conv2.padding = (1, 1)

        del self.model.fc

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(
                self.model.conv1, 
                self.model.bn1, 
                self.model.act1, 
                self.model.conv2, 
                self.model.bn2, 
                self.model.act2
            ),
            self.model.block1,
            self.model.block2,
            nn.Sequential(
                self.model.block3,
                self.model.block4,
                self.model.block5,
                self.model.block6,
                self.model.block7,
                self.model.block8,
                self.model.block9,
                self.model.block10,
                self.model.block11,
            ),
            nn.Sequential(
                self.model.block12, 
                self.model.conv3, 
                self.model.bn3, 
                self.model.act3, 
                self.model.conv4, 
                self.model.bn4,
                self.model.act4
                ),
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self.depth + 1):
            x = stages[i](x)
            if i != 0:
                features.append(x)

        return features

    def load_state_dict(self, state_dict):
        # remove linear
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)

        super().load_state_dict(state_dict)


def xception(**kwargs):
    """ MobileNet V3 """
    model = Xception('xception', **kwargs)
    return model
