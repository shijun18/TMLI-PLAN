import sys
sys.path.append('..')
from model.unet import unet
from model.att_unet import att_unet
from model.res_unet import res_unet
from model.deeplabv3plus import deeplabv3plus
from model.bisenetv1 import bisenetv1

if __name__ == '__main__':

    from torchsummary import summary
    import torch
    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    # unet
    # net = unet('unet',encoder_name='simplenet',in_channels=1,classes=2)
    # net = unet('unet',encoder_name='swin_transformer',in_channels=1,classes=2)
    # net = unet('unet',encoder_name='swinplusr18',in_channels=1,classes=2)
    # net = unet('unet',encoder_name='resnet18',in_channels=1,classes=2)

    # att unet
    # net = att_unet('att_unet',encoder_name='simplenet',in_channels=1,classes=2)
    # net = att_unet('att_unet',encoder_name='swin_transformer',in_channels=1,classes=2)
    # net = att_unet('att_unet',encoder_name='resnet18',in_channels=1,classes=2)

    # res unet
    # net = res_unet('res_unet',encoder_name='simplenet',in_channels=1,classes=2)
    # net = res_unet('res_unet',encoder_name='resnet18',in_channels=1,classes=2)
    # net = res_unet('res_unet',encoder_name='swinplusr18',in_channels=1,classes=2)

    #deeplabv3+
    # net = deeplabv3plus('deeplabv3+',encoder_name='swinplusr18',in_channels=1,classes=2)

    #bisenetv1
    # net = bisenetv1('bisenetv1','resnet18',in_channels=1,classes=2)
    # net = bisenetv1('bisenetv1','swin_transformer',in_channels=1,classes=2)
    # net = bisenetv1('bisenetv1','mobilenetv3_large_075',in_channels=1,classes=2)
    net = bisenetv1('bisenetv1','xception',in_channels=1,classes=2)

    summary(net.cuda(),input_size=(1,512,512),batch_size=1,device='cuda')
    
    net = net.cuda()
    net.train()
    input = torch.randn((2,1,512,512)).cuda()
    output = net(input)
    print(output.size())
    
    from utils import count_params_and_macs
    count_params_and_macs(net.cuda(),(1,1,512,512))