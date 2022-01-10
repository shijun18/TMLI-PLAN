from model.unet import unet
from model.att_unet import att_unet
from model.res_unet import res_unet

if __name__ == '__main__':

    from torchsummary import summary
    import torch
    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # unet
    # net = unet('resnet18_unet',in_channels=1,classes=2)
    # net = unet('unet',in_channels=1,classes=2)
    # net = unet('swin_trans_unet',in_channels=1,classes=2)
    net = unet('swinplusr18_unet',in_channels=1,classes=2)

    # att unet
    # net = att_unet('resnet18_att_unet',in_channels=1,classes=2)
    # net = att_unet('att_unet',in_channels=1,classes=2)
    # net = att_unet('swin_trans_att_unet',in_channels=1,classes=2)

    # res unet
    # net = res_unet('resnet18_res_unet',in_channels=1,classes=2)
    # net = res_unet('res_unet',in_channels=1,classes=2)
    # net = res_unet('swin_trans_res_unet',in_channels=1,classes=2)
      
    summary(net.cuda(),input_size=(1,512,512),batch_size=1,device='cuda')
    
    # net = net.cuda()
    # net.train()
    # input = torch.randn((1,1,512,512)).cuda()
    # output = net(input)
    # print(output.size())
    

    import sys
    sys.path.append('..')
    from utils import count_params_and_macs
    count_params_and_macs(net.cuda(),(1,1,512,512))