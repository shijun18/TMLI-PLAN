import sys
sys.path.append('..')
import torch
from model.encoder import swin_transformer,simplenet,trans_plus_conv,resnet,mobilenetv3,xception

moco_weight_path = {
    'resnet18':'/staff/shijun/torch_projects/Med_Seg/moco/convert_ckpt/moco_v2/resnet18/v1.0-x5/convert_epoch=0174-loss=0.537353-top1=98.037689.pth.tar',
    'swinplusr18':'/staff/shijun/torch_projects/Med_Seg/moco/convert_ckpt/moco_v2/swinplusr18/v5.0-x6/convert_epoch=0272-loss=7.834034-top1=81.369278.pth.tar'
}

def build_encoder(arch='resnet18', weights=None, **kwargs):
        
    arch = arch.lower()
    
    if arch.startswith('resnet'):
        backbone = resnet.__dict__[arch](**kwargs)
    elif arch.startswith('swin_transformer'):
        backbone = swin_transformer.__dict__[arch](**kwargs)
    elif arch.startswith('simplenet'):
        backbone = simplenet.__dict__[arch](**kwargs)
    elif arch.startswith('swinplus'):
        backbone = trans_plus_conv.__dict__[arch](**kwargs)
    elif arch.startswith('mobilenetv3'):
        backbone = mobilenetv3.__dict__[arch](**kwargs)
    elif arch.startswith('xception'):
        backbone = xception.__dict__[arch](**kwargs)
    else:
        raise Exception('Architecture undefined!')

    if weights is not None and isinstance(moco_weight_path[arch], str):
        print('Loading weights for backbone')
        msg = backbone.load_state_dict(
            torch.load(moco_weight_path[arch], map_location=lambda storage, loc: storage)['state_dict'], strict=False)
        if arch.startswith('resnet'):
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
            print(">>>> loaded pre-trained model '{}' ".format(moco_weight_path[arch]))
        print(msg)
    
    return backbone



if __name__ == '__main__':

    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    # net = build_encoder('swin_transformer',n_channels=1)
    # net = build_encoder('resnet18',n_channels=1)
    # net = build_encoder('swinplusr18',n_channels=1)
    # net = build_encoder('simplenet',n_channels=1)
    # net = build_encoder('swinplusr18','moco',n_channels=1)
    # net = build_encoder('mobilenetv3_large_075',n_channels=1)
    net = build_encoder('xception',n_channels=1)
    net = net.cuda()
    net.train()
    input = torch.randn((1,1,512,512)).cuda()
    output = net(input)
    for item in output:
        print(item.size())

    import sys
    sys.path.append('..')
    from utils import count_params_and_macs
    count_params_and_macs(net.cuda(),(1,1,512,512))