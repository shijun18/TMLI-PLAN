import sys
sys.path.append('..')
import torch
from model.encoder import swin_transformer,simplenet,trans_plus_conv,resnet



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
    else:
        raise Exception('Architecture undefined!')

    if weights is not None and isinstance(weights, str):
        print('Loading weights for backbone')
        backbone.load_state_dict(
            torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
    
    return backbone



if __name__ == '__main__':

    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # net = build_encoder('swin_transformer',n_channels=1)
    # net = build_encoder('resnet18',n_channels=1)
    net = build_encoder('swinplusr18',n_channels=1)
    # net = build_encoder('simplenet',n_channels=1)
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