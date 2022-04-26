import os 
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data_utils.data_loader import DataGenerator, To_Tensor, CropResize, Trunc_and_Normalize
from torch.cuda.amp import autocast as autocast
from utils import get_weight_path,multi_dice,multi_hd
import warnings
warnings.filterwarnings('ignore')

def get_net(net_name,encoder_name,channels=1,num_classes=2,input_shape=(512,512)):

    if net_name == 'unet':
        if encoder_name in ['simplenet','swin_transformer','swinplusr18']:
            from model.unet import unet
            net = unet(net_name,encoder_name=encoder_name,in_channels=channels,classes=num_classes,aux_classifier=True)
        else:
            import segmentation_models_pytorch as smp
            net = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=channels,
                classes=num_classes,                     
                aux_params={"classes":num_classes-1} 
            )
    elif net_name == 'unet++':
        if encoder_name is None:
            raise ValueError(
                "encoder name must not be 'None'!"
            )
        else:
            import segmentation_models_pytorch as smp
            net = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=channels,
                classes=num_classes,                     
                aux_params={"classes":num_classes-1} 
            )

    elif net_name == 'FPN':
        if encoder_name is None:
            raise ValueError(
                "encoder name must not be 'None'!"
            )
        else:
            import segmentation_models_pytorch as smp
            net = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=channels,
                classes=num_classes,                     
                aux_params={"classes":num_classes-1} 
            )
    
    elif net_name == 'deeplabv3+':
        if encoder_name in ['swinplusr18']:
            from model.deeplabv3plus import deeplabv3plus
            net = deeplabv3plus(net_name,encoder_name=encoder_name,in_channels=channels,classes=num_classes)
        else:
            import segmentation_models_pytorch as smp
            net = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=channels,
                classes=num_classes,                     
                aux_params={"classes":num_classes-1} 
            )
    elif net_name == 'res_unet':
        from model.res_unet import res_unet
        net = res_unet(net_name,encoder_name=encoder_name,in_channels=channels,classes=num_classes)
    
    elif net_name == 'att_unet':
        from model.att_unet import att_unet
        net = att_unet(net_name,encoder_name=encoder_name,in_channels=channels,classes=num_classes)
    
    ## external transformer + U-like net
    elif net_name == 'UTNet':
        from model.trans_model.utnet import UTNet
        net = UTNet(channels, base_chan=32,num_classes=num_classes, reduce_size=8, block_list='1234', num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True)
    elif net_name == 'UTNet_encoder':
        from model.trans_model.utnet import UTNet_Encoderonly
        # Apply transformer blocks only in the encoder
        net = UTNet_Encoderonly(channels, base_chan=32, num_classes=num_classes, reduce_size=8, block_list='1234', num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True)
    elif net_name =='TransUNet':
        from model.trans_model.transunet import VisionTransformer as ViT_seg
        from model.trans_model.transunet import CONFIGS as CONFIGS_ViT_seg
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = num_classes 
        config_vit.n_skip = 3 
        config_vit.patches.grid = (int(input_shape[0]/16), int(input_shape[1]/16))
        net = ViT_seg(config_vit, img_size=input_shape[0], num_classes=num_classes)
        #net.load_from(weights=np.load('./initmodel/R50+ViT-B_16.npz')) # uncomment this to use pretrain model download from TransUnet git repo

    elif net_name == 'ResNet_UTNet':
        from model.trans_model.resnet_utnet import ResNet_UTNet
        net = ResNet_UTNet(channels, num_classes, reduce_size=8, block_list='1234', num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True)
    
    elif net_name == 'SwinUNet':
        from model.trans_model.swin_unet import SwinUnet, SwinUnet_config
        config = SwinUnet_config()
        config.num_classes = num_classes
        config.in_chans = channels
        net = SwinUnet(config, img_size=input_shape[0], num_classes=num_classes)
    
    return net


def eval_process(test_path,config):
    # data loader
    test_transformer = transforms.Compose([
                Trunc_and_Normalize(config.scale),
                CropResize(dim=config.input_shape,num_class=config.num_classes,crop=config.crop),
                To_Tensor(num_class=config.num_classes)
            ])

    test_dataset = DataGenerator(test_path,
                                roi_number=config.roi_number,
                                num_class=config.num_classes,
                                transform=test_transformer)

    test_loader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)
    
    # get weight
    weight_path = get_weight_path(config.ckpt_path)
    print(weight_path)

    # get net
    net = get_net(config.net_name,config.encoder_name,config.channels,config.num_classes,config.input_shape)
    checkpoint = torch.load(weight_path)
    # print(checkpoint['state_dict'])
    net.load_state_dict(checkpoint['state_dict'])

    pred = []
    true = []
    net = net.cuda()
    net.eval()

    with torch.no_grad():
        for step, sample in enumerate(test_loader):
            data = sample['image']
            target = sample['mask']

            data = data.cuda()

            with autocast(True):
                output = net(data)
            if isinstance(output,tuple) or isinstance(output,list):
                seg_output = output[0]
            else:
                seg_output = output
            seg_output = torch.argmax(torch.softmax(seg_output, dim=1),1).detach().cpu().numpy()                          
            target = torch.argmax(target,1).detach().cpu().numpy()
            pred.append(seg_output)
            true.append(target)
    pred = np.concatenate(pred,axis=0)
    true = np.concatenate(true,axis=0)

    return pred,true


class Config:
    input_shape = (512,512) #(256,256)(512,512)(448,448) 
    num_classes = 8
    channels = 1
    crop = 0
    scale = (-200,600)
    roi_number = None
    net_name = 'unet'
    encoder_name = 'simplenet'
    version = 'v1.0.4'
    fold = 1
    ckpt_path = f'./ckpt/TMLI_UP/seg/{version}/All/fold{str(fold)}'


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # test data
    data_path = '/staff/shijun/dataset/Med_Seg/TMLI/up_2d_test_data'
    # data_path = '/staff/shijun/dataset/Med_Seg/TMLI/2d_data'
    sample_list = ['202398', '202774', '202610', '20210811', '202563', '202818', '202397', '202899', '202414', '202561']
    sample_list.sort()
    start = time.time()
    config = Config()
    
    for fold in range(1,6):
        print('>>>>>>>>>>>> Fold%d >>>>>>>>>>>>'%fold)
        total_dice = []
        total_hd = []
        info_dice = []
        info_hd = []
        config.fold = fold
        config.ckpt_path = f'./ckpt/TMLI_UP/seg/{config.version}/All/fold{str(fold)}'
        for sample in sample_list:
            info_item_dice = []
            info_item_hd = []
            info_item_dice.append(sample)
            info_item_hd.append(sample)
            print('>>>>>>>>>>>> %s is being processed'%sample)
            test_path = [case.path for case in os.scandir(data_path) if case.name.split('_')[0] == sample]
            test_path.sort(key=lambda x:eval(x.split('_')[-1].split('.')[0]))
            print(len(test_path))
            pred,true = eval_process(test_path,config)
            
            # print(pred.shape,true.shape)

            category_dice, avg_dice = multi_dice(true,pred,config.num_classes - 1)
            total_dice.append(category_dice)
            print('category dice:',category_dice)
            print('avg dice: %s'% avg_dice)

            category_hd, avg_hd = multi_hd(true,pred,config.num_classes - 1)
            total_hd.append(category_hd)
            print('category hd:',category_hd)
            print('avg hd: %s'% avg_hd)

            info_item_dice.extend(category_dice)
            info_item_hd.extend(category_hd)

            info_dice.append(info_item_dice)
            info_hd.append(info_item_hd)

        dice_csv = pd.DataFrame(data=info_dice)
        hd_csv = pd.DataFrame(data=info_hd)
        dice_csv.to_csv(f'./result/raw_data/{config.version}_fold{config.fold}_dice.csv')
        hd_csv.to_csv(f'./result/raw_data/{config.version}_fold{config.fold}_hd.csv')

        total_dice = np.stack(total_dice,axis=0) #sample*classes
        total_category_dice = np.mean(total_dice,axis=0)
        total_avg_dice = np.mean(total_category_dice)

        print('total category dice mean:',total_category_dice)
        print('total category dice std:',np.std(total_dice,axis=0))
        print('total dice mean: %s'% total_avg_dice)


        total_hd = np.stack(total_hd,axis=0) #sample*classes
        total_category_hd = np.mean(total_hd,axis=0)
        total_avg_hd = np.mean(total_category_hd)

        print('total category hd mean:',total_category_hd)
        print('total category hd std:',np.std(total_hd,axis=0))
        print('total hd mean: %s'% total_avg_hd)

        print("runtime:%.3f"%(time.time() - start))