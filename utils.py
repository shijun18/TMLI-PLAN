import os,glob
import pandas as pd
import h5py
import numpy as np
import torch
import random
from skimage.metrics import hausdorff_distance

def binary_dice(y_true, y_pred):
    smooth = 1e-7
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def multi_dice(y_true,y_pred,num_classes):
    dice_list = []
    for i in range(num_classes):
        true = (y_true == i+1).astype(np.float32)
        pred = (y_pred == i+1).astype(np.float32)
        dice = binary_dice(true,pred)
        dice_list.append(dice)
    
    dice_list = [round(case, 4) for case in dice_list]
    
    return dice_list, round(np.mean(dice_list),4)


def hd_2d(true,pred):
    hd_list = []
    for i in range(true.shape[0]):
        if np.sum(true[i]) != 0 and np.sum(pred[i]) != 0:
            hd_list.append(hausdorff_distance(true[i],pred[i]))
    
    return np.mean(hd_list)

def multi_hd(y_true,y_pred,num_classes):
    hd_list = []
    for i in range(num_classes):
        true = (y_true == i+1).astype(np.float32)
        pred = (y_pred == i+1).astype(np.float32)
        hd = hd_2d(true,pred)
        hd_list.append(hd)
    
    hd_list = [round(case, 4) for case in hd_list]
    
    return hd_list, round(np.mean(hd_list),4)



def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image


def get_path_with_annotation(input_path,path_col,tag_col):
    path_list = pd.read_csv(input_path)[path_col].values.tolist()
    tag_list = pd.read_csv(input_path)[tag_col].values.tolist()
    final_list = []
    for path, tag in zip(path_list,tag_list):
        if tag != 0:
            final_list.append(path)
    
    return final_list


def get_path_with_annotation_ratio(input_path,path_col,tag_col,ratio=0.5):
    path_list = pd.read_csv(input_path)[path_col].values.tolist()
    tag_list = pd.read_csv(input_path)[tag_col].values.tolist()
    with_list = []
    without_list = []
    for path, tag in zip(path_list,tag_list):
        if tag != 0:
            with_list.append(path)
        else:
            without_list.append(path)
    if int(len(with_list)/ratio) < len(without_list):
        random.shuffle(without_list)
        without_list = without_list[:int(len(with_list)/ratio)]    
    return with_list + without_list


def count_params_and_macs(net,input_shape):
    
    from thop import profile
    input = torch.randn(input_shape)
    input = input.cuda()
    macs, params = profile(net, inputs=(input, ))
    print('%.3f GFLOPs' %(macs/10e9))
    print('%.3f M' % (params/10e6))



def get_weight_path(ckpt_path):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) != 0:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split(':')[-1]))
            return os.path.join(ckpt_path,pth_list[-1])
        else:
            return None
    else:
        return None
    
    

def remove_weight_path(ckpt_path,retain=3):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) >= retain:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split(':')[-1]))
            for pth_item in pth_list[:-retain]:
                os.remove(os.path.join(ckpt_path,pth_item))


def dfs_remove_weight(ckpt_path,retain=3):
    for sub_path in os.scandir(ckpt_path):
        if sub_path.is_dir():
            dfs_remove_weight(sub_path.path,retain)
        else:
            remove_weight_path(ckpt_path,retain)
            break  

if __name__ == "__main__":

    ckpt_path = './ckpt/TMLI_UP/seg/v9.0/All/fold1/'
    dfs_remove_weight(ckpt_path)