import sys
sys.path.append('..')
import os
import json
import pandas as pd 
import numpy as np
from skimage import measure

from converter.utils import hdf5_reader



def csv_maker(input_path,save_path,label_list):
    csv_info = []
    entry = os.scandir(input_path)
    len_ = len(os.listdir(input_path))
    count = 0 
    
    for item in entry:
        csv_item = []
        csv_item.append(item.path)
        tag_array = np.zeros((len(label_list) + 1,),dtype=np.uint8)
        label = hdf5_reader(item.path,'label')
        # print(np.unique(label).astype(np.uint8))
        tag_array[np.unique(label).astype(np.uint8)] = 1
        csv_item.extend(list(tag_array[1:]))
        # print(item.path)
        # print(list(tag_array[1:]))
        csv_info.append(csv_item)
        count += 1
        sys.stdout.write('\r Current: %d / %d'%(count,len_))
    print('\n')
    col = ['path'] + label_list
    csv_file = pd.DataFrame(columns=col, data=csv_info)
    csv_file.to_csv(save_path, index=False)


def area_compute(input_path,save_path,label_list):
    csv_info = []
    entry = os.scandir(input_path)
    len_ = len(os.listdir(input_path))
    count = 0 
    
    for item in entry:
        csv_item = []
        csv_item.append(item.path)
        area_array = list(np.zeros((len(label_list),),dtype=np.uint8))
        label = hdf5_reader(item.path,'label')
        for i in range(len(label_list)):
            roi = (label==i+1).astype(np.uint8)
            roi = measure.label(roi)
            area = []
            for j in range(1,np.amax(roi) + 1):
                area.append(np.sum(roi == j))
            area_array[i] = area
        
        csv_item.extend(area_array)

        csv_info.append(csv_item)
        count += 1
        sys.stdout.write('\r Current: %d / %d'%(count,len_))
    print('\n')
    col = ['path'] + label_list
    csv_file = pd.DataFrame(columns=col, data=csv_info)
    csv_file.to_csv(save_path, index=False)


if __name__ == "__main__":
    json_file = './static_files/TMLI_config.json'
    with open(json_file, 'r') as fp:
        info = json.load(fp)
        input_path = info['2d_data']['save_path']
        save_path = info['2d_data']['csv_path']
        
        # for test data
        # input_path = info['2d_data']['test_path']
        # save_path = info['2d_data']['test_csv_path']
        
    csv_maker(input_path,save_path,info['annotation_list'])
