import sys

from cv2 import data
sys.path.append('..')
import os
import glob
from tqdm import tqdm
import time
import shutil
import json
import numpy as np

from converter.dicom_reader import Dicom_Reader
from converter.utils import save_as_hdf5

# dicom series and rt in different directories.
def dicom_to_hdf5(input_path, save_path, annotation_list, target_format, resample=True):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)

    path_list = os.listdir(input_path)
    start = time.time()
    for ID in tqdm(path_list):
        print('=============%s in Processing============='%ID)
        tmp_images = []
        tmp_labels = []
        data_path = os.path.join(input_path,ID)
        for sub in ['up','down']:
            sub_path = os.path.join(data_path,sub)
            if os.path.exists(sub_path):
                series_path = glob.glob(os.path.join(sub_path, '*' + ID + '*CT*'))[0]
                rt_path = glob.glob(os.path.join(sub_path, '*' + ID + '*RT*'))[0]
                rt_path = glob.glob(os.path.join(rt_path, '*.dcm'))[0]
                
                try:
                    reader = Dicom_Reader(series_path, target_format, rt_path, annotation_list,trunc_flag=False, normalize_flag=False)
                except:
                    print("Error data: %s" % ID)
                    continue
                else:
                    if resample:
                        images = reader.get_resample_images()
                        labels = reader.get_resample_labels()
                    else:
                        images = reader.get_raw_images()
                        labels = reader.get_raw_labels()
                    tmp_images.append(images)
                    tmp_labels.append(labels)
        images = np.concatenate(tmp_images,axis=0).astype(np.int16)
        labels = np.concatenate(tmp_labels,axis=0).astype(np.uint8)
        hdf5_path = os.path.join(save_path, ID + '.hdf5')
        print("=================%s done!================="%ID)

        save_as_hdf5(images, hdf5_path, 'image')
        save_as_hdf5(labels, hdf5_path, 'label')

    print("run time: %.3f" % (time.time() - start))



if __name__ == "__main__":
    json_file = './static_files/TMLI_config.json'

    with open(json_file, 'r') as fp:
        info = json.load(fp)
    dicom_to_hdf5(info['dicom_path'], info['npy_path'], info['annotation_list'], info['target_format'],resample=False)