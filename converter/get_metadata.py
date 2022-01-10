import os
import glob
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import json


def metadata_reader(data_path):

    info = []
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(data_path)
    reader.SetFileNames(dicom_names)
    data = reader.Execute()
    size = list(data.GetSize()[:2])
    z_size = data.GetSize()[-1]
    thick_ness = data.GetSpacing()[-1]
    pixel_spacing = list(data.GetSpacing()[:2])
    info.append(size)
    info.append(z_size)
    info.append(thick_ness)
    info.append(pixel_spacing)
    return info

# CT and RT in different folders
def get_metadata(input_path, save_path):

    id_list = os.listdir(input_path)
    info = []
    for ID in tqdm(id_list):
        info_item = [ID]
        data_path = os.path.join(input_path,ID)
        sub_dir = ['up','down']
        for sub in sub_dir:
            sub_path = os.path.join(data_path,sub)
            if os.path.exists(sub_path):
                series_path = glob.glob(os.path.join(sub_path, '*' + ID + '*CT*'))[0]
                info_item.extend(metadata_reader(series_path))
        info.append(info_item)
    col = ['id'] + ['size', 'num', 'thickness', 'pixel_spacing']*2

    info_data = pd.DataFrame(columns=col,data=info)
    info_data.to_csv(save_path, index=False, header=None)


if __name__ == "__main__":

    # json_file = './static_files/TMLI_config.json'
    json_file = './static_files/TMLI_config_v2.json'

    with open(json_file, 'r') as fp:
        info = json.load(fp)
    get_metadata(info['dicom_path'], info['metadata_path'])
