import os
import glob
import pydicom
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import re


def has_annotation(annotation, annotation_list):
    annotation_list = [re.sub(r'[\s]*','',case.lower()) for case in annotation_list]
    annotation = re.sub(r'[\s]*','',annotation.lower())
    if annotation in annotation_list:
        return True, annotation_list.index(annotation)
    else:
        return False, -1


# CT and RT in different folders
def annotation_check(input_path, save_path, annotation_list):

    info = []
    except_id = []
    except_id = []

    patient_id = os.listdir(input_path)

    for ID in tqdm(patient_id):
        print(ID)
        info_item = []
        info_item.append(ID)

        index_list = list(np.zeros((len(annotation_list), ), dtype=np.int8))

        for item in os.scandir(os.path.join(input_path,ID)):
            rt_path = glob.glob(os.path.join(item.path, '*' + ID + '*RT*'))[0]
            rt_slice = glob.glob(os.path.join(rt_path, '*.dcm'))[0]
            try:
                structure = pydicom.read_file(rt_slice)
            except:
                except_id.append(ID)
                print('RT Error:%s'%ID)
                continue    
            else:
                for i in range(len(structure.ROIContourSequence)):
                    info_item.append(structure.StructureSetROISequence[i].ROIName)
                    flag, index = has_annotation(
                        structure.StructureSetROISequence[i].ROIName, annotation_list)
                    if flag:
                        try:
                            _ = [
                                s.ContourData for s in
                                structure.ROIContourSequence[i].ContourSequence
                            ]
                        except Exception:
                            break
                        else:
                            index_list[index] = index_list[index] + 1
        if not (np.min(index_list) == 1 and np.max(index_list) == 1):
            except_id.append(ID)
            lack_list = []
            for i in range(len(annotation_list)):
                if index_list[i] != 1:
                    lack_list.append(annotation_list[i])
            print('%s without annotations:'%ID,lack_list)
           
        # info_item.sort()
        info.append(info_item)

    info_csv = pd.DataFrame(data=info)
    info_csv.to_csv(save_path, index=False, header=None)

    print(except_id)
    print(len(except_id))




if __name__ == "__main__":

    json_file = './static_files/TMLI_config.json'
    with open(json_file, 'r') as fp:
        info = json.load(fp)
    annotation_check(info['dicom_path'], info['annotation_path'],info['annotation_list'])
