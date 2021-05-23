import numpy as np
import h5py
import SimpleITK as sitk

import glob
import os
import pydicom



def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image


def save_as_hdf5(data, save_path, key):
    hdf5_file = h5py.File(save_path, 'a')
    hdf5_file.create_dataset(key, data=data)
    hdf5_file.close()


def save_as_nii(data, save_path):
    sitk_data = sitk.GetImageFromArray(data)
    sitk.WriteImage(sitk_data, save_path)



## dicom series reader by simpleITK
def dicom_series_readerv_v2(data_path):
  reader = sitk.ImageSeriesReader()
  dicom_names = reader.GetGDCMSeriesFileNames(data_path)
  reader.SetFileNames(dicom_names)
  data = reader.Execute()
  image_array = sitk.GetArrayFromImage(data).astype(np.float32)

  return data,image_array


'''Note
pydicom is faster than simpleITK
e.g. one sample consist of 214 slices
   - pydicom: 1.1s
   - simpleITK: 6.5s    

'''


## dicom series reader by pydicom, rt and series in different folders
def dicom_series_reader(data_path):
    dcms = glob.glob(os.path.join(data_path, '*.dcm'))
    try:
        meta_data = [pydicom.read_file(dcm) for dcm in dcms]
    except:
        meta_data = [pydicom.read_file(dcm,force=True) for dcm in dcms]
        for i in range(len(meta_data)):
            meta_data[i].file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    meta_data.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    images = np.stack([s.pixel_array for s in meta_data],axis=0).astype(np.float32)
    # pixel value transform to HU
    # images [images == -2000] = 0
    images = images * meta_data[0].RescaleSlope + meta_data[0].RescaleIntercept
    return meta_data, images


## dicom series reader by pydicom
def dicom_series_reader_without_postfix(data_path):
    dcms = glob.glob(os.path.join(data_path, 'CT*'))
    dcms = [dcm for dcm in dcms if "dir" not in dcm]
    try:
        meta_data = [pydicom.read_file(dcm) for dcm in dcms]
    except:
        meta_data = [pydicom.read_file(dcm,force=True) for dcm in dcms]
        for i in range(len(meta_data)):
            meta_data[i].file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    meta_data.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    images = np.stack([s.pixel_array for s in meta_data],axis=0).astype(np.float32)
    # pixel value transform to HU
    # images [images == -2000] = 0
    images = images * meta_data[0].RescaleSlope + meta_data[0].RescaleIntercept
    return meta_data, images


## nii.gz reader
def nii_reader(data_path):
    data = sitk.ReadImage(data_path)
    image = sitk.GetArrayFromImage(data).astype(np.float32)
    return data,image


def trunc_gray(img, in_range=(-1000, 600)):
    img = img - in_range[0]
    scale = in_range[1] - in_range[0]
    img[img < 0] = 0
    img[img > scale] = scale

    return img
    

def normalize(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    return img

if __name__ == "__main__":
    data_path = '/staff/shijun/dataset/TMLI/raw_data/181552/up/181552_CT'
    meta_data,image = dicom_series_readerv_v2(data_path)
    print(meta_data)
    meta_data,image = dicom_series_reader(data_path)
    print(image.shape)
    print(meta_data[0].SliceThickness)
    print(meta_data[0].ImageOrientationPatient)
    print(meta_data[0])
    