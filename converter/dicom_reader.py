import sys
sys.path.append('../../')
import SimpleITK as sitk
import numpy as np
import pydicom

from skimage.transform import resize
from skimage.exposure.exposure import rescale_intensity
from skimage.draw import polygon
import cv2
import re

from converter.utils import dicom_series_reader,trunc_gray,normalize,dicom_series_reader_without_postfix


# by pydicom 
class Dicom_Reader(object):
    def __init__(self,
                 series_path,
                 target_format=None,
                 rt_path=None,
                 annotation_list=None,
                 trunc_flag=False,
                 normalize_flag=False,
                 with_postfix=True):
        self.series_path = series_path
        self.target_format = target_format
        self.rt_path = rt_path
        self.annotation_list = annotation_list
        self.num_class = len(annotation_list)
        if with_postfix:
            self.meta_data, self.images = dicom_series_reader(self.series_path)
        else:
            self.meta_data, self.images = dicom_series_reader_without_postfix(self.series_path)
        self.trunc_flag = trunc_flag
        self.normalize_flag = normalize_flag

    def get_raw_images(self):
        if self.trunc_flag and self.target_format is not None:
            images = trunc_gray(self.images, self.target_format['scale'])
            if self.normalize_flag:
                return normalize(images)
            else:
                return images
        else:
            if self.normalize_flag:
                return normalize(self.images)
            else:
                return self.images

    def get_denoising_images(self):
        normal_image = trunc_gray(self.images,
                                  in_range=(-1000, 600))  #(-1000,600)
        normal_image = normalize(normal_image)
        tmp_images = self.get_raw_images()
        new_images = np.zeros_like(self.images, dtype=np.float32)
        for i in range(self.images.shape[0]):
            body = self.get_body(normal_image[i])
            new_images[i] = body * tmp_images[i]

        return new_images

    def get_resample_info(self):
        info = {}
        info['ori_shape'] = self.images.shape
        info['inplane_size'] = self.meta_data[0].Rows
        info['z_scale'] = self.meta_data[0].SliceThickness / self.target_format['thickness']
        info['z_size'] = int(np.rint(info['z_scale'] * self.images.shape[0]))

        return info
    
    # resample on depth 
    def get_resample_images(self, in_raw=True):
        info = self.get_resample_info()
        if in_raw:
            if info['inplane_size'] == self.target_format['size'][0] and info['z_scale'] == 1:
                return self.get_raw_images()
            else:
                images = self.get_raw_images()
                images = resize(images,
                                (info['z_size'], ) + tuple(self.target_format['size']),
                                mode='constant')
                return images


        else:
            if info['inplane_size'] == self.target_format['size'][0] and info['z_scale'] == 1:
                return self.get_denoising_images()
            else:
                images = self.get_denoising_images()
                images = resize(images,
                                (info['z_size'], ) + tuple(self.target_format['size']),
                                mode='constant')
                return images


    def get_raw_labels(self):
        if self.rt_path == None:
            raise ValueError("Need a RT data path!!")
        else:
            structure = pydicom.read_file(self.rt_path,force=True)
            try:
                contours = self.draw_contours(structure, self.annotation_list)
            except Exception:
                raise ValueError('Extract ROI Contours Error!')

            raw_label = self.draw_labels(self.images.shape, contours, self.meta_data, self.annotation_list)

            return raw_label


    def get_resample_labels(self):
        info = self.get_resample_info()
        if info['inplane_size'] == self.target_format['size'][0] and info['z_scale'] == 1:
            return self.get_raw_labels()
        else:
            raw_label = self.get_raw_labels()
            labels = np.zeros((info['z_size'], ) + tuple(self.target_format['size']),dtype=np.float32)
            for i in range(self.num_class):
                roi = resize((raw_label == i + 1).astype(np.float32),
                             (info['z_size'], ) +tuple(self.target_format['size']),
                             mode='constant')
                labels[roi >= 0.5] = i + 1
            return labels


    def cropping(self, array, crop):
        return array[:, crop:-crop, crop:-crop]

    def padding(self, array, pad):
        return np.pad(array, ((0, 0), (pad, pad), (pad, pad)), 'constant')

    def draw_contours(self, structure, annotation_list):
        contours = {}
        for i in range(len(structure.ROIContourSequence)):
            ROIName = re.sub(r'[\s]*','',structure.StructureSetROISequence[i].ROIName.lower())
            tmp_annotation_list = [re.sub(r'[\s]*','',case.lower()) for case in annotation_list]
            if ROIName in tmp_annotation_list:
                contour_item = {}
                contour_item['number'] = structure.ROIContourSequence[i].ReferencedROINumber
                contour_item['name'] = annotation_list[tmp_annotation_list.index(ROIName)]
                assert contour_item['number'] == structure.StructureSetROISequence[i].ROINumber
                contour_item['coord_point'] = [
                    s.ContourData
                    for s in structure.ROIContourSequence[i].ContourSequence
                ]
                contours[contour_item['name']] = contour_item

        return contours

    def draw_labels(self, shape, contours, meta_data, annotation_list):
        z = [np.around(s.ImagePositionPatient[2], 0) for s in meta_data]

        origin = np.array(meta_data[0].ImagePositionPatient[:2])
        spacing = np.array(meta_data[0].PixelSpacing)
        orient = meta_data[0].ImageOrientationPatient
        transfmat = np.array([orient[:2], orient[3:5]])
        transfmat = np.array([transfmat[0] * spacing[0], transfmat[1] * spacing[1]])
        transfmat = np.linalg.inv(transfmat)

        labels = np.zeros(shape, dtype=np.float32)
        # assert len(contours) == len(annotation_list)
        
        for annotation in annotation_list:
            try:
                con = contours[annotation]
            except:
                # print('lack:',annotation)
                continue
            # print('***** %s in Extracting ***** '%annotation)
            count = 0
            ROI_NUMBER = annotation_list.index(annotation) + 1

            if 'coord_point' not in con or len(con['coord_point']) == 0:
                print('There is no point!')
            for item in con['coord_point']:
                points = np.array(item).reshape((-1, 3))
                try:
                    assert np.amax(np.abs(np.diff(points[:, 2]))) == 0
                    z_index = z.index(np.around(points[0, 2], 0))
                except:
                    count +=1
                    print(con['name'])
                    continue
                points = points[:, :2] - origin
                points = np.array([
                    np.matmul(points[i], transfmat) for i in range(points.shape[0])
                ])
                r = points[:, 1]
                c = points[:, 0]
                rr, cc = polygon(r, c)
                labels[z_index, rr, cc] = ROI_NUMBER
            if count != 0:
                print('lack %d slices in %s'%(count,annotation))
            else:
                print('%s is done!'%(annotation))
        return labels


    def get_body(self,image):
        img = rescale_intensity(image, out_range=(0, 255))
        img = img.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        body = cv2.erode(img, kernel, iterations=1)
        kernel_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        blur = cv2.GaussianBlur(body, (5, 5), 0)
        _, body = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, kernel_1, iterations=3)
        contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        area = [[i, cv2.contourArea(contours[i])] for i in range(len(contours))]
        area.sort(key=lambda x: x[1], reverse=True)
        body = np.zeros_like(body, dtype=np.uint8)
        for i in range(min(len(area),3)):
            if area[i][1] > area[0][1] / 20:
                contour = contours[area[i][0]]
                r = contour[:, 0, 1]
                c = contour[:, 0, 0]
                rr, cc = polygon(r, c)
                body[rr, cc] = 1
        body = cv2.medianBlur(body, 5)

        return body




'''world to image coordinates 

def convertToImgCoord(xyz, origin, transfmat_toimg):
  # convert world to image coordinates
  xyz = xyz - origin
  xyz = np.round(np.matmul(transfmat_toimg, xyz))
  return xyz


def getImgWorldTransfMats(spacing, transfmat):
  # calc image to world to image transtarget_formation matrixes
  transfmat = np.array([transfmat[0:3], transfmat[3:6], transfmat[6:9]])
  for d in range(3):
    transfmat[0:3, d] = transfmat[0:3, d]*spacing[d]
  transfmat_toworld = transfmat  # image to world coordinates conversion matrix
  # world to image coordinates conversion matrix
  transfmat_toimg = np.linalg.inv(transfmat)
  return transfmat_toimg, transfmat_toworld

'''