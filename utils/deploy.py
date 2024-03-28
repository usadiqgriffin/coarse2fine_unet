import numpy as np
from scipy.ndimage.interpolation import affine_transform
import glob
import os
import matplotlib.pyplot as plt
import pydicom
from scipy import misc
import utils.image as im_util


# Deploy Segmentation only !!!
def deploy_seg(self):

    #dicom_path = '/mnt/SSD_1/1001553'
    #dicom_path = '/mnt/HDD_1/Studies/online_learning/JD_CBD_24_000D'
    #dicom_path = '/mnt/SSD_1/debug/original_3cv/JF_78_6K_KE_054Y/series0006-Body' # 3CV generalization issue
    dicom_path = '/mnt/SSD_1/debug/2B_Z4_NZ_ND_Bio_015Y/series0007-Body'



    target_dir = dicom_path + '/segmentation/'
    if not os.path.isdir(target_dir):
        os.system('mkdir ' + target_dir)

    self.restore_session()

    dcm_files = glob.glob(dicom_path + '/*.dcm')

    for i in range(len(dcm_files)):

        try:
            ds = dicom.read_file(dcm_files[i])
            image = ds.pixel_array
            pixel_height = ds.PixelSpacing[0]
            pixel_width = ds.PixelSpacing[1]
        except:
            continue


        # Scaling and histogram equalization
        if self.do_his_equ:
            image = im_util.his_equal(image)
        else:
            image = im_util.adjust_range(image)

        ### Cropping ###
        rotation_angle = 0
        in_shape = np.array(np.shape(image))
        out_shape = np.array([self.box_height, self.box_width])
        in_res = np.array([pixel_height, pixel_width])
        out_res = np.array([1.855, 1.855])
        mirroring = True

        transform, offset = im_util.getTransform2D(in_shape, out_shape, in_res, out_res, rotation_angle, mirroring)

        cropped_image = affine_transform(
                image,transform.T,order=2,offset=offset,output_shape=out_shape,cval=0.0,output=np.float32)

        images = np.empty([1, self.box_height, self.box_width, 1])
        images[0,:,:,0] = cropped_image

        seg_out = self.model.session.run(self.model.prediction, feed_dict={self.model.x: images, self.model.phase: 0})

        im_filename_prefix = dcm_files[i].split('/')[-1][:-4]

        misc.imsave(target_dir + im_filename_prefix + '_seg.jpg', seg_out[0])


# Deploy Classification only (open==0, closed==1, none==2) !!!
def deploy_cls(self):

    dicom_path = '/mnt/SSD_1/1001553'

    target_dir = dicom_path + '/classification/'
    if not os.path.isdir(target_dir):
        os.system('mkdir ' + target_dir)
    else:
        os.system('rm ' + target_dir + '*')

    self.restore_session()

    dcm_files = glob.glob(dicom_path + '/*.dcm')

    for i in range(len(dcm_files)):

        try:
            ds = dicom.read_file(dcm_files[i])
            data = ds.pixel_array
            pixel_height = ds.PixelSpacing[0]
            pixel_width = ds.PixelSpacing[1]
        except:
            continue

        ###### Resizing data and segmentation ######
        width_resized = int(data.shape[1] * pixel_width / 1.855)
        height_resized = int(data.shape[0] * pixel_height / 1.855)

        image = misc.imresize(data, [height_resized, width_resized], interp='nearest', mode='F')

        # Scaling and histogram equalization
        if self.do_his_equ:
            image = his_equal(image)
        else:
            image = adjust_range(image)

        ### Cropping ###
        rotation_angle = 0
        in_shape = np.array(np.shape(image))
        out_shape = np.array([self.box_height, self.box_width])
        in_res = np.array([1.855, 1.855])
        out_res = np.array([1.855, 1.855])
        mirroring = False

        transform, offset = getTransform2D(in_shape, out_shape, in_res, out_res, rotation_angle, mirroring)

        cropped_image = affine_transform(
                image,transform.T,order=2,offset=offset,output_shape=out_shape,cval=0.0,output=np.float32)

        images = np.empty([1, self.box_height, self.box_width, 1])
        images[0,:,:,0] = cropped_image

        cls_out = self.model.session.run(self.model.cls_output, feed_dict={self.model.x: images, self.model.phase: 0})

        if cls_out[0] == 0:
            misc.imsave(target_dir + dcm_files[i].split('/')[-1][:-4] + '_cls_open.jpg', cropped_image)
        elif cls_out[0] == 1:
            misc.imsave(target_dir + dcm_files[i].split('/')[-1][:-4] + '_cls_closed.jpg', cropped_image)
        elif cls_out[0] == 2:
            misc.imsave(target_dir + dcm_files[i].split('/')[-1][:-4] + '_cls_none.jpg', cropped_image)
        else:
            raise ValueError('ERROR')


# Deploy Regression only (of the LV endo/epi opening points) !!!
def deploy_reg(self):

    #dicom_path = '/mnt/SSD_1/1001553'
    #dicom_path = '/mnt/HDD_2/Kaggle/sax_1000_48'
    #dicom_path = '/mnt/HDD_2/Kaggle/sax_1001_5'
    #dicom_path = '/mnt/HDD_2/Kaggle/sax_1002_5'
    #dicom_path = '/mnt/HDD_2/Kaggle/sax_1003_5'
    #dicom_path = '/mnt/HDD_2/Kaggle/sax_1004_5'
    #dicom_path = '/mnt/HDD_2/Kaggle/sax_1005_5'
    #dicom_path = '/mnt/HDD_2/Kaggle/sax_1006_58'
    #dicom_path = '/mnt/HDD_2/Kaggle/sax_1007_9'
    #dicom_path = '/mnt/HDD_2/Kaggle/sax_1008_6'
    #dicom_path = '/mnt/HDD_2/Kaggle/sax_1009_11'
    #dicom_path = '/mnt/HDD_2/Kaggle/sax_1010_6'
    dicom_path = '/mnt/HDD_2/Kaggle/sax_1011_5'

    target_dir = dicom_path + '/regression/'
    if not os.path.isdir(target_dir):
        os.system('mkdir ' + target_dir)
    else:
        os.system('rm ' + target_dir + '*')

    self.restore_session()

    dcm_files = glob.glob(dicom_path + '/*.dcm')

    for i in range(len(dcm_files)):

        try:
            ds = dicom.read_file(dcm_files[i])
            data = ds.pixel_array
            pixel_height = ds.PixelSpacing[0]
            pixel_width = ds.PixelSpacing[1]
        except:
            continue

        ###### Resizing data and segmentation ######
        width_resized = int(data.shape[1] * pixel_width / 1.855)
        height_resized = int(data.shape[0] * pixel_height / 1.855)

        image = misc.imresize(data, [height_resized, width_resized], interp='nearest', mode='F')

        # Scaling and histogram equalization
        if self.do_his_equ:
            image = his_equal(image)
        else:
            image = adjust_range(image)

        ### Cropping ###
        rotation_angle = 0
        in_shape = np.array(np.shape(image))
        out_shape = np.array([self.box_height, self.box_width])
        in_res = np.array([1.855, 1.855])
        out_res = np.array([1.855, 1.855])
        mirroring = False

        transform, offset = getTransform2D(in_shape, out_shape, in_res, out_res, rotation_angle, mirroring)

        cropped_image = affine_transform(
                image,transform.T,order=2,offset=offset,output_shape=out_shape,cval=0.0,output=np.float32)

        images = np.empty([1, self.box_height, self.box_width, 1])
        images[0,:,:,0] = cropped_image

        reg_out = self.model.session.run(self.model.reg_output, feed_dict={self.model.x: images, self.model.phase: 0})

        point1 = reg_out[0][0:2]
        point2 = reg_out[0][2:4]
        point3 = reg_out[0][4:6]
        point4 = reg_out[0][6:8]

        point1[0] = point1[0] * self.box_height
        point1[1] = point1[1] * self.box_width
        point2[0] = point2[0] * self.box_height
        point2[1] = point2[1] * self.box_width
        point3[0] = point3[0] * self.box_height
        point3[1] = point3[1] * self.box_width
        point4[0] = point4[0] * self.box_height
        point4[1] = point4[1] * self.box_width

        #plt.subplot(1,1,1); plt.imshow(images[0,:,:,0], cmap="gray");plt.scatter(point1[1], point1[0]);plt.scatter(point2[1], point2[0]);plt.scatter(point3[1], point3[0]);plt.scatter(point4[1], point4[0])
        #plt.show()

        cropped_image[int(round(point1[0])), int(round(point1[1]))] = 0
        cropped_image[int(round(point2[0])), int(round(point2[1]))] = 0
        cropped_image[int(round(point3[0])), int(round(point3[1]))] = 0
        cropped_image[int(round(point4[0])), int(round(point4[1]))] = 0

        misc.imsave(target_dir + dcm_files[i].split('/')[-1][:-4] + '_reg.jpg', cropped_image)
