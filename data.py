import numpy as np
import os
import random
import glob
import imageio
import pandas as pd
import SimpleITK as sitk
import utils.stn3d as stn3d


class Data(object):

    def np2image(self, np_vol, ref_image, ref_origin=None, ref_spacing=None, ref_direction=None):

        image = sitk.GetImageFromArray(np_vol)

        if ref_origin is None:
            ref_origin = ref_image.GetOrigin()
        if ref_spacing is None:
            ref_spacing = ref_image.GetSpacing()
        if ref_direction is None:
            ref_direction = ref_image.GetDirection()

        image.SetOrigin(ref_origin)
        image.SetSpacing(ref_spacing)
        image.SetDirection(ref_direction)

        return image

    def load_data(self, load_mode, model, data_opts):
                
        lvo_r127_base_path = '/mnt/HDD_5/experiment_data/rotem/lvo_detection/lvo_r127/output/deploy_step_26638/'
        lvo_r129_base_path = '/mnt/HDD_5/experiment_data/rotem/lvo_detection/lvo_r129/output/deploy_step_20301/'
        
        #train_set_loc = pd.read_csv(lvo_r127_base_path + 'deploy_step_26638_load_mode_100.csv', low_memory=False)
        train_set = pd.read_csv(data_opts.data_path + data_opts.train_csv, low_memory=False)
        #assert(len(train_set_loc) == len_base_path + 'deploy_step_20301_seg_thresh_0.5_load_mode_100.csv', low_memory=False)(train_set))

        #val_set_loc = pd.read_csv(lvo_r127_base_path + 'deploy_step_26638_load_mode_101.csv', low_memory=False)
        val_set = pd.read_csv(data_opts.data_path + data_opts.val_csv, low_memory=False)
        #assert(len(val_set_loc) == len(val_set))

        #test_set_loc = pd.read_csv(lvo_r127_base_path + 'deploy_step_26638_load_mode_102.csv', low_memory=False)
        test_set = pd.read_csv(data_opts.data_path + data_opts.test_csv, low_memory=False)
        #assert(len(test_set_loc) == len(test_set))

        fda_set_loc = pd.read_csv(lvo_r127_base_path + 'deploy_step_26638_load_mode_103.csv', low_memory=False)
        fda_set = pd.read_csv(lvo_r129_base_path + 'deploy_step_20301_seg_thresh_0.5_load_mode_103.csv', low_memory=False)
        #assert(len(fda_set_loc) == len(fda_set))

        extra_test_set_loc = pd.read_csv(lvo_r127_base_path + 'deploy_step_26638_load_mode_106.csv', low_memory=False)
        extra_test_set = pd.read_csv(lvo_r129_base_path + 'deploy_step_20301_seg_thresh_0.5_load_mode_106.csv', low_memory=False)
        assert(len(extra_test_set_loc) == len(extra_test_set))
        extra_test_set = extra_test_set[1000:]  # Running on a subset due to large memory need (N=1462)

        # DEBUG
        #train_set = train_set[0:8]
        #val_set = val_set[0:8]

        if load_mode == 0:
            datasets = [train_set, val_set, []]
        elif load_mode == 1:
            datasets = [[], val_set, []]
        elif load_mode == 2:
            datasets = [[], [], test_set]
        elif load_mode == 3:
            datasets = [[], [], fda_set]
        elif load_mode == 4:
            datasets = [train_set[0:1], [], []]
        elif load_mode == 6:
            datasets = [[], [], extra_test_set]

        # Loading the appropriate data to memory
        # counter = 0
        for i in range(len(datasets)):
            for j in range(len(datasets[i])):
                '''
                counter = counter + 1
                if counter > 5:
                    break
                '''
                row = datasets[i].iloc[j, :]
                pred_lv_location_in_raw_space_path = row['pred_lv_location_in_raw_space']

                patient_name = row['patient_folder_name']
                print('Loading', patient_name)

                internal_base_path = os.path.dirname(pred_lv_location_in_raw_space_path)

                # Loading volumes
                cta0_internal_path = internal_base_path + '/cta0_internal.nii.gz'
                clot_internal_path = internal_base_path + '/clot_internal.nii.gz'
                #vessel_internal_path = internal_base_path + '/vessel_internal.nii.gz'

                if os.path.isfile(cta0_internal_path):
                    cta0_internal_itk = sitk.ReadImage(cta0_internal_path, sitk.sitkFloat32)
                    clot_internal_itk = sitk.ReadImage(clot_internal_path, sitk.sitkFloat32)
                else:
                    # Should be generated by the vessel seg model, except for exception cases that occured there
                    raw_cta_path = row['raw_cta_path']
                    target_label = row['target_label']
                    cta0_itk = sitk.ReadImage(raw_cta_path, sitk.sitkFloat32)
                    clot_itk = None
                    if target_label == 1 and load_mode not in [3, 6]:
                        raw_clot_path = row['coreg_clot']
                        assert(os.path.dirname(raw_clot_path) == os.path.dirname(raw_cta_path))
                        clot_itk = sitk.ReadImage(raw_clot_path, sitk.sitkFloat32)
                        if (clot_itk.GetSize() != cta0_itk.GetSize()):
                            print('Skipping: CTA and Clot are not the same size!!!')
                            continue
                    print('Error: the train/eval data loading of the vessel segmentation model is expected to generate these internal space volumes')
                    exit()

                # Loading volumes

                '''mcta_internal = np.zeros(shape = [96, 256, 256, 2])
                mcta_internal[:, :, :, 0] = sitk.GetArrayFromImage(cta0_internal_itk)
                mcta_internal[:, :, :, 1] = sitk.GetArrayFromImage(cta0_internal_itk)'''

                mcta_internal = sitk.GetArrayFromImage(cta0_internal_itk)
                #cta0_internal = sitk.GetArrayFromImage(cta0_internal_itk)
                #cta1_internal = sitk.GetArrayFromImage(cta0_internal_itk)
                clot_internal = sitk.GetArrayFromImage(clot_internal_itk)

                # vessel_internal_path corresponds to the reference SyNRA vessel internal, 
                # which is used to train/eval the vessel segmentation model only.
                # Here, we load pred_vessel_internal,
                # which corresponds to the output of the vessel segmentation model.
                pred_vessel_internal_path = row['pred_vessel_internal']
                pred_vessel_internal_itk = sitk.ReadImage(pred_vessel_internal_path, sitk.sitkFloat32)
                pred_vessel_internal = sitk.GetArrayFromImage(pred_vessel_internal_itk)

                example = [mcta_internal,
                           clot_internal,
                           row,
                           pred_vessel_internal]

                if i == 0:
                    self.train_set.append(example)
                elif i == 1:
                    self.val_set.append(example)
                elif i == 2:
                    self.test_set.append(example)

        print()
        print('**************************************')
        print('len(self.train_set) =', len(self.train_set))
        print('len(self.val_set) =', len(self.val_set))
        print('len(self.test_set) =', len(self.test_set))
        print('Done Loading')
        print('**************************************')
        

    def load_data_deploy(self, load_mode, model):

        print('load depoy data')


    def __init__(self, load_mode, model, data_opts):

        # Initializing the datasets
        self.train_set = []
        self.val_set = []
        self.test_set = []

        if load_mode == 100:
            self.load_data_deploy(load_mode, model)
        else:
            self.load_data(load_mode, model, data_opts)
