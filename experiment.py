import numpy as np
import tensorflow as tf
import os
import random
import utils.stn3d as stn3d
from hypertester.hypertest import Hypertest
import SimpleITK as sitk
import pandas as pd
import scipy.ndimage as ndimage

class Experiment(object):

    def __init__(self, data, model, output_path):

        self.data = data
        self.model = model
        self.output_path = output_path


    def to_uint8(self, image):
        image = image * 255
        image = image.astype('uint8') 
        return image

    
    def save_nii_to_file(self, image, pixdim, save_path, file_name):
        if not os.path.isdir(save_path):
            os.system('mkdir -p ' + save_path)
        itk_image = sitk.GetImageFromArray(image)
        #itk_image.SetOrigin(ref_image.GetOrigin())
        itk_image.SetDirection(np.array([1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]))
        itk_image.SetSpacing(np.array([pixdim[2], pixdim[1], pixdim[0]]))
        sitk.WriteImage(itk_image, save_path + file_name + '.nii.gz')


    def debug_input(self, augment):

        internal_shape = self.model.internal_shape
        internal_pixdim = self.model.internal_pixdim
        batch_size = 2

        image = np.zeros(shape=[batch_size, internal_shape[0], internal_shape[1], internal_shape[2]])
        clot_mask = np.zeros(shape=[batch_size, internal_shape[0], internal_shape[1], internal_shape[2]])
        trans_mat = np.zeros(shape=[batch_size, 3, 4])

        image[0] = self.data.train_set[0][0]
        clot_mask[0] = self.data.train_set[0][1]

        # Augmentation
        if augment:
            flip_z = False
            flip_y = False
            flip_x = False

            # Degrees
            rotate_z = 0
            rotate_y = 0
            rotate_x = 0

            # stn_units = voxels * spacing * 2
            print(internal_shape)
            print(internal_pixdim)
            factor = 8  # DEBUG
            translate_Z = (internal_shape[0] / factor) * internal_pixdim[0] * 2
            translate_y = 0
            translate_x = 0
        else:
            flip_z = False
            flip_y = False
            flip_x = False
            rotate_z = 0
            rotate_y = 0
            rotate_x = 0
            translate_Z = 0
            translate_y = 0
            translate_x = 0 

        trans_mat[0] = stn3d.get_transform(
            in_dims=internal_shape,
            in_res=internal_pixdim, 
            out_dims=internal_shape,
            out_res=internal_pixdim, 
            flip=[flip_z, flip_y, flip_x], 
            rotate=[rotate_z, rotate_y, rotate_x],
            translate=[translate_Z, translate_y, translate_x])
            
        # Running augmentation
        image_trans, clot_mask_trans = self.model.sess.run(
            [self.model.image_trans,
             self.model.clot_mask_trans],
            feed_dict={
             self.model.image: image,
             self.model.clot_mask: clot_mask,
             self.model.trans_mat: trans_mat})

        target_path = self.output_path + '/debug_input/'            
        self.save_nii_to_file(image_trans[0, :, :, :, 0], internal_pixdim, target_path, 'image_trans_' + str(augment))
        self.save_nii_to_file(clot_mask_trans[0], internal_pixdim, target_path, 'clot_mask_trans_' + str(augment))


    def train(self):

        epochs = 200
        save_model_epoch = 20
        batch_size = 3
        internal_shape = self.model.internal_shape
        internal_pixdim = self.model.internal_pixdim

        train_batches =  int(np.ceil(len(self.data.train_set) / batch_size))
        val_batches = int(np.ceil(len(self.data.val_set) / batch_size))

        print('\n**************************************')
        print('len(self.data.train_set) =', len(self.data.train_set))
        print('train_batches =', train_batches)
        print('len(self.data.val_set) =', len(self.data.val_set))
        print('val_batches =', val_batches)
        print('**************************************\n')

        for epoch_index in range(epochs):

            # Training
            random.shuffle(self.data.train_set)

            for batch_index in range(train_batches):

                self.model.global_step = self.model.global_step + 1

                start_index = np.minimum(batch_index * batch_size, len(self.data.train_set) - batch_size)

                image = np.zeros(shape=[batch_size, internal_shape[0], internal_shape[1], internal_shape[2]])
                clot_mask = np.zeros(shape=[batch_size, internal_shape[0], internal_shape[1], internal_shape[2]])
                trans_mat = np.zeros(shape=[batch_size, 3, 4])

                for i in range(batch_size):
                    image[i] = self.data.train_set[start_index + i][0]
                    clot_mask[i] = self.data.train_set[start_index + i][1]

                    # Augmentation
                    flip_z = np.random.random() > 0.5
                    flip_y = np.random.random() > 0.5
                    flip_x = np.random.random() > 0.5

                    # Degrees
                    rotate_z = np.random.random() * 360
                    rotate_y = (np.random.random() * 40) - 20
                    rotate_x = (np.random.random() * 40) - 20

                    # stn_units = voxels * spacing * 2
                    factor = 16  # 1/factor of the frame in all axes
                    translate_z_units = (internal_shape[0] / factor) * internal_pixdim[0] * 2
                    translate_y_units = (internal_shape[1] / factor) * internal_pixdim[1] * 2
                    translate_x_units = (internal_shape[2] / factor) * internal_pixdim[2] * 2
                    translate_z = (np.random.random() * translate_z_units * 2) - translate_z_units
                    translate_y = (np.random.random() * translate_y_units * 2) - translate_y_units
                    translate_x = (np.random.random() * translate_x_units * 2) - translate_x_units

                    trans_mat[i] = stn3d.get_transform(
                        in_dims=internal_shape,
                        in_res=internal_pixdim, 
                        out_dims=internal_shape,
                        out_res=internal_pixdim, 
                        flip=[flip_z, flip_y, flip_x], 
                        rotate=[rotate_z, rotate_y, rotate_x],
                        translate=[translate_z, translate_y, translate_x])
                    
                # Training
                _, loss, summary_loss, summary_loss0 = self.model.sess.run(
                    [self.model.optimizer, 
                     self.model.loss,
                     self.model.loss_summary_node,
                     self.model.loss0_summary_node],
                    feed_dict={
                        self.model.image: image,
                        self.model.clot_mask: clot_mask,
                        self.model.trans_mat: trans_mat,
                        self.model.training: 1})

                print('Epoch: ' + str(epoch_index) + '/' + str(epochs) + ', Batch: ' + str(batch_index) + '/' + str(train_batches) + ', Train Loss: ' + str(loss))

                # Tensorboard
                if batch_index % 10 == 0:
                    self.model.summary_writer_train.add_summary(summary_loss, self.model.global_step)
                    self.model.summary_writer_train.add_summary(summary_loss0, self.model.global_step)
                    self.model.summary_writer_train.flush()

            # Saving the model to file
            if (epoch_index % save_model_epoch == 0) or (epoch_index == epochs - 1):
                print('Saving model for epoch ' + str(epoch_index) + ' / global step ' + str(self.model.global_step))
                self.model.saver.save(self.model.sess, self.output_path + '/models/model', self.model.global_step)
                self.eval(load_mode=1)  # Evaluating the model

            # Validation
            if (epoch_index % 1 == 0) or (epoch_index == epochs - 1):

                print('Validating Epoch ' + str(epoch_index) + '/' + str(epochs) + ' ...')
                avg_loss = 0

                for batch_index in range(val_batches):

                    start_index = np.minimum(batch_index * batch_size, len(self.data.val_set) - batch_size)

                    image = np.zeros(shape=[batch_size, internal_shape[0], internal_shape[1], internal_shape[2]])
                    clot_mask = np.zeros(shape=[batch_size, internal_shape[0], internal_shape[1], internal_shape[2]])
                    trans_mat = np.zeros(shape=[batch_size, 3, 4])

                    for i in range(batch_size):
                        image[i] = self.data.val_set[start_index + i][0]
                        clot_mask[i] = self.data.val_set[start_index + i][1]
                        
                        trans_mat[i] = stn3d.get_transform(
                            in_dims=internal_shape,
                            in_res=internal_pixdim,
                            out_dims=internal_shape,
                            out_res=internal_pixdim)

                    # Validation
                    loss, summary_loss, summary_loss0 = self.model.sess.run(
                        [self.model.loss,
                         self.model.loss_summary_node,
                         self.model.loss0_summary_node],
                        feed_dict={
                            self.model.image: image,
                            self.model.clot_mask: clot_mask,
                            self.model.trans_mat: trans_mat,
                            self.model.training: 0})

                    avg_loss = avg_loss + loss

                    # Tensorboard
                    self.model.summary_writer_val.add_summary(summary_loss, self.model.global_step + batch_index)
                    self.model.summary_writer_val.add_summary(summary_loss0, self.model.global_step + batch_index)


                avg_loss = avg_loss / val_batches
                print('Epoch: ' + str(epoch_index) + '/' + str(epochs) + ', Validation Loss: ' + str(avg_loss))


    def export_to_file(self, target_path, image_trans, clot_mask_trans, pred, pred_vessel):
        internal_pixdim = self.model.internal_pixdim
        self.save_nii_to_file(image_trans[0, :, :, :, 0], internal_pixdim, target_path, 'image_trans')
        self.save_nii_to_file(clot_mask_trans[0], internal_pixdim, target_path, 'clot_mask_trans')
        self.save_nii_to_file(pred[0], internal_pixdim, target_path, 'pred')
        self.save_nii_to_file(pred_vessel, internal_pixdim, target_path, 'pred_vessel')

    
    def calculate_min_voxel_distance(self, vessel, clot):

        growing_clot = np.copy(clot)

        distance = 0
        while True:
            if np.sum(vessel * growing_clot) > 0:
                break
            else:
                distance = distance + 1
                struct = ndimage.generate_binary_structure(3, 3)
                growing_clot = ndimage.binary_dilation(growing_clot, structure=struct).astype(growing_clot.dtype)
        
        return distance


    def eval_clot_mask(self, model_outputs, eval_file_path):

        dice3d_list = []
        
        for i in range(len(model_outputs)):

            clot_mask_trans = model_outputs[i][1].astype('uint8')
            pred_clot = model_outputs[i][2].astype('uint8')
            row = model_outputs[i][3]
            label = int(row['target_label'])

            if label == 0:
                continue
            
            # Computing 3D Dice for the current volume
            nominator = 2 * np.sum(clot_mask_trans * pred_clot)
            denominator = np.sum(clot_mask_trans) + np.sum(pred_clot)
            dice = float(nominator / denominator)
            if denominator != 0:
                dice3d_list.append(dice)
            else:
                print('denominator == 0')
                continue

        mean_dice3d = round(np.mean(dice3d_list), 2)
        std_dice3d = round(np.std(dice3d_list), 2)
        median_dice3d = round(np.median(dice3d_list), 2)
        hit_ratio = round(100 * (np.sum(np.array(dice3d_list) > 0) / float(len(dice3d_list))), 2)

        with open(eval_file_path, 'w') as eval_file:
            eval_file.write('global_step = ' + str(self.model.global_step))
            eval_file.write('\nlen(dice3d_list) = ' + str(len(dice3d_list)))
            eval_file.write('\nmean_dice3d = ' + str(mean_dice3d))
            eval_file.write('\nstd_dice3d = ' + str(std_dice3d))
            eval_file.write('\nmedian_dice3d = ' + str(median_dice3d))
            eval_file.write('\nhit_ratio = ' + str(hit_ratio))
        

    def eval_lvo_classifier(self, model_outputs, eval_file_path):

        export_to_file = True

        row0 = model_outputs[0][3]
        output_df = pd.DataFrame(columns=row0.index.values)

        eval_dir = os.path.dirname(eval_file_path)

        # Initializing pred_prob_arr and label_bin_arr for the ROC analysis
        pred_prob_arr = []
        label_bin_arr = []
        for i in range(len(model_outputs)):

            row = model_outputs[i][3]
            pred_prob = model_outputs[i][5][0]

            label = int(row['target_label'])
            label_bin = bool(label > 0)

            # Used for ROC analysis
            pred_prob_arr.append(pred_prob)
            label_bin_arr.append(label_bin)

        # ROC Analysis
        classification0_val = Hypertest(y=pred_prob_arr,
                                        t=label_bin_arr,
                                        name='binary_classification_val_roc',
                                        task='classification',
                                        results_folder=eval_dir,
                                        experiment_list=['roc_analysis']).run()
        classification0_val.report()
        classification0_val.save()

        auc = classification0_val.get_result('auc')
        sen_auc = classification0_val.get_result('sen_selected').point_estimate
        spe_auc = classification0_val.get_result('spe_selected').point_estimate
        thresh_auc = classification0_val.get_result('thres_selected')

        # Calculating sensitivity/specificity with fixed threshold
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        tp_counter = 0
        tn_counter = 0
        pred_bin_arr = []
        pred_clot_cubic_mm_arr = []
        for i in range(len(model_outputs)):

            image_trans = model_outputs[i][0]
            clot_mask_trans = model_outputs[i][1].astype('uint8')
            pred_clot = model_outputs[i][2].astype('uint8')
            row = model_outputs[i][3]
            pred_vessel = model_outputs[i][4].astype('uint8')
            pred_bin = bool(int((model_outputs[i][6][0])))
            pred_clot_cubic_mm = model_outputs[i][7][0]

            patient_name = row['patient_folder_name']
            output_df = output_df.append(row, ignore_index=True)

            label_bin = label_bin_arr[i]

            pred_bin_arr.append(pred_bin)
            pred_clot_cubic_mm_arr.append(pred_clot_cubic_mm)

            if pred_bin==True and label_bin==True:
                tp = tp + 1
                if tp_counter < 5:
                    target_path = eval_dir + '/examples/TP/' + patient_name + '/'
                    if export_to_file:
                        self.export_to_file(target_path, image_trans, clot_mask_trans, pred_clot, pred_vessel)
                    tp_counter = tp_counter + 1
            elif pred_bin==False and label_bin==False:
                tn = tn + 1
                if tn_counter < 5:
                    target_path = eval_dir + '/examples/TN/' + patient_name + '/'
                    if export_to_file:
                        self.export_to_file(target_path, image_trans, clot_mask_trans, pred_clot, pred_vessel)
                    tn_counter = tn_counter + 1
            elif pred_bin==True and label_bin==False:
                fp = fp + 1
                target_path = eval_dir + '/examples/FP/' + patient_name + '/'
                if export_to_file:
                    self.export_to_file(target_path, image_trans, clot_mask_trans, pred_clot, pred_vessel)
            elif pred_bin==False and label_bin==True:
                fn = fn + 1
                target_path = eval_dir + '/examples/FN/' + patient_name + '/'
                if export_to_file:
                    self.export_to_file(target_path, image_trans, clot_mask_trans, pred_clot, pred_vessel)

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        with open(eval_file_path, 'a') as eval_file:
            eval_file.write('\n\nglobal_step = ' + str(self.model.global_step))
            eval_file.write('\nlen(label_bin_arr) = ' + str(len(label_bin_arr)))
            eval_file.write('\nauc = ' + str(auc))
            eval_file.write('\nsen_auc = ' + str(sen_auc))
            eval_file.write('\nspe_auc = ' + str(spe_auc))
            eval_file.write('\nthresh_auc = ' + str(thresh_auc))
            eval_file.write('\nthresh_auc_mm^3 = ' + str(thresh_auc * self.model.image_cubic_mm))
            eval_file.write('\ndist_thresh_dilate = ' + str(self.model.dist_thresh_dilate))

            eval_file.write('\n\naccuracy_fixed = ' + str(accuracy))
            eval_file.write('\nsensitivity_fixed = ' + str(sensitivity))
            eval_file.write('\nspecificity_fixed = ' + str(specificity))
            eval_file.write('\nthresh_fixed = ' + str(self.model.pred_thresh_mm / self.model.image_cubic_mm))
            eval_file.write('\nthresh_fixed_mm^3 = ' + str(self.model.pred_thresh_mm))
            eval_file.write('\ndist_thresh_dilate = ' + str(self.model.dist_thresh_dilate))

        # Updating the output spreadsheet
        output_df['pred_prob'] = pred_prob_arr
        output_df['pred_clot_cubic_mm'] = pred_clot_cubic_mm_arr
        output_df['pred_bin'] = pred_bin_arr
        output_df['label_bin'] = label_bin_arr
        output_df.to_csv(eval_dir + '/eval_output.csv')


    def eval(self, load_mode):
        
        if load_mode == 1:
            # Forcing the model to run on the validation set only
            datasets = [[], self.data.val_set, []]
        else:
            # Running the model on all the train/val/test set, depending on what is loaded to memory (i.e. load_mode)
            datasets = [self.data.train_set, self.data.val_set, self.data.test_set]

        model_outputs = []  # will contain everything needed to run the test and generate the report
        loss0_arr = []

        for dataset in datasets:

            for i in range(len(dataset)):

                print(i, '/', len(dataset))

                # Validating the validation set
                image = np.expand_dims(dataset[i][0], 0)
                clot_mask = np.expand_dims(dataset[i][1], 0)
                row = dataset[i][2]
                pred_vessel_internal = dataset[i][3]

                '''
                # SyNRA failed in the below cases.
                # lvo_r124 fixed 6/9 cases.
                # The remaining 3 cases are:
                # ESCAPENA1_73-005: significant amount of missing slices at the top of the head
                # PREDICT_02-002: ROI is actually fine. The volume is smoothed/MIPed.
                # PREDICT_09-040: significant number of missing slices at the top of the head
                if patient_name in [
                    '1131-5477',
                    'ESCAPENA1_20-024',
                    'ESCAPENA1_73-005',
                    'ESCAPENA1_73-012',
                    'PREDICT_02-002',
                    'PREDICT_09-005',
                    'PREDICT_09-040',
                    'PREDICT_011-049',
                    'TEMPO_01_026']:
                    continue
                '''

                trans_mat = np.expand_dims(stn3d.get_transform(
                    in_dims=self.model.internal_shape,
                    in_res=self.model.internal_pixdim,
                    out_dims=self.model.internal_shape,
                    out_res=self.model.internal_pixdim), 0)

                image_trans, clot_mask_trans, pred_clot, loss0 = self.model.sess.run(
                    [self.model.image_trans,
                     self.model.clot_mask_trans,
                     self.model.pred_clot,
                     self.model.loss0],
                    feed_dict={
                        self.model.image: image,
                        self.model.clot_mask: clot_mask,
                        self.model.vessel_mask_trans: np.expand_dims(pred_vessel_internal, 0),
                        self.model.trans_mat: trans_mat,
                        self.model.training: 0})

                model_outputs.append([image_trans,
                                      clot_mask_trans,
                                      pred_clot,
                                      row,
                                      pred_vessel_internal])
                loss0_arr.append(loss0)

        '''
        # Writing to file
        eval_dir = self.output_path + '/eval/eval_step_' + str(self.model.global_step) + '_' + str(self.model.seg_thresh) + \
            '_' + str(self.model.pred_thresh_mm) + '_' + str(self.model.dist_thresh_dilate) + '_' + str(load_mode) + '/'
        if not os.path.isdir(eval_dir):
            os.system('mkdir -p ' + eval_dir)
        eval_file_path = eval_dir + 'eval_output.txt'        

        # Evaluating the clot mask
        if load_mode < 3:
            self.eval_clot_mask(model_outputs, eval_file_path)

        # Evaluating the LVO classifier
        self.eval_lvo_classifier(model_outputs, eval_file_path)

        with open(eval_file_path, 'a') as eval_file:
            eval_file.write('\n\nlen(model_outputs) = ' + str(len(model_outputs)))
            eval_file.write('\nnp.mean(loss0_arr) = ' + str(np.mean(loss0_arr)))
            eval_file.write('\nseg_thresh = ' + str(self.model.seg_thresh))
            eval_file.write('\nload_mode = ' + str(load_mode))
        
        with open(eval_file_path, 'r') as eval_file:
            for line in eval_file:
                print(line.replace('\n', ''))

        '''


    def deploy(self, save_nifti):
        
        print('Deploy model')