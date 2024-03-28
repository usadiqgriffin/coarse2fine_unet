import tensorflow as tf
import time
import numpy as np
import glob
import os

import matplotlib.pyplot as plt

import utils.image as im_util
import utils.model as model_util
import subprocess
from scipy.misc import imsave



def eval_post_seg(self):
    """Evaluate the 3D mask correction experiment."""

    ###########################
    # DEBUG
    #self.val_study_paths = self.val_study_paths[0:2]
    #self.val_study_paths = ['/mnt/SSD_1/SaxData_debug_2/3448454/']
    ###########################

    print('---------------------- Evaluation for BioBank validation dataset ----------------------')
    print('Dataset:', self.dataset)

    start_eval = time.time()

    model_util.restore_session(self.checkpoint_dir, self.model.saver, self.model.session)

    iou = [] # aka Jaccard Index
    dice = []

    # Per contour stats
    iou_endo = []
    dice_endo = []

    iou_epi = []
    dice_epi = []

    iou_rv_endo = []
    dice_rv_endo = []


    # For each of the 3D images in the validation set:
    for d in range(len(self.val_study_paths)):

        print('Evaluating study ', d, '/', len(self.val_study_paths))

        study_image_paths = glob.glob(self.val_study_paths[d] + '*_seg3d.npy')

        for im in range(len(study_image_paths)):

            seg3d_mutated = np.load(study_image_paths[im])
            seg3d_gt = np.load(study_image_paths[im])
            header = np.load(study_image_paths[im][:-10] + '_header.npy').tolist()

            src_shape = seg3d_mutated.shape

            ### Scaling and Cropping ###
            shape_in = np.array([src_shape[1], src_shape[2]])
            res_in = np.array([float(header['pixel_height']), float(header['pixel_width'])])
            res_out = np.array([1.855, 1.855])
            rotate_degree = 0
            mirror = False
            _, _, trans_vec, canvas_shape = im_util.get_transform2d(shape_in, res_in, res_out, rotate_degree, mirror)

            # making sure seg3d_x is of shape [self.crop_shape[0], None, None]
            # I wanted to do this in the network but tf.pad doesn't except non-static value
            # and src_depth is not static
            target_depth = self.crop_shape[0]
            seg3d_mutated = im_util.adjust_depth(seg3d_mutated, target_depth)
            seg3d_gt = im_util.adjust_depth(seg3d_gt, target_depth)

            # Here, images and segs are ready
            preds, segs = self.model.session.run(
                                [self.model.prediction, self.model.y_cropped],
                                feed_dict={
                                self.model.x: seg3d_mutated,
                                self.model.y: seg3d_gt,
                                self.model.train: 0,
                                self.model.preprocess: 1,
                                self.model.canvas_shape: canvas_shape,
                                self.model.trans: trans_vec})

            '''
            ### DEBUG: Making sure the post processing node can be evaluated ###
            _, _, trans_vec_post, canvas_shape_post = im_util.get_transform2d(self.crop_shape[1:], res_out, res_in, rotate_degree, mirror)

            post = self.model.session.run(
                                self.model.seg_pred_final,
                                feed_dict={
                                self.model.x: seg3d_mutated,
                                self.model.y: seg3d_gt,
                                self.model.train: 0,
                                self.model.preprocess: 1,
                                self.model.src_shape: shape_in,
                                self.model.canvas_shape: canvas_shape,
                                self.model.trans: trans_vec,
                                self.model.canvas_shape_post: canvas_shape_post,
                                self.model.trans_post: trans_vec_post})
            #########################################################################3
            '''

            # removing the batch dimension
            preds = preds[0]
            segs = segs[0]

            # Computing iou and dice for the entire batch
            for i in range(target_depth):

                #plt.subplot(1,2,1); plt.imshow(preds[i], cmap="gray")
                #plt.subplot(1,2,2); plt.imshow(segs[i], cmap="gray")
                #plt.show()

                # Lv Endo
                seg_endo = np.copy(segs[i])
                seg_endo[seg_endo != 1] = 0

                pred_endo = np.copy(preds[i])
                pred_endo[pred_endo != 1] = 0

                seg_endo = np.clip(seg_endo, 0, 1)
                pred_endo = np.clip(pred_endo, 0, 1)

                inter_endo = np.count_nonzero(pred_endo * seg_endo)
                pred_endo_size = np.count_nonzero(pred_endo)
                seg_endo_size = np.count_nonzero(seg_endo)
                union_endo = pred_endo_size + seg_endo_size - inter_endo

                # Lv Epi
                seg_epi = np.copy(segs[i])
                seg_epi[seg_epi != 2] = 0

                pred_epi = np.copy(preds[i])
                pred_epi[pred_epi != 2] = 0

                seg_epi = np.clip(seg_epi, 0, 1)
                pred_epi = np.clip(pred_epi, 0, 1)

                inter_epi = np.count_nonzero(pred_epi * seg_epi)
                pred_epi_size = np.count_nonzero(pred_epi)
                seg_epi_size = np.count_nonzero(seg_epi)
                union_epi = pred_epi_size + seg_epi_size - inter_epi

                # Rv Endo
                seg_rv_endo = np.copy(segs[i])
                seg_rv_endo[seg_rv_endo != 3] = 0

                pred_rv_endo = np.copy(preds[i])
                pred_rv_endo[pred_rv_endo != 3] = 0

                seg_rv_endo = np.clip(seg_rv_endo, 0, 1)
                pred_rv_endo = np.clip(pred_rv_endo, 0, 1)

                inter_rv_endo = np.count_nonzero(pred_rv_endo * seg_rv_endo)
                pred_rv_endo_size = np.count_nonzero(pred_rv_endo)
                seg_rv_endo_size = np.count_nonzero(seg_rv_endo)
                union_rv_endo = pred_rv_endo_size + seg_rv_endo_size - inter_rv_endo

                # Total intersection/union
                intersection = inter_endo + inter_epi + inter_rv_endo
                union = np.count_nonzero(segs[i] + preds[i])

                if union_endo > 0:
                    iou_endo.append(float(inter_endo)/float(union_endo))
                    dice_endo.append(2*float(inter_endo)/float(union_endo+inter_endo))

                if union_epi > 0:
                    iou_epi.append(float(inter_epi)/float(union_epi))
                    dice_epi.append(2*float(inter_epi)/float(union_epi+inter_epi))

                if union_rv_endo > 0:
                    iou_rv_endo.append(float(inter_rv_endo)/float(union_rv_endo))
                    dice_rv_endo.append(2*float(inter_rv_endo)/float(union_rv_endo+inter_rv_endo))

                if union > 0:
                    iou.append(float(intersection)/float(union))
                    dice.append(2*float(intersection)/float(union+intersection))


                # Writing to TensorBoard
                if im % 100 == 0:
                    node_images = tf.summary.image('val_batch_' + str(im) + '_seg3d_mutated',
                                tf.cast(tf.expand_dims(self.model.x, -1), tf.float32),
                                max_outputs=16)
                    node_segs = tf.summary.image('val_batch_' + str(im) + '_seg3d_gt',
                                tf.cast(tf.expand_dims(self.model.y, -1), tf.float32),
                                max_outputs=16)
                    node_preds = tf.summary.image('val_batch_' + str(im) + '_preds',
                                tf.cast(tf.expand_dims(tf.squeeze(self.model.prediction, axis=0), -1), tf.float32),
                                max_outputs=16)
                    summary_images, summary_segs, summary_preds = self.model.session.run(
                                [node_images, node_segs, node_preds],
                                feed_dict={
                                self.model.x: seg3d_mutated,
                                self.model.y: seg3d_gt,
                                self.model.train: 0,
                                self.model.preprocess: 1,
                                self.model.canvas_shape: canvas_shape,
                                self.model.trans: trans_vec})

                    self.val_writer.add_summary(summary_images, im)
                    self.val_writer.add_summary(summary_segs, im)
                    self.val_writer.add_summary(summary_preds, im)
                    self.val_writer.flush()


    out_file = open(self.checkpoint_dir + 'eval_' + self.dataset + '/eval_output.txt', 'w')

    print()
    print('total_3D_masks =', len(iou))
    out_file.write("\n%s %s\n" % ('total_3D_masks =', len(iou)))

    print()
    print('Mean Jaccard Index (IoU) =', np.mean(iou))
    print('Std Jaccard Index (IoU) =', np.std(iou))
    print('Min Jaccard Index (IoU) =', np.min(iou))
    print('Max Jaccard Index (IoU) =', np.max(iou))
    out_file.write("\n%s %s\n" % ('Mean Jaccard Index (IoU) =', np.mean(iou)))
    out_file.write("%s %s\n" % ('Std Jaccard Index (IoU) =', np.std(iou)))
    out_file.write("%s %s\n" % ('Min Jaccard Index (IoU) =', np.min(iou)))
    out_file.write("%s %s\n" % ('Max Jaccard Index (IoU) =', np.max(iou)))

    '''
    # Rotation_Analysis
    self.f_iou_mean.write(str(np.mean(iou)) + '\n')
    self.f_iou_std.write(str(np.std(iou)) + '\n')
    '''

    print()
    print('Mean Dice Index =', np.mean(dice))
    print('Std Dice Index =', np.std(dice))
    print('Min Dice Index =', np.min(dice))
    print('Max Dice Index =', np.max(dice))
    out_file.write("\n%s %s\n" % ('Mean Dice Index =', np.mean(dice)))
    out_file.write("%s %s\n" % ('Std Dice Index =', np.std(dice)))
    out_file.write("%s %s\n" % ('Min Dice Index =', np.min(dice)))
    out_file.write("%s %s\n" % ('Max Dice Index =', np.max(dice)))


    print()
    print('Mean iou_endo =', np.mean(iou_endo))
    print('Std iou_endo =', np.std(iou_endo))
    print('Mean dice_endo =', np.mean(dice_endo))
    print('Std dice_endo =', np.std(dice_endo))
    out_file.write("\n%s %s\n" % ('Mean iou_endo =', np.mean(iou_endo)))
    out_file.write("%s %s\n" % ('Std iou_endo =', np.std(iou_endo)))
    out_file.write("%s %s\n" % ('Mean dice_endo =', np.mean(dice_endo)))
    out_file.write("%s %s\n" % ('Std dice_endo =', np.std(dice_endo)))

    print()
    print('Mean iou_epi =', np.mean(iou_epi))
    print('Std iou_epi =', np.std(iou_epi))
    print('Mean dice_epi =', np.mean(dice_epi))
    print('Std dice_epi =', np.std(dice_epi))
    out_file.write("\n%s %s\n" % ('Mean iou_epi =', np.mean(iou_epi)))
    out_file.write("%s %s\n" % ('Std iou_epi =', np.std(iou_epi)))
    out_file.write("%s %s\n" % ('Mean dice_epi =', np.mean(dice_epi)))
    out_file.write("%s %s\n" % ('Std dice_epi =', np.std(dice_epi)))

    print()
    print('Mean iou_rv_endo =', np.mean(iou_rv_endo))
    print('Std iou_rv_endo =', np.std(iou_rv_endo))
    print('Mean dice_rv_endo =', np.mean(dice_rv_endo))
    print('Std dice_rv_endo =', np.std(dice_rv_endo))
    out_file.write("\n%s %s\n" % ('Mean iou_rv_endo =', np.mean(iou_rv_endo)))
    out_file.write("%s %s\n" % ('Std iou_rv_endo =', np.std(iou_rv_endo)))
    out_file.write("%s %s\n" % ('Mean dice_rv_endo =', np.mean(dice_rv_endo)))
    out_file.write("%s %s\n" % ('Std dice_rv_endo =', + np.std(dice_rv_endo)))

    print()
    print('Validation took:', (time.time() - start_eval)/60, 'minutes')
    out_file.write("\n%s %s %s\n" % ('Validation took:', (time.time() - start_eval)/60, 'minutes'))

    out_file.close()

    print()
    print('Summary:')
    print('Mean Dice Index =', np.mean(dice))
    print('Mean dice_endo =', np.mean(dice_endo))
    print('Mean dice_epi =', np.mean(dice_epi))
    print('Mean dice_rv_endo =', np.mean(dice_rv_endo))

    self.val_writer.close()

def eval_seg_reg(self, contour_names, reg, disabled_reg_indices = [], task_id = 0):
    """Evaluate segmentation on all contours in contour_names.

    Evaluate regression values, if reg is True
    Can handle any number of contours as long as len(contour_names) == number_of_labels - 1
    Can handle any segmentation task (e.g. SAX and LAX)
    task_id == 0: Seg, Reg, Seg+Reg, Cls; task_id == 1: transfer learning Seg+Reg
    """

    ###########################
    # DEBUG
    #self.val_study_paths = self.val_study_paths[0:2]
    #self.val_study_paths = ['/mnt/SSD_1/SaxData_debug_2/3448454/']
    ###########################

    # Adjusting for the variable names in the transfer learning experiment
    if task_id == 1:
        self.reg_out_size = self.reg_size2
        self.model.session = self.model.sess
        self.model.saver = self.model.train_saver
        self.model.y_reg_cropped = self.model.y_reg1_cropped
        self.model.y_cropped = self.model.y_seg_cropped
        self.model.y = self.model.y_seg
        self.model.y_reg = self.model.y_reg1

    print('---------------------- Evaluation Start ----------------------')
    print('Dataset:', self.dataset)

    start_eval = time.time()

    # Making sure the model is restored
    model_util.restore_session(self.checkpoint_dir, self.model.saver, self.model.session)

    norm_pixel_height = 1.855
    norm_pixel_width = 1.855

    iou = [] # aka Jaccard Index
    dice = []
    if reg:
        reg_error_mm = []

    iou_per_contour = []
    dice_per_contour = []

    for i in range(len(contour_names)):
        iou_per_contour.append([])
        dice_per_contour.append([])

    batch_index = 0
    num_of_batches = 0
    images = np.empty([self.batch_size, self.crop_shape[0], self.crop_shape[1], 1])
    segs = np.empty([self.batch_size, self.crop_shape[0], self.crop_shape[1]], dtype = np.int64)
    if reg:
        landmarks_arr = np.empty([self.batch_size, self.reg_out_size])

    # For each of the images in the validation set:
    for d in range(len(self.val_study_paths)):

        print('Evaluating study ', d, '/', len(self.val_study_paths))

        study_image_paths = glob.glob(self.val_study_paths[d] + '*_image.npy')

        for im in range(len(study_image_paths)):

            image = np.load(study_image_paths[im])

            header = np.load(study_image_paths[im][:-10] + '_header.npy').tolist()

            seg_path = study_image_paths[im][:-10] + '_seg.npy'
            if os.path.isfile(seg_path):
                seg = np.load(seg_path)
            else: # Ignoring non-contoured images
                continue

            ### Regression values ###
            if reg:
                landmarks = np.array(np.load(study_image_paths[im][:-10] + '_landmarks.npy'))

            ### Scaling and Cropping###
            shape_in = np.array(np.shape(image))
            res_in = np.array([float(header['pixel_height']), float(header['pixel_width'])])
            res_out = np.array([norm_pixel_height, norm_pixel_width])
            rotate_degree = 0
            mirror = False
            transform, offset, trans_vec, canvas_shape = im_util.get_transform2d(shape_in, res_in, res_out, rotate_degree, mirror)

            if task_id == 0:
                if not reg:
                    # Preprocessing each image separately (different shapes)
                    cropped_image, cropped_seg = self.model.session.run(
                        [self.model.x_cropped, self.model.y_cropped],
                        feed_dict={
                        self.model.x: np.reshape(image, [1, shape_in[0], shape_in[1], 1]),
                        self.model.y: np.reshape(seg, [1, shape_in[0], shape_in[1]]),
                        self.model.preprocess: 1,
                        self.model.src_shape: shape_in,
                        self.model.canvas_shape: canvas_shape,
                        self.model.trans: trans_vec,
                        self.model.crop_shape: self.crop_shape})
            elif task_id == 1:
                if reg:
                    cropped_image, cropped_seg, croped_reg = self.model.session.run(
                        [self.model.x_cropped, self.model.y_cropped, self.model.y_reg_cropped],
                        feed_dict={
                        self.model.x: np.reshape(image, [1, shape_in[0], shape_in[1], 1]),
                        self.model.y: np.reshape(seg, [1, shape_in[0], shape_in[1]]),
                        self.model.y_reg: np.expand_dims(landmarks.reshape(-1), 0),
                        self.model.preprocess: 1,
                        self.model.src_shape: shape_in,
                        self.model.canvas_shape: canvas_shape,
                        self.model.trans: trans_vec,
                        self.model.crop_shape: self.crop_shape})

            images[batch_index,:,:,0] = cropped_image[0,:,:,0]
            segs[batch_index,:,:] = cropped_seg[0,:,:]
            if reg:
                landmarks_arr[batch_index,:] = croped_reg

            batch_index = batch_index + 1

            if batch_index == self.batch_size or (d == len(self.val_study_paths)-1 and im == len(study_image_paths)-1):

                if task_id == 0:
                    if reg:
                        preds, preds_reg = self.model.session.run(
                            [self.model.prediction, self.model.reg_output],
                            feed_dict={
                            self.model.x: images,
                            self.model.train: 0,
                            self.model.preprocess: 0,
                            self.model.canvas_shape: np.zeros(2),
                            self.model.trans: np.zeros(8),
                            self.model.crop_shape:  np.zeros(2)})
                    else:
                        preds = self.model.session.run(
                            self.model.prediction,
                            feed_dict={
                            self.model.x: images,
                            self.model.train: 0,
                            self.model.preprocess: 0,
                            self.model.src_shape: np.zeros(2),
                            self.model.canvas_shape: np.zeros(2),
                            self.model.trans: np.zeros(8),
                            self.model.crop_shape:  np.zeros(2)})
                elif task_id == 1:
                    if reg:
                        preds, preds_reg = self.model.session.run(
                            [self.model.prediction, self.model.TrainReg],
                            feed_dict={
                            self.model.x: images,
                            self.model.encoder_train: 0,
                            self.model.decoder_train: 0,
                            self.model.preprocess: 0,
                            self.model.canvas_shape: np.zeros(2),
                            self.model.trans: np.zeros(8),
                            self.model.crop_shape:  np.zeros(2)})

                # Computing iou and dice for the entire batch
                for i in range(batch_index):

                    # Total intersection
                    intersection = 0
                    union = 0

                    for j in range(len(contour_names)):

                        seg_cont = np.copy(segs[i])
                        seg_cont[seg_cont != (j + 1)] = 0

                        pred_cont = np.copy(preds[i])
                        pred_cont[pred_cont != (j + 1)] = 0

                        seg_cont = np.clip(seg_cont, 0, 1)
                        pred_cont = np.clip(pred_cont, 0, 1)

                        inter_cont = np.count_nonzero(pred_cont * seg_cont)
                        pred_cont_size = np.count_nonzero(pred_cont)
                        seg_cont_size = np.count_nonzero(seg_cont)
                        union_cont = pred_cont_size + seg_cont_size - inter_cont

                        if union_cont > 0:
                            iou_per_contour[j].append(float(inter_cont) / float(union_cont))
                            dice_per_contour[j].append(2*float(inter_cont) / float(union_cont + inter_cont))

                        intersection = intersection + inter_cont
                        union = union + union_cont

                    if union > 0:
                        iou.append(float(intersection) / float(union))
                        dice.append(2*float(intersection) / float(union + intersection))

                    if reg:
                        for j in range(0, self.reg_out_size, 2):
                            if j in disabled_reg_indices:
                                continue
                            reg_error_mm.append(norm_pixel_height * self.crop_shape[0] * abs(preds_reg[i,j] - landmarks_arr[i,j]))
                            reg_error_mm.append(norm_pixel_width * self.crop_shape[1] * abs(preds_reg[i,j+1] - landmarks_arr[i,j+1]))

                # Writing to TensorBoard
                if num_of_batches % 10 == 0:
                    node_images = tf.summary.image('val_batch_' + str(num_of_batches) + '_images', tf.cast(self.model.x, tf.float32), max_outputs=3)
                    node_segs = tf.summary.image('val_batch_' + str(num_of_batches) + '_segs', tf.cast(tf.expand_dims(self.model.y, -1), tf.float32), max_outputs=3)
                    node_preds = tf.summary.image('val_batch_' + str(num_of_batches) + '_preds', tf.cast(tf.expand_dims(self.model.prediction, -1), tf.float32), max_outputs=3)
                    if task_id == 0:
                        summary_images, summary_segs, summary_preds = self.model.session.run(
                            [node_images, node_segs, node_preds],
                            feed_dict={
                            self.model.x: images,
                            self.model.y: segs,
                            self.model.train: 0,
                            self.model.preprocess: 0,
                            self.model.src_shape: np.zeros(2),
                            self.model.canvas_shape: np.zeros(2),
                            self.model.trans: np.zeros(8),
                            self.model.crop_shape:  np.zeros(2)})
                    elif task_id == 1:
                        summary_images, summary_segs, summary_preds = self.model.session.run(
                            [node_images, node_segs, node_preds],
                            feed_dict={
                            self.model.x: images,
                            self.model.y: segs,
                            self.model.encoder_train: 0,
                            self.model.decoder_train: 0,
                            self.model.preprocess: 0,
                            self.model.src_shape: np.zeros(2),
                            self.model.canvas_shape: np.zeros(2),
                            self.model.trans: np.zeros(8),
                            self.model.crop_shape:  np.zeros(2)})

                    self.val_writer.add_summary(summary_images, num_of_batches)
                    self.val_writer.add_summary(summary_segs, num_of_batches)
                    self.val_writer.add_summary(summary_preds, num_of_batches)
                    self.val_writer.flush()

                # Restarting variables
                num_of_batches = num_of_batches + 1
                batch_index = 0



    out_file = open(self.checkpoint_dir + 'eval_' + self.dataset + '/eval_output.txt', 'w')

    print()
    print('total_images =', len(iou))
    out_file.write("\n%s %s\n" % ('total_images =', len(iou)))

    print()
    print('Mean IoU =', np.mean(iou))
    print('Std IoU =', np.std(iou))
    print('Mean Dice =', np.mean(dice))
    print('Std Dice =', np.std(dice))
    out_file.write("\n%s %s\n" % ('Mean IoU =', np.mean(iou)))
    out_file.write("%s %s\n" % ('Std IoU =', np.std(iou)))
    out_file.write("%s %s\n" % ('Mean Dice =', np.mean(dice)))
    out_file.write("%s %s\n" % ('Std Dice =', np.std(dice)))

    for i in range(len(contour_names)):
        print()
        print('Mean iou', contour_names[i], '=', np.mean(iou_per_contour[i]))
        print('Std iou', contour_names[i], '=', np.std(iou_per_contour[i]))
        print('Mean dice', contour_names[i], '=', np.mean(dice_per_contour[i]))
        print('Std dice', contour_names[i], '=', np.std(dice_per_contour[i]))
        out_file.write("\n%s %s %s %s\n" % ('Mean iou', contour_names[i], '=', np.mean(iou_per_contour[i])))
        out_file.write("%s %s %s %s\n" % ('Std iou', contour_names[i], '=', np.std(iou_per_contour[i])))
        out_file.write("%s %s %s %s\n" % ('Mean dice', contour_names[i], '=', np.mean(dice_per_contour[i])))
        out_file.write("%s %s %s %s\n" % ('Std dice', contour_names[i], '=', np.std(dice_per_contour[i])))

    if reg:
        print()
        print('Mean Regression error (in mm) =', np.mean(reg_error_mm))
        print('Std Regression error (in mm) =', np.std(reg_error_mm))
        out_file.write("\n%s %s\n" % ('Mean Regression error (in mm) =', np.mean(reg_error_mm)))
        out_file.write("%s %s\n" % ('Std Regression error (in mm) =', np.std(reg_error_mm)))

    print()
    print('Validation took:', (time.time() - start_eval)/60, 'minutes')
    out_file.write("\n%s %s %s\n" % ('Validation took:', (time.time() - start_eval)/60, 'minutes'))

    out_file.close()

    print()
    print('Summary:')
    print('Mean Dice =', np.mean(dice))
    for i in range(len(contour_names)):
        print('Mean dice', contour_names[i], '=', np.mean(dice_per_contour[i]))
    if reg:
        print('Mean Regression error (in mm) =', np.mean(reg_error_mm))

    self.val_writer.close()


def eval_cls(self):
    """Evaluate classification only into {open, closed, none}"""

    ###########################
    # DEBUG
    #self.val_study_paths = self.val_study_paths[0:2]
    #self.val_study_paths = ['/mnt/SSD_1/SaxData_debug_2/3448454/']
    ###########################

    print('---------------------- Evaluation for BioBank validation dataset ----------------------')
    print('Dataset:', self.dataset)

    start_eval = time.time()

    model_util.restore_session(self.checkpoint_dir, self.model.saver, self.model.session)

    # Used for the classification (pred-actual)
    oo = 0 # open-open
    oc = 0 # open-closed
    on = 0 # open-none
    co = 0
    cc = 0
    cn = 0
    no = 0
    nc = 0
    nn = 0

    batch_index = 0
    num_of_batches = 0
    images = np.empty([self.batch_size, self.crop_shape[0], self.crop_shape[1], 1])
    cls_values = np.empty([self.batch_size], dtype = np.int64)

    # For each of the images in the validation set:
    for d in range(len(self.val_study_paths)):

        print('Evaluating study ', d, '/', len(self.val_study_paths))

        study_image_paths = glob.glob(self.val_study_paths[d] + '/*_image.npy')

        for im in range(len(study_image_paths)):

            image = np.load(study_image_paths[im])

            header = np.load(study_image_paths[im][:-10] + '_header.npy').tolist()
            seg_path = study_image_paths[im][:-10] + '_seg.npy'
            reg_path = study_image_paths[im][:-10] + '_landmarks.npy'

            ### Classification ###
            if os.path.isfile(seg_path) and os.path.isfile(reg_path):
                cls_value = 0 # open

            elif os.path.isfile(seg_path) and not os.path.isfile(reg_path):
                # Making sure it's not only a closed RV Endo (i.e. that there's at least one lv_epi pixel)
                seg = np.load(seg_path)
                seg[seg != 2] = 0
                if np.count_nonzero(seg) > 0:
                    cls_value = 1 # closed
                else:
                    cls_value = 2 # None

            elif not os.path.isfile(seg_path) and not os.path.isfile(reg_path):
                cls_value = 2 # None


            ### Scaling and Cropping ###
            shape_in = np.array(np.shape(image))
            res_in = np.array([float(header['pixel_height']), float(header['pixel_width'])])
            res_out = np.array([1.855, 1.855])
            rotate_degree = 0
            mirror = False
            _, _, trans_vec, canvas_shape = im_util.get_transform2d(shape_in, res_in, res_out, rotate_degree, mirror)

            # Preprocessing each image separately (different shapes)
            cropped_image = self.model.session.run(
                self.model.x_cropped,
                feed_dict={
                self.model.x: np.reshape(image, [1, shape_in[0], shape_in[1], 1]),
                self.model.preprocess: 1,
                self.model.canvas_shape: canvas_shape,
                self.model.trans: trans_vec,
                self.model.crop_shape: self.crop_shape})


            images[batch_index,:,:,0] = cropped_image[0,:,:,0]
            cls_values[batch_index] = cls_value
            batch_index = batch_index + 1

            if batch_index == self.batch_size or (d == len(self.val_study_paths)-1 and im == len(study_image_paths)-1):

                cls_out = self.model.session.run(
                    self.model.cls_output,
                    feed_dict={
                    self.model.x: images,
                    self.model.train: 0,
                    self.model.preprocess: 0,
                    self.model.canvas_shape: np.zeros(2),
                    self.model.trans: np.zeros(8),
                    self.model.crop_shape: np.zeros(2)})

                # Computing iou and dice for the entire batch
                for i in range(batch_index):

                    # Comparing cls_values and cls_out
                    cls_pred = cls_out[i]
                    cls_truth = cls_values[i]
                    if cls_pred == 0 and cls_truth == 0:
                        oo = oo + 1
                    elif cls_pred == 0 and cls_truth == 1:
                        oc = oc + 1
                    elif cls_pred == 0 and cls_truth == 2:
                        on = on + 1
                    elif cls_pred == 1 and cls_truth == 0:
                        co = co + 1
                    elif cls_pred == 1 and cls_truth == 1:
                        cc = cc + 1
                    elif cls_pred == 1 and cls_truth == 2:
                        cn = cn + 1
                    elif cls_pred == 2 and cls_truth == 0:
                        no = no + 1
                    elif cls_pred == 2 and cls_truth == 1:
                        nc = nc + 1
                    elif cls_pred == 2 and cls_truth == 2:
                        nn = nn + 1


                # Writing to TensorBoard
                if num_of_batches % 10 == 0:
                    node_images = tf.summary.image('val_batch_' + str(num_of_batches) + '_images', tf.cast(self.model.x, tf.float32), max_outputs=3)
                    summary_images = self.model.session.run(
                        node_images,
                        feed_dict={
                        self.model.x: images})
                    self.val_writer.add_summary(summary_images, num_of_batches)
                    self.val_writer.flush()

                # Restarting variables
                num_of_batches = num_of_batches + 1
                batch_index = 0


    out_file = open(self.checkpoint_dir + 'eval_' + self.dataset + '/eval_output.txt', 'w')


    print()
    TP_open = oo
    FP_open = oc + on
    TN_open = cc + cn + nc + nn
    FN_open = co + no
    ACC_open = float(TP_open + TN_open) / float(TP_open + TN_open + FP_open + FN_open)
    TP_closed = cc
    FP_closed = co + cn
    TN_closed = oo + on + no + nn
    FN_closed = oc + nc
    ACC_closed = float(TP_closed + TN_closed) / float(TP_closed + TN_closed + FP_closed + FN_closed)
    TP_none = nn
    FP_none = no + nc
    TN_none = oo + oc + co + cc
    FN_none = on + cn
    ACC_none = float(TP_none + TN_none) / float(TP_none + TN_none + FP_none + FN_none)
    print('ACC_open =', ACC_open)
    print('ACC_closed =', ACC_closed)
    print('ACC_none =', ACC_none)
    print('oo =', oo)
    print('oc =', oc)
    print('on =', on)
    print('co =', co)
    print('cc =', cc)
    print('cn =', cn)
    print('no =', no)
    print('nc =', nc)
    print('nn =', nn)
    out_file.write("\n%s %s\n" % ('ACC_open =', ACC_open))
    out_file.write("%s %s\n" % ('ACC_closed =', ACC_closed))
    out_file.write("%s %s\n" % ('ACC_none =', ACC_none))
    out_file.write("%s %s\n" % ('oo =', oo))
    out_file.write("%s %s\n" % ('oc =', oc))
    out_file.write("%s %s\n" % ('on =', on))
    out_file.write("%s %s\n" % ('co =', co))
    out_file.write("%s %s\n" % ('cc =', cc))
    out_file.write("%s %s\n" % ('cn =', cn))
    out_file.write("%s %s\n" % ('no =', no))
    out_file.write("%s %s\n" % ('nc =', nc))
    out_file.write("%s %s\n" % ('nn =', nn))

    print()
    print('Validation took:', (time.time() - start_eval)/60, 'minutes')
    out_file.write("\n%s %s %s\n" % ('Validation took:', (time.time() - start_eval)/60, 'minutes'))

    out_file.close()

    self.val_writer.close()


def eval_reg(self, contour_names):
    """Evaluate regression only experiments."""

    ###########################
    # DEBUG
    #self.val_study_paths = self.val_study_paths[0:2]
    #self.val_study_paths = ['/mnt/SSD_1/SaxData_debug_2/3448454/']
    ###########################

    print('---------------------- Evaluation for BioBank validation dataset ----------------------')
    print('Dataset:', self.dataset)

    start_eval = time.time()

    model_util.restore_session(self.checkpoint_dir, self.model.saver, self.model.session)

    # Distance in mm to either one of the contour end points
    dist = []
    for i in range(len(contour_names)):
        dist.append([])

    batch_index = 0
    num_of_batches = 0
    images = np.empty([self.batch_size, self.crop_shape[0], self.crop_shape[1], 1])
    landmarks_arr = np.empty([self.batch_size, self.reg_out_size])

    # For each of the images in the validation set:
    for d in range(len(self.val_study_paths)):

        print('Evaluating study ', d, '/', len(self.val_study_paths))

        study_image_paths = glob.glob(self.val_study_paths[d] + '/*_image.npy')

        for im in range(len(study_image_paths)):

            image = np.load(study_image_paths[im])

            header = np.load(study_image_paths[im][:-10] + '_header.npy').tolist()

            seg_path = study_image_paths[im][:-10] + '_seg.npy'
            landmarks_path = study_image_paths[im][:-10] + '_landmarks.npy'

            if os.path.isfile(landmarks_path) and os.path.isfile(seg_path):
                landmarks = np.array(np.load(landmarks_path))
            else:
                continue # skipping images with no open contours

            ### Scaling and Cropping ###
            shape_in = np.array(np.shape(image))
            res_in = np.array([float(header['pixel_height']), float(header['pixel_width'])])
            res_out = np.array([1.855, 1.855])
            rotate_degree = 0
            mirror = False
            _, _, trans_vec, canvas_shape = im_util.get_transform2d(shape_in, res_in, res_out, rotate_degree, mirror)

            # Preprocessing each image separately (different shapes)
            cropped_image, cropped_landmarks = self.model.session.run(
                [self.model.x_cropped, self.model.y_reg_cropped],
                feed_dict={
                self.model.x: np.reshape(image, [1, shape_in[0], shape_in[1], 1]),
                self.model.y_reg: np.expand_dims(landmarks.reshape(-1), 0),
                self.model.preprocess: 1,
                self.model.src_shape: shape_in,
                self.model.canvas_shape: canvas_shape,
                self.model.trans: trans_vec,
                self.model.crop_shape: self.crop_shape})

            images[batch_index,:,:,0] = cropped_image[0,:,:,0]
            landmarks_arr[batch_index,:] = cropped_landmarks[0,:]
            batch_index = batch_index + 1

            if batch_index == self.batch_size or (d == len(self.val_study_paths)-1 and im == len(study_image_paths)-1):

                reg_out = self.model.session.run(
                    self.model.reg_output,
                    feed_dict={
                    self.model.x: images,
                    self.model.train: 0,
                    self.model.preprocess: 0,
                    self.model.src_shape: np.zeros(2),
                    self.model.canvas_shape: np.zeros(2),
                    self.model.trans: np.zeros(8),
                    self.model.crop_shape: np.zeros(2)})

                # Computing iou and dice for the entire batch
                for i in range(batch_index):

                    #plt.subplot(1,2,1); plt.imshow(preds[i], cmap="gray")
                    #plt.subplot(1,2,2); plt.imshow(segs[i], cmap="gray")
                    #plt.show()

                    reg_truth = landmarks_arr[i]
                    reg_pred = reg_out[i]

                    for j in range(len(contour_names)):
                        dist[j].append((((reg_truth[j*4 + 0]-reg_pred[j*4 + 0])*self.crop_shape[0]*1.855)**2 + ((reg_truth[j*4 + 1]-reg_pred[j*4 + 1])*self.crop_shape[1]*1.855)**2)**0.5)
                        dist[j].append((((reg_truth[j*4 + 2]-reg_pred[j*4 + 2])*self.crop_shape[0]*1.855)**2 + ((reg_truth[j*4 + 3]-reg_pred[j*4 + 3])*self.crop_shape[1]*1.855)**2)**0.5)


                # Writing to TensorBoard
                if num_of_batches % 10 == 0:
                    node_images = tf.summary.image('val_batch_' + str(num_of_batches) + '_images', tf.cast(self.model.x, tf.float32), max_outputs=3)
                    summary_images = self.model.session.run(
                        node_images,
                        feed_dict={
                        self.model.x: images})
                    self.val_writer.add_summary(summary_images, num_of_batches)
                    self.val_writer.flush()

                # Restarting variables
                num_of_batches = num_of_batches + 1
                batch_index = 0


    out_file = open(self.checkpoint_dir + 'eval_' + self.dataset + '/eval_output.txt', 'w')


    for i in range(len(contour_names)):
        print()
        print('Mean', contour_names[i], 'dist (mm) =', np.mean(dist[i]))
        print('Std', contour_names[i], 'dist (mm) =', np.std(dist[i]))
        out_file.write("\n%s %s %s %s\n" % ('Mean', contour_names[i], 'dist (mm) =', np.mean(dist[i])))
        out_file.write("%s %s %s %s\n" % ('Std', contour_names[i], 'dist (mm) =', np.std(dist[i])))

    print()
    print('Validation took:', (time.time() - start_eval)/60, 'minutes')
    out_file.write("\n%s %s %s\n" % ('Validation took:', (time.time() - start_eval)/60, 'minutes'))

    out_file.close()

    self.val_writer.close()


def eval_mc_bold(self, g, eval_mode):
    """Evaluate the Single Modality Motion Correction line of experiments on multi cycle studies
    
    eval_mode == 1: Processing half of all corresponding phases in first vs last cardiac cycle
    eval_mode == 2: Processing first vs half of all other phases in the first cardiac cycle (Same as the evaluation on sax biobank)

    Evaluation is performed using:
    
    1. Global intensity difference
    2. Myocardium intensity difference
    3. Myocardium Dice

    """

    start_eval = time.time()

    # Global intensity criteria
    mov_ref_rms = []
    warped_ref_rms = []

    # Ref Myo intensity criteria (based on ML Segmentation)
    mov_ref_rms_in_ref_myo_mask = []
    warped_ref_rms_in_ref_myo_mask = []

    # Myo Dice
    mov_ref_myo_dice = []
    warped_ref_myo_dice = []

    output_path = self.checkpoint_dir + 'eval_' + self.dataset + '/val_output_' + str(eval_mode)
    subprocess.run(["mkdir", self.checkpoint_dir + 'eval_' + self.dataset])
    subprocess.run(["mkdir", output_path])

    val_subset = len(self.val_study_paths) ## 11
    print('##### Only', val_subset, 'studies will be used for evaluation #####')

    for st in range(val_subset):

        print('Evaluating study ', st, '/', len(self.val_study_paths))

        study_image_paths = glob.glob(self.val_study_paths[st] + '*_img.npy')
        path_split = study_image_paths[0].split('/')
        path_prefix = path_split[0] + '/' + path_split[1] + '/' + path_split[2] + '/' + path_split[3] + '/' + path_split[4] + '/'

        # Each Bold study have exactly 2 slices; each of which can have a different number of cardiac cycles
        max_cycle = [-1, -1]
        max_phase = [-1, -1]
        for im in range(len(study_image_paths)):

            slice = int(study_image_paths[im].split('/')[-1].split('_')[0])
            cycle = int(study_image_paths[im].split('/')[-1].split('_')[1])
            phase = int(study_image_paths[im].split('/')[-1].split('_')[2])

            if cycle > max_cycle[slice]:
                max_cycle[slice] = cycle
            if phase > max_phase[slice]:
                max_phase[slice] = phase


        study_path = output_path + '/' + str(st)
        subprocess.run(["mkdir", study_path])

        # For each slice
        for i in range(2):

            slice_path = study_path + '/' + str(i)
            subprocess.run(["mkdir", slice_path])

            if eval_mode == 2:
                img1 = np.load(path_prefix + str(i) + '_' + str(0) + '_' + str(0) + '_img.npy')
                hdr1 = np.load(path_prefix + str(i) + '_' + str(0) + '_' + str(0) + '_hdr.npy').tolist()

            for j in range(int((max_phase[i] + 1) / 2)):
                
                if eval_mode == 1:
                    img1 = np.load(path_prefix + str(i) + '_' + str(0) + '_' + str(j) + '_img.npy')
                    hdr1 = np.load(path_prefix + str(i) + '_' + str(0) + '_' + str(j) + '_hdr.npy').tolist()
                elif eval_mode == 2:
                    img2 = np.load(path_prefix + str(i) + '_' + str(0) + '_' + str(j) + '_img.npy')
                    hdr2 = np.load(path_prefix + str(i) + '_' + str(0) + '_' + str(j) + '_hdr.npy').tolist()

                if float(hdr1['PixelSpacing'][0]) != float(hdr1['PixelSpacing'][1]):
                    print('Non-isotropic scaling is not supported')
                    exit()

                for k in range(max_cycle[i] + 1):

                    if (eval_mode == 1 and k == max_cycle[i]) or (eval_mode == 2 and k == 0):

                        if eval_mode == 1:
                            img2 = np.load(path_prefix + str(i) + '_' + str(k) + '_' + str(j) + '_img.npy')
                            hdr2 = np.load(path_prefix + str(i) + '_' + str(k) + '_' + str(j) + '_hdr.npy').tolist()

                        ### Scaling and Cropping ###
                        shape_in = np.array(np.shape(img1))
                        res_in = np.array([float(hdr1['PixelSpacing'][0]), float(hdr1['PixelSpacing'][1])])
                        res_out = np.array([1.855, 1.855])
                        rotate_degree = 0
                        mirror = False
                        _, _, trans_vec, canvas_shape = im_util.get_transform2d(
                            shape_in, 
                            res_in, 
                            res_out, 
                            rotate_degree, 
                            mirror)

                        img1_reshaped = np.reshape(img1, [1, shape_in[0], shape_in[1], 1])
                        img2_reshaped = np.reshape(img2, [1, shape_in[0], shape_in[1], 1])

                        old_structure = 0
                        if old_structure:
                            x_mov_cropped, x_ref_cropped = self.model.session.run(
                                [self.model.x1_cropped, self.model.x2_cropped],
                                feed_dict={
                                    self.model.x1: img1_reshaped,
                                    self.model.x2: img2_reshaped,
                                    self.model.canvas_shape: canvas_shape,
                                    self.model.trans: trans_vec})

                            warped, flow = self.model.session.run(
                                [self.model.warped, self.model.flow],
                                feed_dict={
                                    self.model.x_mov: x_mov_cropped,
                                    self.model.x_ref: x_ref_cropped,
                                    self.model.train: 0})
                        else:
                            warped, flow, x_mov_cropped, x_ref_cropped = self.model.session.run(
                                [self.model.warped, self.model.flow, self.model.x_mov_cropped, self.model.x_ref_cropped],
                                feed_dict={
                                    self.model.x_mov: img1_reshaped,
                                    self.model.x_ref: img2_reshaped,
                                    self.model.train: 0,
                                    self.model.preprocess: 1,
                                    self.model.canvas_shape: canvas_shape,
                                    self.model.trans: trans_vec})

                        # Running the sax segmentation network on the moving and reference images
                        images = np.concatenate([img1_reshaped, img2_reshaped], axis=0)
                        preds = self.sax_model.session.run(
                            self.sax_model.prediction,
                            feed_dict={
                                self.sax_model.x: images,
                                self.sax_model.train: 0,
                                self.sax_model.preprocess: 1,
                                self.sax_model.canvas_shape: canvas_shape,
                                self.sax_model.trans: trans_vec,
                                self.sax_model.crop_shape: self.crop_shape})
                        
                        mov_myo_mask = (preds[0]==2)
                        ref_myo_mask = (preds[1]==2)

                        # Computing intensity differences
                        mov_ref_sqr_diff = np.power(x_mov_cropped - x_ref_cropped, 2)
                        warped_ref_sqr_diff = np.power(warped - x_ref_cropped, 2)

                        mov_ref_rms.append(np.sqrt(np.mean(mov_ref_sqr_diff)))
                        warped_ref_rms.append(np.sqrt(np.mean(warped_ref_sqr_diff)))
                        
                        ref_myo_mask_count = np.sum(ref_myo_mask)
                        if ref_myo_mask_count != 0:
                            mov_ref_rms_in_ref_myo_mask.append(np.sqrt(np.sum(mov_ref_sqr_diff[0,:,:,0] * ref_myo_mask) / ref_myo_mask_count))
                            warped_ref_rms_in_ref_myo_mask.append(np.sqrt(np.sum(warped_ref_sqr_diff[0,:,:,0] * ref_myo_mask) / ref_myo_mask_count))

                    
                        # Saving images to disk
                        imsave(slice_path + '/' + str(j) + '_mov.png', x_mov_cropped[0,:,:,0])
                        imsave(slice_path + '/' + str(j) + '_ref.png', x_ref_cropped[0,:,:,0])
                        imsave(slice_path + '/' + str(j) + '_warped.png', warped[0,:,:,0])
                        imsave(slice_path + '/' + str(j) + '_flow_n_x.png', flow[0,:,:,0])
                        imsave(slice_path + '/' + str(j) + '_flow_n_y.png', flow[0,:,:,1])
                        flow_norm = np.uint8((((np.maximum(np.minimum(flow, 8), -8) + 8) / 16) * 255))
                        imsave(slice_path + '/' + str(j) + '_flow_8_x.png', flow_norm[0,:,:,0])
                        imsave(slice_path + '/' + str(j) + '_flow_8_y.png', flow_norm[0,:,:,1])                    

                        warped_myo_mask = im_util.warp_mask(mov_myo_mask, flow)
                        
                        imsave(slice_path + '/' + str(j) + '__mov_myo_mask.png', mov_myo_mask)
                        imsave(slice_path + '/' + str(j) + '__ref_myo_mask.png', ref_myo_mask)
                        imsave(slice_path + '/' + str(j) + '__warped_myo_mask.png', warped_myo_mask)                    

                        # Computing the dice of the myo masks
                        mov_ref_myo_dice.append(im_util.dice(mov_myo_mask, ref_myo_mask))
                        warped_ref_myo_dice.append(im_util.dice(warped_myo_mask, ref_myo_mask))

        
    print()
    print('total image pairs =', len(mov_ref_rms))

    print()
    print('Mov-Ref RMS pixel diff =', round(np.mean(mov_ref_rms), 3), '(std =', round(np.std(mov_ref_rms), 3), ')')
    print('Warped-Ref RMS pixel diff =', round(np.mean(warped_ref_rms), 3), '(std =', round(np.std(warped_ref_rms), 3), ')')

    print()
    print('Mov-Ref RMS pixel diff (in reference myo mask) =', round(np.mean(mov_ref_rms_in_ref_myo_mask), 3), '(std =', round(np.std(mov_ref_rms_in_ref_myo_mask), 3), ')')
    print('Warped-Ref RMS pixel diff (in reference myo mask) =', round(np.mean(warped_ref_rms_in_ref_myo_mask), 3), '(std =', round(np.std(warped_ref_rms_in_ref_myo_mask), 3), ')')
    
    print()
    print('Mov-Ref Mean Myocardium Dice =', round(np.mean(mov_ref_myo_dice), 3), '(std =', round(np.std(mov_ref_myo_dice), 3), ')')
    print('Warped-Ref Mean Myocardium Dice =', round(np.mean(warped_ref_myo_dice), 3), '(std =', round(np.std(warped_ref_myo_dice), 3), ')')  

    print()
    print('Validation took:', (time.time() - start_eval)/60, 'minutes')


def eval_mc(self, g):
    """Evaluate the Single Modality Motion Correction line of experiments on single cycle SAX studies
    
    Evaluation is performed using:
    
    1. Global intensity difference
    2. Myocardium intensity difference
    3. Myocardium Dice

    """

    start_eval = time.time()

    # Global intensity criteria
    mov_ref_rms = []
    warped_ref_rms = []

    # Ref Myo intensity criteria (based on ML Segmentation)
    mov_ref_rms_in_ref_myo_mask = []
    warped_ref_rms_in_ref_myo_mask = []

    # Myo Dice
    mov_ref_myo_dice = []
    warped_ref_myo_dice = []


    output_path = self.checkpoint_dir + 'eval_' + self.dataset + '/val_output'
    subprocess.run(["mkdir", self.checkpoint_dir + 'eval_' + self.dataset])
    subprocess.run(["mkdir", output_path])


    if self.dataset == 'biobank':
        val_subset = 10

    print('##### Only', val_subset, 'studies will be used for evaluation #####')

    for st in range(val_subset):

        print('Evaluating study ', st, '/', len(self.val_study_paths))

        if self.dataset == 'biobank':
            study_image_paths = glob.glob(self.val_study_paths[st] + 'sax/' + '*_img.npy')
            path_split = study_image_paths[0].split('/')
            path_prefix = path_split[0] + '/' + path_split[1] + '/' + path_split[2] + '/' + path_split[3] + '/' + path_split[4] + '/' + path_split[5] + '/'
            
            
        # For some studies, the number of phases in the first slice differs from all other slices
        # e.g in '/mnt/HDD_2/BioBankNumpy/5697914/sax/' slice 9 has 15 phases while all the other slices has 50
        # So we handle each slice separately
        slices = []
        for im in range(len(study_image_paths)):
            slice = int(study_image_paths[im].split('/')[-1].split('_')[0])
            if slice not in slices:
                slices.append(slice)

        if self.dataset == 'biobank':
            # using middle slices only
            #slices = slices[int(len(slices)/2) - 1:int(len(slices)/2) + 2]
            slices = sorted(slices)[int(len(slices)/4):int(len(slices)/4) + int(len(slices)/2)]

        phases = []
        for i in range(len(slices)):
            phases.append([])

        for im in range(len(study_image_paths)):
            slice = int(study_image_paths[im].split('/')[-1].split('_')[0])
            phase = int(study_image_paths[im].split('/')[-1].split('_')[1])
            if slice in slices:
                slice_index = slices.index(slice)
                if phase not in phases[slice_index]:
                    phases[slice_index].append(phase)

        for i in range(len(phases)):
            phases[i] = sorted(phases[i])


        study_path = output_path + '/' + str(st)
        subprocess.run(["mkdir", study_path])

        for i in range(len(slices)):

            slice_path = study_path + '/' + str(i)
            subprocess.run(["mkdir", slice_path])

            images_in_slice = []
            headers_in_slice = []

            for j in range(len(phases[i])):

                path_split = study_image_paths[im].split('/')
                if self.dataset == 'biobank':
                    image = np.load(path_prefix + str(slices[i]).zfill(2) + '_' + str(phases[i][j]).zfill(2) + '_img.npy')
                    header = np.load(path_prefix + str(slices[i]).zfill(2) + '_' + str(phases[i][j]).zfill(2) + '_hdr.npy').tolist()
                    if float(header['PixelSpacing'][0]) != float(header['PixelSpacing'][1]):
                        print('Non-isotropic scaling is not supported')
                        exit()

                images_in_slice.append(image)
                headers_in_slice.append(header)


            ## Evaluating the first phase (as mov) with half of all phases (as ref)(including itself)
            ind1 = 0
            img1 = images_in_slice[ind1]
            hdr1 = headers_in_slice[ind1]

            for ind2 in range(len(images_in_slice)):

                # Evaluate half the sequance in middle slices only
                if ind2 <= int(len(images_in_slice) / 2):

                    img2 = images_in_slice[ind2]
                    hdr2 = headers_in_slice[ind2]

                    ### Scaling and Cropping ###
                    shape_in = np.array(np.shape(img1))
                    if self.dataset == 'biobank':
                        res_in = np.array([float(hdr1['PixelSpacing'][0]), float(hdr1['PixelSpacing'][1])])
                    res_out = np.array([1.855, 1.855])
                    rotate_degree = 0
                    mirror = False
                    _, _, trans_vec, canvas_shape = im_util.get_transform2d(
                        shape_in, 
                        res_in, 
                        res_out, 
                        rotate_degree, 
                        mirror)

                    img1_reshaped = np.reshape(img1, [1, shape_in[0], shape_in[1], 1])
                    img2_reshaped = np.reshape(img2, [1, shape_in[0], shape_in[1], 1])

                    old_structure = 0
                    if old_structure:
                        x_mov_cropped, x_ref_cropped = self.model.session.run(
                            [self.model.x1_cropped, self.model.x2_cropped],
                            feed_dict={
                            self.model.x1: img1_reshaped,
                            self.model.x2: img2_reshaped,
                            self.model.canvas_shape: canvas_shape,
                            self.model.trans: trans_vec})

                        warped, flow = self.model.session.run(
                            [self.model.warped, self.model.flow],
                            feed_dict={
                            self.model.x_mov: x_mov_cropped,
                            self.model.x_ref: x_ref_cropped,
                            self.model.train: 0})
                    else:
                        warped, flow, x_mov_cropped, x_ref_cropped = self.model.session.run(
                            [self.model.warped, self.model.flow, self.model.x_mov_cropped, self.model.x_ref_cropped],
                            feed_dict={
                                self.model.x_mov: img1_reshaped,
                                self.model.x_ref: img2_reshaped,
                                self.model.train: 0,
                                self.model.preprocess: 1,
                                self.model.canvas_shape: canvas_shape,
                                self.model.trans:  trans_vec})

                    # Running the sax segmentation network on the moving and reference images
                    images = np.concatenate([img1_reshaped, img2_reshaped], axis=0)
                    preds = self.sax_model.session.run(
                        self.sax_model.prediction,
                        feed_dict={
                            self.sax_model.x: images,
                            self.sax_model.train: 0,
                            self.sax_model.preprocess: 1,
                            self.sax_model.canvas_shape: canvas_shape,
                            self.sax_model.trans: trans_vec,
                            self.sax_model.crop_shape: self.crop_shape})
                    
                    mov_myo_mask = (preds[0]==2)
                    ref_myo_mask = (preds[1]==2)

                    # Computing intensity differences
                    mov_ref_sqr_diff = np.power(x_mov_cropped - x_ref_cropped, 2)
                    warped_ref_sqr_diff = np.power(warped - x_ref_cropped, 2)

                    mov_ref_rms.append(np.sqrt(np.mean(mov_ref_sqr_diff)))
                    warped_ref_rms.append(np.sqrt(np.mean(warped_ref_sqr_diff)))
                    
                    ref_myo_mask_count = np.sum(ref_myo_mask)
                    if ref_myo_mask_count != 0:
                        mov_ref_rms_in_ref_myo_mask.append(np.sqrt(np.sum(mov_ref_sqr_diff[0,:,:,0] * ref_myo_mask) / ref_myo_mask_count))
                        warped_ref_rms_in_ref_myo_mask.append(np.sqrt(np.sum(warped_ref_sqr_diff[0,:,:,0] * ref_myo_mask) / ref_myo_mask_count))

                    # Saving images to disk
                    imsave(slice_path + '/' + str(ind2) + '_mov.png', x_mov_cropped[0,:,:,0])
                    imsave(slice_path + '/' + str(ind2) + '_ref.png', x_ref_cropped[0,:,:,0])
                    imsave(slice_path + '/' + str(ind2) + '_warped.png', warped[0,:,:,0])
                    imsave(slice_path + '/' + str(ind2) + '_flow_n_x.png', flow[0,:,:,0])
                    imsave(slice_path + '/' + str(ind2) + '_flow_n_y.png', flow[0,:,:,1])
                    flow_norm = np.uint8((((np.maximum(np.minimum(flow, 8), -8) + 8) / 16) * 255))
                    imsave(slice_path + '/' + str(ind2) + '_flow_8_x.png', flow_norm[0,:,:,0])
                    imsave(slice_path + '/' + str(ind2) + '_flow_8_y.png', flow_norm[0,:,:,1])                    

                    warped_myo_mask = im_util.warp_mask(mov_myo_mask, flow)
                    
                    imsave(slice_path + '/' + str(ind2) + '__mov_myo_mask.png', mov_myo_mask)
                    imsave(slice_path + '/' + str(ind2) + '__ref_myo_mask.png', ref_myo_mask)
                    imsave(slice_path + '/' + str(ind2) + '__warped_myo_mask.png', warped_myo_mask)                    

                    # Computing the dice of the myo masks
                    mov_ref_myo_dice.append(im_util.dice(mov_myo_mask, ref_myo_mask))
                    warped_ref_myo_dice.append(im_util.dice(warped_myo_mask, ref_myo_mask))


        
    print()
    print('total image pairs =', len(mov_ref_rms))

    print()
    print('Mov-Ref RMS pixel diff =', round(np.mean(mov_ref_rms), 3), '(std =', round(np.std(mov_ref_rms), 3), ')')
    print('Warped-Ref RMS pixel diff =', round(np.mean(warped_ref_rms), 3), '(std =', round(np.std(warped_ref_rms), 3), ')')

    print()
    print('Mov-Ref RMS pixel diff (in reference myo mask) =', round(np.mean(mov_ref_rms_in_ref_myo_mask), 3), '(std =', round(np.std(mov_ref_rms_in_ref_myo_mask), 3), ')')
    print('Warped-Ref RMS pixel diff (in reference myo mask) =', round(np.mean(warped_ref_rms_in_ref_myo_mask), 3), '(std =', round(np.std(warped_ref_rms_in_ref_myo_mask), 3), ')')
    
    print()
    print('Mov-Ref Mean Myocardium Dice =', round(np.mean(mov_ref_myo_dice), 3), '(std =', round(np.std(mov_ref_myo_dice), 3), ')')
    print('Warped-Ref Mean Myocardium Dice =', round(np.mean(warped_ref_myo_dice), 3), '(std =', round(np.std(warped_ref_myo_dice), 3), ')')  

    print()
    print('Validation took:', (time.time() - start_eval)/60, 'minutes')


def eval_mc_mi(self, is_patch=False):
    """Evaluate the motion correction experiment using mutual information difference.

    Used for the Biobank and T1

    This function can be used in two modes:
    1. self.exp_mode == 1:
    Evaluating the mi_estimator against the true mi of a random pair of images from the same slice.
    2. self.exp_mode == 2:
    Evaluating the warper against the maximum mi, i.e. the mi of the ref image with itself.

    This function works for the mmmc_x line of experiments
    """

    print('Dataset:', self.dataset)

    start_eval = time.time()

    # model_util.restore_session(self.checkpoint_dir_restore, self.model.saver, self.model.session)

    mi_diff = []

    output_path = self.checkpoint_dir + 'eval_' + self.dataset + '/val_output'
    subprocess.run(["mkdir", self.checkpoint_dir + 'eval_' + self.dataset])
    subprocess.run(["mkdir", output_path])

    if self.dataset == 'biobank':
        val_subset = 10
    elif self.dataset == 'biobank_shmolli':
        val_subset = 10

    print('##### Only', val_subset, 'studies will be used for evaluation #####')
    # For each of the studies in the validation set:
    # for st in range(len(self.val_study_paths)):
    for st in range(val_subset):

        print('Evaluating study ', st, '/', len(self.val_study_paths))

        if self.dataset == 'biobank':
            study_image_paths = glob.glob(self.val_study_paths[st] + 'sax/' + '*_img.npy')
            path_split = study_image_paths[0].split('/')
            path_prefix = path_split[0] + '/' + path_split[1] + '/' + path_split[2] + '/' + path_split[3] + '/' + path_split[4] + '/' + path_split[5] + '/'
        elif self.dataset == 'biobank_shmolli':
            study_image_paths = glob.glob(self.val_study_paths[st] + '*_img.npy')
            path_split = study_image_paths[0].split('/')
            path_prefix = path_split[0] + '/' + path_split[1] + '/' + path_split[2] + '/' + path_split[3] + '/' + path_split[4] + '/'

        # For some studies, the number of phases in the first slice differs from all other slices
        # e.g in '/mnt/HDD_2/BioBankNumpy/5697914/sax/' slice 9 has 15 phases while all the other slices has 50
        # So we handle each slice separately
        slices = []
        for im in range(len(study_image_paths)):
            slice = int(study_image_paths[im].split('/')[-1].split('_')[0])
            if slice not in slices:
                slices.append(slice)

        if self.dataset == 'biobank':
            # using middle slices only
            #slices = slices[int(len(slices)/2) - 1:int(len(slices)/2) + 2]
            slices = sorted(slices)[int(len(slices)/4):int(len(slices)/4) + int(len(slices)/2)]

        phases = []
        for i in range(len(slices)):
            phases.append([])

        for im in range(len(study_image_paths)):
            slice = int(study_image_paths[im].split('/')[-1].split('_')[0])
            phase = int(study_image_paths[im].split('/')[-1].split('_')[1])
            if slice in slices:
                slice_index = slices.index(slice)
                if phase not in phases[slice_index]:
                    phases[slice_index].append(phase)

        for i in range(len(phases)):
            phases[i] = sorted(phases[i])

        if self.dataset == 'biobank_shmolli':
            # the preprocessing script makes sure there's a single series (slice) in each study
            assert(len(slices) == 1)
            # skip studies which have a single phase
            if (len(phases[0]) == 1):
                continue

        if self.exp_mode == 2:
            study_path = output_path + '/' + str(st)
            subprocess.run(["mkdir", study_path])

        for i in range(len(slices)):

            images_in_slice = []
            headers_in_slice = []

            for j in range(len(phases[i])):

                if self.dataset == 'biobank':
                    image = np.load(path_prefix + str(slices[i]).zfill(2) + '_' + str(phases[i][j]).zfill(2) + '_img.npy')
                    header = np.load(path_prefix + str(slices[i]).zfill(2) + '_' + str(phases[i][j]).zfill(2) + '_hdr.npy').tolist()
                    if float(header['PixelSpacing'][0]) != float(header['PixelSpacing'][1]):
                        print('Non-isotropic scaling is not supported')
                        exit()
                elif self.dataset == 'biobank_shmolli':
                    image = np.load(path_prefix + str(slices[i]) + '_' + str(phases[i][j]) + '_img.npy')
                    header = np.load(path_prefix + str(slices[i]) + '_' + str(phases[i][j]) + '_hdr.npy').tolist()
                    if float(header['pixel_height']) != float(header['pixel_width']):
                        print('Non-isotropic scaling is not supported')
                        exit()

                images_in_slice.append(image)
                headers_in_slice.append(header)


            ## Evaluating the first phase (as mov) with all phases (as ref)(including itself)
            ind1 = 0
            img1 = images_in_slice[ind1]
            hdr1 = headers_in_slice[ind1]

            for ind2 in range(len(images_in_slice)):

                img2 = images_in_slice[ind2]
                hdr2 = headers_in_slice[ind2]

                ### Scaling and Cropping ###
                shape_in = np.array(np.shape(img1))
                if self.dataset == 'biobank':
                    res_in = np.array([float(hdr1['PixelSpacing'][0]), float(hdr1['PixelSpacing'][1])])
                elif self.dataset == 'biobank_shmolli':
                    res_in = np.array([float(hdr1['pixel_height']), float(hdr1['pixel_width'])])
                res_out = np.array([1.855, 1.855])
                rotate_degree = 0
                mirror = False

                _, _, trans_vec, canvas_shape = im_util.get_transform2d(
                    shape_in,
                    res_in,
                    res_out,
                    rotate_degree,
                    mirror)

                # Preprocessing
                x_mov_cropped, x_ref_cropped = self.model.session.run(
                    [self.model.x_mov_cropped, self.model.x_ref_cropped],
                    feed_dict={
                        self.model.x_mov: np.reshape(img1, [1, shape_in[0], shape_in[1], 1]),
                        self.model.x_ref: np.reshape(img2, [1, shape_in[0], shape_in[1], 1]),
                        self.model.preprocess: 1,
                        self.model.canvas_shape: canvas_shape,
                        self.model.trans: trans_vec})

                if self.exp_mode == 1:
                    # using the real mi between the mov and ref image as a target value
                    if is_patch:
                        target_mi = im_util.patch_mi(x_mov_cropped[0,:,:,0], x_ref_cropped[0,:,:,0], self.num_of_bins, self.patch_size, self.patch_stride, self.patch_count)
                    else:
                        target_mi = im_util.mi(x_mov_cropped[0,:,:,0], x_ref_cropped[0,:,:,0], self.num_of_bins)
                elif self.exp_mode == 2:
                    # using the estimated mi of the ref image with itself as a target value
                    target_mi = self.model.session.run(
                        [self.model.pred_mi],
                        feed_dict={
                            self.model.x_mov: x_ref_cropped,
                            self.model.x_ref: x_ref_cropped,
                            self.model.train: 0,
                            self.model.preprocess: 0,
                            self.model.canvas_shape: np.zeros([2]),
                            self.model.trans: np.zeros([8]),
                            self.model.skip_warper: 1})[0]

                pred_mi = self.model.session.run(
                    [self.model.pred_mi],
                    feed_dict={
                        self.model.x_mov: x_mov_cropped,
                        self.model.x_ref: x_ref_cropped,
                        self.model.train: 0,
                        self.model.preprocess: 0,
                        self.model.canvas_shape: np.zeros([2]),
                        self.model.trans: np.zeros([8]),
                        self.model.skip_warper: self.skip_warper})

                mi_diff.append(np.abs(pred_mi - target_mi))

                # Saving images to disk
                if self.dataset == 'biobank':
                    end_phase = int(len(images_in_slice) / 2)
                elif self.dataset == 'biobank_shmolli':
                    end_phase = int(len(images_in_slice))

                if self.exp_mode == 2 and i == int(len(slices) / 2) and ind2 <= end_phase:
                    warped, flow = \
                        self.model.session.run(
                            [self.model.warped, self.model.flow],
                            feed_dict={
                                self.model.x_mov: x_mov_cropped,
                                self.model.x_ref: x_ref_cropped,
                                self.model.train: 0,
                                self.model.preprocess: 0,
                                self.model.canvas_shape: np.zeros([2]),
                                self.model.trans: np.zeros([8]),
                                self.model.skip_warper: self.skip_warper})

                    file_name_prefix = study_path + '/' + str(ind2)
                    imsave(file_name_prefix + '_mov.png', x_mov_cropped[0,:,:,0])
                    imsave(file_name_prefix + '_ref.png', x_ref_cropped[0,:,:,0])
                    imsave(file_name_prefix + '_warped.png', warped[0,:,:,0])
                    imsave(file_name_prefix + '_flow_n_x.png', flow[0,:,:,0])
                    imsave(file_name_prefix + '_flow_n_y.png', flow[0,:,:,1])
                    flow = np.uint8((((np.maximum(np.minimum(flow, 8), -8) + 8) / 16) * 255))
                    imsave(file_name_prefix + '_flow_8_x.png', flow[0,:,:,0])
                    imsave(file_name_prefix + '_flow_8_y.png', flow[0,:,:,1])

                    # drawing the joint histograms
                    #mov_ref_hist = im_util.joint_hist(x_mov_cropped[0,:,:,0], x_ref_cropped[0,:,:,0], self.num_of_bins)
                    #warped_ref_hist = im_util.joint_hist(warped[0,:,:,0], x_ref_cropped[0,:,:,0], self.num_of_bins)
                    #imsave(file_name_prefix + '_mov_ref_hist.png', mov_ref_hist)
                    #imsave(file_name_prefix + '_warped_ref_hist.png', warped_ref_hist)


    out_file = open(self.checkpoint_dir + 'eval_' + self.dataset + '/eval_mode_' + str(self.exp_mode) + '.txt', 'w')

    print()
    print('Total image pairs =', len(mi_diff))

    print()
    print('Mean mi diff =', np.mean(mi_diff))
    print('Std mi diff =', np.std(mi_diff))
    print('Max mi diff =', np.max(mi_diff))
    print('Min mi diff =', np.min(mi_diff))
    out_file.write("\n%s %s\n" % ('Mean mi diff =', np.mean(mi_diff)))
    out_file.write("%s %s\n" % ('Std mi diff =', np.std(mi_diff)))
    out_file.write("%s %s\n" % ('Max mi diff =', np.max(mi_diff)))
    out_file.write("%s %s\n" % ('Min mi diff =', np.min(mi_diff)))

    print()
    print('Validation took:', (time.time() - start_eval)/60, 'minutes')
    out_file.write("\n%s %s %s\n" % ('Validation took:', (time.time() - start_eval)/60, 'minutes'))

    out_file.close()


def eval_mc_mi_perfusion(self, is_patch=False):
    """Evaluate the motion correction experiment using mutual information difference.

    Used for the Perfusion dataset

    This function can be used in two modes:
    1. self.exp_mode == 1:
    Evaluating the mi_estimator against the true mi of a random pair of images from the same slice.
    2. self.exp_mode == 2:
    Evaluating the warper against the maximum mi, i.e. the mi of the ref image with itself.

    This function works for the mmmc_x line of experiments
    """

    print('Dataset:', self.dataset)

    start_eval = time.time()

    # model_util.restore_session(self.checkpoint_dir_restore, self.model.saver, self.model.session)

    mi_diff = []

    output_path = self.checkpoint_dir + 'eval_' + self.dataset + '/val_output'
    subprocess.run(["mkdir", self.checkpoint_dir + 'eval_' + self.dataset])
    subprocess.run(["mkdir", output_path])

    if self.dataset == 'perfusion':
        val_subset = len(self.val_study_paths)

    print('##### Only', val_subset, 'studies will be used for evaluation #####')
    # For each of the studies in the validation set:
    # for st in range(len(self.val_study_paths)):
    for st in range(val_subset):

        print('Evaluating study ', st, '/', len(self.val_study_paths))

        if self.exp_mode == 2:
            study_path = output_path + '/' + str(st)
            subprocess.run(["mkdir", study_path])

        if self.dataset == 'perfusion':
            study_image_paths = glob.glob(self.val_study_paths[st] + '*_img.npy')
            path_split = study_image_paths[0].split('/')
            path_prefix = path_split[0] + '/' + path_split[1] + '/' + path_split[2] + '/' + path_split[3] + '/' + path_split[4] + '/'


       # Handling each series-slice pair separately
        series_ids = []
        series_slices = []
        for im in range(len(study_image_paths)):
            series = int(study_image_paths[im].split('/')[-1].split('_')[0])
            slice = int(study_image_paths[im].split('/')[-1].split('_')[2])
            if series not in series_ids:
                series_ids.append(series)
                series_slices.append([])
            series_index = series_ids.index(series)
            if slice not in series_slices[series_index]:
                series_slices[series_index].append(slice)


        for i in range(len(series_slices)):

            for j in range(len(series_slices[i])):

                slice_image_paths = glob.glob(self.val_study_paths[st] + str(series_ids[i]) + '_*_' + str(series_slices[i][j]) + '_img.npy')
                
                images_in_slice = []
                headers_in_slice = []

                for k in range(len(slice_image_paths)):

                    path_split = slice_image_paths[k].split('/')
                    path_prefix = path_split[0] + '/' + path_split[1] + '/' + path_split[2] + '/' + path_split[3] + '/' + path_split[4] + '/'
                    file_name = path_split[5]
                    phase = file_name.split('_')[1]

                    image = np.load(path_prefix + str(series_ids[i]) + '_' + phase + '_' + str(series_slices[i][j]) + '_img.npy')
                    header = np.load(path_prefix + str(series_ids[i]) + '_' + phase + '_' + str(series_slices[i][j]) + '_hdr.npy').tolist()
                    
                    images_in_slice.append(image)
                    headers_in_slice.append(header)


                ## Evaluating the first phase (as mov) with all phases (as ref)(including itself)
                ind1 = 0
                img1 = images_in_slice[ind1]
                hdr1 = headers_in_slice[ind1]

                for ind2 in range(len(images_in_slice)):

                    img2 = images_in_slice[ind2]
                    hdr2 = headers_in_slice[ind2]

                    ### Scaling and Cropping ###
                    shape_in = np.array(np.shape(img1))
                    if self.dataset == 'biobank' or self.dataset == 'perfusion':
                        res_in = np.array([float(hdr1['PixelSpacing'][0]), float(hdr1['PixelSpacing'][1])])
                    elif self.dataset == 'biobank_shmolli':
                        res_in = np.array([float(hdr1['pixel_height']), float(hdr1['pixel_width'])])
                    res_out = np.array([1.855, 1.855])
                    rotate_degree = 0
                    mirror = False

                    _, _, trans_vec, canvas_shape = im_util.get_transform2d(
                        shape_in,
                        res_in,
                        res_out,
                        rotate_degree,
                        mirror)

                    # Preprocessing
                    x_mov_cropped, x_ref_cropped = self.model.session.run(
                        [self.model.x_mov_cropped, self.model.x_ref_cropped],
                        feed_dict={
                            self.model.x_mov: np.reshape(img1, [1, shape_in[0], shape_in[1], 1]),
                            self.model.x_ref: np.reshape(img2, [1, shape_in[0], shape_in[1], 1]),
                            self.model.preprocess: 1,
                            self.model.canvas_shape: canvas_shape,
                            self.model.trans: trans_vec})

                    if self.exp_mode == 1:
                        # using the real mi between the mov and ref image as a target value
                        if is_patch:
                            target_mi = im_util.patch_mi(x_mov_cropped[0,:,:,0], x_ref_cropped[0,:,:,0], self.num_of_bins, self.patch_size, self.patch_stride, self.patch_count)
                        else:
                            target_mi = im_util.mi(x_mov_cropped[0,:,:,0], x_ref_cropped[0,:,:,0], self.num_of_bins)
                    elif self.exp_mode == 2:
                        # using the estimated mi of the ref image with itself as a target value
                        target_mi = self.model.session.run(
                            [self.model.pred_mi],
                            feed_dict={
                                self.model.x_mov: x_ref_cropped,
                                self.model.x_ref: x_ref_cropped,
                                self.model.train: 0,
                                self.model.preprocess: 0,
                                self.model.canvas_shape: np.zeros([2]),
                                self.model.trans: np.zeros([8]),
                                self.model.skip_warper: 1})[0]

                    pred_mi = self.model.session.run(
                        [self.model.pred_mi],
                        feed_dict={
                        self.model.x_mov: x_mov_cropped,
                        self.model.x_ref: x_ref_cropped,
                        self.model.train: 0,
                        self.model.preprocess: 0,
                        self.model.canvas_shape: np.zeros([2]),
                        self.model.trans: np.zeros([8]),
                        self.model.skip_warper: self.skip_warper})

                    mi_diff.append(np.abs(pred_mi - target_mi))

                    # Saving images to disk
                    if self.dataset == 'perfusion':
                        end_phase = int(len(images_in_slice))

                    if self.exp_mode == 2 and ind2 <= end_phase:

                        warped, flow = self.model.session.run(
                            [self.model.warped, self.model.flow],
                            feed_dict={
                                self.model.x_mov: x_mov_cropped,
                                self.model.x_ref: x_ref_cropped,
                                self.model.train: 0,
                                self.model.preprocess: 0,
                                self.model.canvas_shape: np.zeros([2]),
                                self.model.trans: np.zeros([8]),
                                self.model.skip_warper: self.skip_warper})

                        file_name_prefix = study_path + '/' + str(i) + '_' + str(j) + '_' + str(ind2)
                        imsave(file_name_prefix + '_mov.png', x_mov_cropped[0,:,:,0])
                        imsave(file_name_prefix + '_ref.png', x_ref_cropped[0,:,:,0])
                        imsave(file_name_prefix + '_warped.png', warped[0,:,:,0])
                        imsave(file_name_prefix + '_flow_n_x.png', flow[0,:,:,0])
                        imsave(file_name_prefix + '_flow_n_y.png', flow[0,:,:,1])
                        flow = np.uint8((((np.maximum(np.minimum(flow, 8), -8) + 8) / 16) * 255))
                        imsave(file_name_prefix + '_flow_8_x.png', flow[0,:,:,0])
                        imsave(file_name_prefix + '_flow_8_y.png', flow[0,:,:,1])
                        
                        # drawing the joint histograms
                        #mov_ref_hist = im_util.joint_hist(x_mov_cropped[0,:,:,0], x_ref_cropped[0,:,:,0], self.num_of_bins)
                        #warped_ref_hist = im_util.joint_hist(warped[0,:,:,0], x_ref_cropped[0,:,:,0], self.num_of_bins)
                        #imsave(file_name_prefix + '_mov_ref_hist.png', mov_ref_hist)
                        #imsave(file_name_prefix + '_warped_ref_hist.png', warped_ref_hist)

                    
    out_file = open(self.checkpoint_dir + 'eval_' + self.dataset + '/eval_mode_' + str(self.exp_mode) + '.txt', 'w')

    print()
    print('Total image pairs =', len(mi_diff))

    print()
    print('Mean mi diff =', np.mean(mi_diff))
    print('Std mi diff =', np.std(mi_diff))
    print('Max mi diff =', np.max(mi_diff))
    print('Min mi diff =', np.min(mi_diff))
    out_file.write("\n%s %s\n" % ('Mean mi diff =', np.mean(mi_diff)))
    out_file.write("%s %s\n" % ('Std mi diff =', np.std(mi_diff)))
    out_file.write("%s %s\n" % ('Max mi diff =', np.max(mi_diff)))
    out_file.write("%s %s\n" % ('Min mi diff =', np.min(mi_diff)))

    print()
    print('Validation took:', (time.time() - start_eval)/60, 'minutes')
    out_file.write("\n%s %s %s\n" % ('Validation took:', (time.time() - start_eval)/60, 'minutes'))

    out_file.close()