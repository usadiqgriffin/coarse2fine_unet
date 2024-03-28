import numpy as np
import csv
import tensorflow as tf
import os
import datetime
import subprocess
import sys
import random
import time
import nibabel as nib
import scipy.misc
import utils.stn3d as stn3d
from tqdm import tqdm
from datetime import datetime
import glob
from tensorflow.python.tools import freeze_graph
import logging
from tensorflow.keras.layers import UpSampling3D, AveragePooling3D

def tensor_print(x, text):
    tf.compat.v1.Print(x, [x], message=text + ": ", summarize=80)
class Model(object):


    def fix_border_issue(self, image):

        min_val = tf.reduce_min(image)

        fixed_image = tf.pad(image[:, 1:-1, 1:-1, 1:-1, :], 
                             tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]),
                             mode='CONSTANT', 
                             constant_values=min_val)
        return fixed_image

    
    def deep_dice_bce_loss(self, true, pred, deep_pred, smooth=1e-5):
        
        label_n = 0
        def dice_loss(true, pred):
            true = tf.cast(true, tf.float32)
            pred = tf.cast(pred, tf.float32)
            #logging.debug(f"True shape:{true.shape}, pred shape:{pred.shape}")
            numerator = tf.reduce_sum(true * pred, axis=[1, 2, 3]) + smooth
            denominator = tf.reduce_sum(true, axis=[1, 2, 3]) + tf.reduce_sum(pred, axis=[1, 2, 3]) + smooth
            loss = -(numerator / denominator)
            return loss

        def call(true, pred):
            
            logging.debug(f"True shape:{true.shape}, pred shape:{pred.shape}, deep pred shape:{len(deep_pred)}")
            logging.debug(f"Deep pred shapes:({deep_pred[0].shape}), ({deep_pred[1].shape}), ({deep_pred[2].shape})")

            true = tf.cast(tf.equal(true, label_n * tf.ones_like(true)), tf.float32)
            pred = tf.cast(pred[..., label_n], tf.float32)
            
            foreground_dice_loss = dice_loss(true, pred)
            background_dice_loss = dice_loss(1 - true, 1 - pred)
            bce = tf.keras.losses.binary_crossentropy(true, pred, from_logits=False)
            bce = tf.reduce_mean(bce)
            loss = foreground_dice_loss + background_dice_loss + bce
            return loss
            
        return call(true, pred)

    
    def dice_bce_loss(true, pred, label_n, smooth=1e-5):
        
        def dice_loss(true, pred):
            true = tf.cast(true, tf.float32)
            pred = tf.cast(pred, tf.float32)
            numerator = tf.reduce_sum(true * pred, axis=[1, 2, 3]) + smooth
            denominator = tf.reduce_sum(true, axis=[1, 2, 3]) + tf.reduce_sum(pred, axis=[1, 2, 3]) + smooth
            loss = -(numerator / denominator)
            return loss

        def call(true, pred):
            
            true = tf.cast(tf.equal(true, label_n * tf.ones_like(true)), tf.float32)
            pred = tf.cast(pred[..., label_n], tf.float32)
            
            foreground_dice_loss = dice_loss(true, pred)
            background_dice_loss = dice_loss(1 - true, 1 - pred)
            bce = tf.keras.losses.binary_crossentropy(true, pred, from_logits=False)
            bce = tf.reduce_mean(bce)
            loss = foreground_dice_loss + background_dice_loss + bce
            return loss
            
        return call(true, pred)
    
    
    def __init__(self, output_path, clean_output, create_summary, gpu):

        self.output_path = output_path

        if gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = '1'
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ''

        # Defining the Tensorflow graph
        self.sess = tf.Session()

        if clean_output:
            # Cleaning the output folder
            if os.path.isdir(output_path):
                now = datetime.now()
                now_str = now.strftime("%Y%m%d_%H%M%S")
                os.system('mv ' + output_path + ' ' + output_path + '_' + now_str)

        if create_summary:
            self.summary_writer_train = tf.summary.FileWriter(output_path + '/tensorboard/train', graph=self.sess.graph)
            self.summary_writer_val = tf.summary.FileWriter(output_path + '/tensorboard/val', graph=self.sess.graph)


    def classifier(self, input):

        down1 = tf.layers.conv3d(input, 8, (3, 3, 3), (1, 1, 1), 'same', activation=tf.nn.relu)  # [batch, 80, 192, 160]
        down1 = tf.layers.conv3d(down1, 8, (3, 3, 3), (1, 1, 1), 'same', activation=tf.nn.relu)
        down1 = tf.layers.batch_normalization(down1, training=self.training)

        down2 = tf.layers.conv3d(down1, 8, (3, 3, 3), (2, 2, 2), 'same', activation=tf.nn.relu)  # [40, 96, 80]
        down2 = tf.layers.conv3d(down2, 8, (3, 3, 3), (1, 1, 1), 'same', activation=tf.nn.relu)
        down2 = tf.layers.batch_normalization(down2, training=self.training)

        down3 = tf.layers.conv3d(down2, 16, (3, 3, 3), (2, 2, 2), 'same', activation=tf.nn.relu)  # [20, 48, 40]
        down3 = tf.layers.conv3d(down3, 16, (3, 3, 3), (1, 1, 1), 'same', activation=tf.nn.relu)
        down3 = tf.layers.batch_normalization(down3, training=self.training)

        down4 = tf.layers.conv3d(down3, 16, (3, 3, 3), (2, 2, 2), 'same', activation=tf.nn.relu)  # [10, 24, 20]
        down4 = tf.layers.conv3d(down4, 16, (3, 3, 3), (1, 1, 1), 'same', activation=tf.nn.relu)
        down4 = tf.layers.batch_normalization(down4, training=self.training)

        down5 = tf.layers.conv3d(down4, 32, (3, 3, 3), (2, 2, 2), 'same', activation=tf.nn.relu)  # [5, 12, 10]
        down5 = tf.layers.conv3d(down5, 32, (3, 3, 3), (1, 1, 1), 'same', activation=tf.nn.relu)
        down5 = tf.layers.batch_normalization(down5, training=self.training)

        down6 = tf.layers.conv3d(down5, 32, (3, 3, 3), (2, 2, 2), 'same', activation=tf.nn.relu)  # [3, 6, 5]
        down6 = tf.layers.conv3d(down6, 32, (3, 3, 3), (1, 1, 1), 'same', activation=tf.nn.relu)
        down6 = tf.layers.batch_normalization(down6, training=self.training)

        down7 = tf.layers.conv3d(down6, 64, (3, 3, 3), (1, 2, 2), 'same', activation=tf.nn.relu)  # [3, 3, 3]
        down7 = tf.layers.conv3d(down7, 64, (3, 3, 3), (1, 1, 1), 'same', activation=tf.nn.relu)
        down7 = tf.layers.batch_normalization(down7, training=self.training)

        latent = tf.layers.conv3d(down7, 64, (3, 3, 3), (1, 1, 1), 'valid', activation=tf.nn.relu)  # [1, 1, 1]
        latent = tf.layers.conv3d(latent, 64, (1, 1, 1), (1, 1, 1), 'valid', activation=tf.nn.relu)
        latent = tf.layers.batch_normalization(latent, training=self.training)

        latent_flat = tf.reshape(latent, [self.batch, 64])  # [batch, 64]

        return latent_flat


    def initialize_weights(self, global_step):

        self.saver = tf.train.Saver(max_to_keep=None)
        init = tf.global_variables_initializer()

        if global_step == 0:
            self.sess.run(init)
        else:
            if global_step == -1 :
                ckpt_list = glob.glob(self.output_path + '/models/model-*.meta')
                epoch_list = []
                for ckpt in ckpt_list:
                    epoch_list.append(int(ckpt.split('/')[-1].split('.')[0].split('-')[-1]))
                epoch_list = sorted(epoch_list)
                global_step = epoch_list[-1]

            print('\n********************************')
            print('Loading model-' + str(global_step))
            print('********************************\n')
            self.saver.restore(self.sess, self.output_path + '/models/model-' + str(global_step))

        self.global_step = global_step


    def freeze(self, output_path, experiment_id, node_names, global_step):

        models_dir = output_path + '/models/'
        state = tf.train.get_checkpoint_state(models_dir)
        path = state.model_checkpoint_path
        path_arr = path.split('-')
        assert(len(path_arr) == 2)

        if global_step == -1:
            input_checkpoint_path = path  # restoring the model from the most recent epoch
        else:
            assert(global_step > 0)
            input_checkpoint_path = path_arr[0] + '-' + str(global_step)  # restoring the model from 'global_step'

        input_graph_path = output_path + '/frozen/' + experiment_id + '_graph_input.pb'
        tf.train.write_graph(self.sess.graph_def, output_path + '/frozen/', experiment_id + '_graph_input.pb', True)
        tf.reset_default_graph()

        input_saver_def_path = ""
        input_binary = False

        print()
        print('Freezing', input_checkpoint_path)
        print()

        output_nodes_string = ''
        for i in range(len(node_names)):
            if i == (len(node_names) - 1):
                output_nodes_string = output_nodes_string + node_names[i]
            else:
                output_nodes_string = output_nodes_string + node_names[i] + ','

        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        output_graph_path = output_path + '/frozen/' + experiment_id + '_graph_output.pb'
        clear_devices = True
        initializer_nodes = ''

        freeze_graph.freeze_graph(
            input_graph_path,
            input_saver_def_path,
            input_binary,
            input_checkpoint_path,
            output_nodes_string,
            restore_op_name,
            filename_tensor_name,
            output_graph_path,
            clear_devices,
            initializer_nodes)
        

class coarse_to_fine(Model):

    def __init__(self, output_path, clean_output, create_summary, gpu):
        super().__init__(output_path, clean_output, create_summary, gpu)

    def define_model(self, seg_thresh, pred_thresh_mm, dist_thresh_dilate):

        self.internal_shape = np.array([96, 256, 256])
        self.internal_pixdim = np.array([0.625, 0.5, 0.5])
        self.voxel_count = self.internal_shape[0] * self.internal_shape[1] * self.internal_shape[2]
        self.voxel_cubic_mm = self.internal_pixdim[0] * self.internal_pixdim[1] * self.internal_pixdim[2]
        self.image_cubic_mm = self.voxel_count * self.voxel_cubic_mm
        self.max_image_val = 1000
        self.seg_thresh = seg_thresh
        self.pred_thresh_mm = pred_thresh_mm
        self.dist_thresh_dilate = dist_thresh_dilate  # Not used (should be verified manually)

        with tf.variable_scope('Input'):

            self.image = tf.placeholder(tf.float32, shape=(None, None, None, None), name='image')
            self.clot_mask = tf.placeholder(tf.float32, shape=(None, None, None, None), name='clot_mask')
            self.vessel_mask_trans = tf.placeholder(tf.float32, shape=(None, None, None, None), name='vessel_mask_trans')
            #self.labels = tf.placeholder(tf.float32, shape=(None), name='labels')

            self.trans_mat = tf.placeholder(tf.float32, shape=(None, 3, 4), name='trans_mat')  # [batch, 3, 4]
            self.training = tf.placeholder(tf.bool, name='training')

        with tf.variable_scope('Preprocessing'):

            self.batch = tf.shape(self.image)[0]

            # image and label to be transformed has to 5 ranked
            self.image_trans = tf.placeholder(tf.float32, shape=(None, None, None, None), name='image')
            self.image_trans = stn3d.spatial_transformer_network(
                #self.image,
                tf.expand_dims(self.image, -1),
                self.trans_mat,
                self.internal_shape,
                'trilinear')

            self.clot_mask_trans = stn3d.spatial_transformer_network(
                tf.expand_dims(self.clot_mask, -1),
                self.trans_mat,
                self.internal_shape,
                'nearest')[:, :, :, :, 0]

            self.image_norm = tf.clip_by_value(self.image_trans, 0, self.max_image_val) / self.max_image_val
        
        #tf.Print(tf.shape(self.image_trans), [tf.shape(self.image_trans)], message="Transformed input image shape: ", summarize=80)
        with tf.variable_scope('classifier'): 
            
            self.image_test = tf.stack([tf.squeeze(self.image_norm, -1), tf.squeeze(self.image_norm, -1)], axis=-1)

            logging.critical(f"Model input rank:{len(self.image_test.get_shape())}")

            assert len(self.image_test.get_shape()) == 5
            self.logits, self.deep_outputs = self.c2f_unet(self.image_test)
            self.logits = self.logits[:, :, :, :, 0]   # [batch, 96, 256, 256]
            self.sigmoid = tf.math.sigmoid(self.logits, name='sigmoid')  # [batch, 96, 256, 256]

            self.softmax = tf.nn.softmax(self.logits, name='softmax_cls')  # [batch, 96, 256, 256, 2]
            self.pred = tf.argmax(self.softmax, axis=-1)  # [batch, 96, 256, 256]

            self.pred_clot = tf.identity(tf.cast(self.sigmoid > self.seg_thresh, tf.float32), name='pred_clot')  # [batch, 96, 256, 256]
        

        with tf.variable_scope('loss'):

            '''
            # Cross Entropy Classification loss
            self.loss0 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.clot_mask_trans, 'int32'))  # [batch, 96, 256, 256]
            self.loss0 = tf.reduce_mean(self.loss0)
            self.loss0_summary_node = tf.summary.scalar('loss0', self.loss0)
            '''

            # Weighted Cross Entropy Classification loss
            pos_weight = 10.0
            print(f"Clot mask rank:{len(self.clot_mask_trans.get_shape())}")
            assert len(self.clot_mask_trans.get_shape()) == 4
            self.loss0 = tf.nn.weighted_cross_entropy_with_logits(logits=self.logits, targets=self.clot_mask_trans, pos_weight=tf.constant(pos_weight))  # [batch, 96, 256, 256]
            self.loss0 = tf.reduce_mean(self.loss0)
            self.loss0_summary_node = tf.summary.scalar('loss0', self.loss0)

            # Deep dice loss 
            logging.debug(f"Clot mask shape:{self.clot_mask_trans.shape}, logits shape:{self.logits.shape}, softmax shape:{self.softmax.shape}")
            self.loss1 = self.deep_dice_bce_loss(true=self.clot_mask_trans, pred=tf.expand_dims(self.softmax, -1), deep_pred=self.deep_outputs)  # [batch]
            self.loss1 = tf.reduce_mean(self.loss1)
            self.loss1_summary_node = tf.summary.scalar('loss1', self.loss1)

            '''
            # Dice loss (foreground)
            self.loss1 = self.dice_loss(true=self.clot_mask_trans, softmax_entry=self.softmax[:, :, :, :, 1])  # [batch]
            self.loss1 = tf.reduce_mean(self.loss1)
            self.loss1_summary_node = tf.summary.scalar('loss1', self.loss1)
            '''

            '''
            # Dice loss (background)
            self.loss2 = self.dice_loss(true=(1-self.clot_mask_trans), softmax_entry=self.softmax[:, :, :, :, 0])  # [batch]
            self.loss2 = tf.reduce_mean(self.loss2)
            self.loss2_summary_node = tf.summary.scalar('loss2', self.loss2)
            '''
            
            '''
            # L2 regression loss
            self.loss0 = (self.clot_map_trans - self.logits) ** 2  # [batch, 80, 192, 160, 1]
            self.loss0 = tf.reduce_sum(self.loss0, [1, 2, 3, 4]) / 2  # [batch]
            self.loss0 = tf.reduce_mean(self.loss0)
            self.loss0_summary_node = tf.summary.scalar('loss0', self.loss0)
            '''

            # Total loss
            self.loss = self.loss0  # + self.loss1 + self.loss2
            self.loss = self.loss0  + self.loss1
            self.loss_summary_node = tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('optimizer'):

            # Batch Normalization
            # Ensures that we execute the update_ops() before performing the train_step
            # This updates the estimated population statistics during training, which is later used during testing
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)
                
    def c2f_unet(self, inputs):

        #assert tf.rank(input) == 5

        self.down_coarse_list = []
        self.up_coarse_list = []
        self.down_fine_list = []
        self.up_fine_list = []

        input_size = inputs.shape
        channels = inputs.shape[-1]
        logging.debug(f"c2f UNet: Input shape:{inputs.shape}")

        logging.debug(f"Coarse Model:")

        x = inputs
        for layer in range(0, 5):
            x = self.cbam_block(x, 'coarse_down_' + str(layer), 2**(layer + 3))
            self.down_coarse_list.append(x)

        for layer in range(3, -1, -1):
            x = self.fusion_block(self.down_coarse_list[layer + 1], 2**(layer + 3), 'coarse_up_'+str(layer))
            self.up_coarse_list.append(x)

        coarse_output = self.fusion_block(tf.concat([self.up_coarse_list[3], self.down_coarse_list[0]], axis=-1), channels, 'coarse_out')

        logging.debug(f"Fine Model:")
        x = inputs
        for layer in range(0, 5):
            x = self.cbam_block(x, 'fine_down_' + str(layer), 2**(layer + 3))
            self.down_fine_list.append(x)

        x = self.down_fine_list[4]
        for layer in range(3, -1, -1):
            y = self.up_coarse_list[3 - layer]
            input_size = self.up_coarse_list[3 - layer].shape
            #y = AveragePooling3D(pool_size=(2, 2, 2))(self.up_coarse_list[3 - layer])
            #y = tf.image.resize_images(self.up_coarse_list[3 - layer], tf.divide(input_size[1:], tf.constant(2)))
            x = self.cnn_tblock(up_coarse=y, up_fine=x, down_fine=self.down_fine_list[layer], filters=2**(layer + 3), name="fine_up_" + str(layer))
            self.up_fine_list.append(tf.layers.conv3d(x, 1, (3, 3, 3), (1, 1, 1), 'same', activation=tf.nn.sigmoid))

        x = tf.layers.conv3d_transpose(x, 1, (3, 3, 3), (2, 2, 2), 'same', activation=tf.nn.relu)
        self.up_fine_list.append(tf.layers.batch_normalization(x))

        logging.critical(f"\nCoarse to fine model built!")
        #model_deep = Model(inputs, [coarse_output, self.up_fine_list[0], self.up_fine_list[1], self.up_fine_list[2], self.up_fine_list[3]])
        #model_deep = Model(inputs, [coarse_output, up_fine1, up_fine2, up_fine3, up_fine4, up_fine5], name="coarse_to_fine_unet")

        #return model_deep, self.up_fine_list[3]
        return coarse_output, self.up_fine_list

    def channel_attention(self, input_feature, name, ratio=1):
    
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        bias_initializer = tf.constant_initializer(value=0.0)
        
        with tf.variable_scope(name):
            
            channel = input_feature.get_shape()[-1]
            #channel = 2

            avg_pool = tf.reduce_mean(input_feature, axis=[1, 2, 3], keepdims=True)
            
            assert avg_pool.get_shape()[1:] == (1,1,1, channel)
            avg_pool = tf.layers.dense(inputs=avg_pool,
                                        units=channel//ratio,
                                        activation=tf.nn.relu,
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer=bias_initializer,
                                        name='mlp_0',
                                        reuse=None)   
            assert avg_pool.get_shape()[1:] == (1,1,1,channel//ratio)
            avg_pool = tf.layers.dense(inputs=avg_pool,
                                        units=channel,                             
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer=bias_initializer,
                                        name='mlp_1',
                                        reuse=None)    
            assert avg_pool.get_shape()[1:] == (1,1,1,channel)

            max_pool = tf.reduce_max(input_feature, axis=[1,2,3], keepdims=True)    
            assert max_pool.get_shape()[1:] == (1,1,1,channel)
            max_pool = tf.layers.dense(inputs=max_pool,
                                        units=channel//ratio,
                                        activation=tf.nn.relu,
                                        name='mlp_0',
                                        reuse=True)   
            assert max_pool.get_shape()[1:] == (1,1,1,channel//ratio)
            max_pool = tf.layers.dense(inputs=max_pool,
                                        units=channel,                             
                                        name='mlp_1',
                                        reuse=True)  
            assert max_pool.get_shape()[1:] == (1,1,1,channel)

            scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')
            
        return input_feature * scale

    def spatial_attention(self, input_feature, filters, name):
        kernel_size = 7
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        with tf.variable_scope(name):
            avg_pool = tf.reduce_mean(input_feature, axis=[-1], keepdims=True)
            assert avg_pool.get_shape()[-1] == 1
            max_pool = tf.reduce_max(input_feature, axis=[-1], keepdims=True)
            assert max_pool.get_shape()[-1] == 1
            concat = tf.concat([avg_pool,max_pool], -1)
            assert concat.get_shape()[-1] == 2
            
            concat = tf.layers.conv3d(concat,
                                    filters=1,
                                    kernel_size=[kernel_size,kernel_size,kernel_size],
                                    strides=[1, 1, 1],
                                    padding="same",
                                    activation=None,
                                    kernel_initializer=kernel_initializer,
                                    use_bias=False,
                                    name='conv')
            
            assert concat.get_shape()[-1] == 1
            concat = tf.sigmoid(concat, 'sigmoid')

            # Not part of standard spatial attention: meant to adjust output size and features
            concat = tf.layers.conv3d(input_feature * concat,
                        filters=filters,
                        kernel_size=[3, 3, 3],
                        strides=[2, 2, 2],
                        padding="same",
                        activation=None,
                        kernel_initializer=kernel_initializer,
                        use_bias=False,
                        name='conv_resize')
            
        return concat 

    def cbam_block(self, input_feature, name, filters, ratio=1):
        """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
        As described in https://arxiv.org/abs/1807.06521.
        """
        
        with tf.variable_scope(name):
            attention_feature = self.channel_attention(input_feature, 'ch_at', ratio)
            attention_feature = self.spatial_attention(attention_feature, filters, 'sp_at')
            #attention_feature = tf.layers.dense(attention_feature, n_units, activation=partial(tf.nn.leaky_relu, alpha=0.01))
            attention_feature = tf.nn.leaky_relu(attention_feature)
        logging.debug(f"c2f UNet: {name}:{attention_feature.shape}")
        return attention_feature

    def fusion_block(self, input_feature, filters, name):
        with tf.variable_scope(name):
            in_filters = input_feature.shape[-1]
            dconv_1 = tf.layers.conv3d(input_feature, filters=in_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding = 'same', activation=tf.nn.relu, dilation_rate=(1, 1, 1))

            dconv_2 = tf.layers.conv3d(input_feature, filters=in_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding = 'same', activation=tf.nn.relu, dilation_rate=(2, 2, 2))

            dconv_4 = tf.layers.conv3d(input_feature, filters=in_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding = 'same', activation=tf.nn.relu, dilation_rate=(4, 4, 4))

            fusion = tf.concat([dconv_1, dconv_2], axis=-1)
            fusion = tf.layers.conv3d(fusion, filters=in_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding = 'same', activation=tf.nn.relu)

            fusion = tf.concat([fusion, dconv_4], axis=-1)
            fusion = tf.layers.conv3d(fusion, filters=in_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding = 'same', activation=tf.nn.relu)            

            fusion = tf.subtract(input_feature, fusion, "fusion_out")

            # Not part of standard spatial attention: meant to adjust output size and features
            fusion = tf.layers.conv3d_transpose(fusion, filters=filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding = 'same', activation=tf.nn.relu)     
            
            logging.debug(f"c2f UNet: {name}:{fusion.shape}")

        return fusion

    def cnn_tblock(self, up_coarse, up_fine, down_fine, filters, name):

        t_filters = up_coarse.shape[-1]
        #logging.debug(f"\nc2f UNet: up_coarse:{up_coarse.shape}")
        #logging.debug(f"c2f UNet: up_fine (x 2 to compare):{up_fine.shape}")
        #logging.debug(f"c2f UNet: down_fine:{down_fine.shape}")
        with tf.variable_scope(name):
            x_fine = tf.layers.conv3d_transpose(up_fine, t_filters, (3, 3, 3), (2, 2, 2), 'same', activation=tf.nn.relu)
            x_fine = tf.layers.batch_normalization(x_fine)

            x_fine = tf.layers.conv3d(tf.concat([up_coarse, down_fine, x_fine], axis=-1), filters, (3, 3, 3), (1, 1, 1), 'same', activation=tf.nn.relu)  
            x_fine = tf.layers.batch_normalization(x_fine)

        logging.debug(f"c2f UNet: {name}:{x_fine.shape}")
        return x_fine

