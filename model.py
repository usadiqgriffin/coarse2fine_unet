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
from tqdm import tqdm
from datetime import datetime
import glob
import logging
from tensorflow.keras.layers import UpSampling3D, AveragePooling3D

def tensor_print(x, text):
    tf.compat.v1.Print(x, [x], message=text + ": ", summarize=80)
class Model(object):


    def deep_dice_bce_loss(self, true, pred, deep_pred, smooth=1e-5):
        
        label_0 = 0
        def dice_loss(true, pred):
            true = tf.cast(true, tf.float32)
            pred = tf.cast(pred, tf.float32)
            #logging.debug(f"True shape:{true.shape}, pred shape:{pred.shape}")
            num = tf.reduce_sum(true * pred, axis=[1, 2, 3]) + smooth
            den = tf.reduce_sum(true, axis=[1, 2, 3]) + tf.reduce_sum(pred, axis=[1, 2, 3]) + smooth
            loss = -(num / den)
            return loss

        def call(true, pred):
            
            logging.debug(f"True shape:{true.shape}, pred shape:{pred.shape}, deep pred shape:{len(deep_pred)}")
            logging.debug(f"Deep pred shapes:({deep_pred[0].shape}), ({deep_pred[1].shape}), ({deep_pred[2].shape})")

            true = tf.cast(tf.equal(true, label_0 * tf.ones_like(true)), tf.float32)
            pred = tf.cast(pred[..., label_0], tf.float32)
            
            foreground_dice_loss = dice_loss(true, pred)
            background_dice_loss = dice_loss(1 - true, 1 - pred)
            bce = tf.keras.losses.binary_crossentropy(true, pred, from_logits=False)
            bce = tf.reduce_mean(bce)
            loss = foreground_dice_loss + background_dice_loss + bce
            return loss
            
        return call(true, pred)
    
    def dice_bce_loss(true, pred, label_0, smooth=1e-7):
        
        def dice_loss(true, pred):
            true = tf.cast(true, tf.float32)
            pred = tf.cast(pred, tf.float32)
            num = tf.reduce_sum(true * pred, axis=[1, 2, 3]) + smooth
            den = tf.reduce_sum(true, axis=[1, 2, 3]) + tf.reduce_sum(pred, axis=[1, 2, 3]) + smooth
            loss = -(num / den)
            return loss

        def call(true, pred):
            
            true = tf.cast(tf.equal(true, label_0 * tf.ones_like(true)), tf.float32)
            pred = tf.cast(pred[..., label_0], tf.float32)
            
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

            self.saver.restore(self.sess, self.output_path + '/models/model-' + str(global_step))

        self.global_step = global_step
       
class coarse_to_fine(Model):

    def __init__(self, output_path, clean_output, create_summary, gpu):
        super().__init__(output_path, clean_output, create_summary, gpu)

        with tf.variable_scope('classifier'): 
            
            self.image_test = tf.stack([tf.squeeze(self.image_norm, -1), tf.squeeze(self.image_norm, -1)], axis=-1)

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

