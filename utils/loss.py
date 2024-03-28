import tensorflow as tf
import numpy as np
from utils.layers import *


# Computing the joint entropy of img1 and img2 based on a grid of int values
# img1/img2 are of shape [crop_height, crop_width, 1] with values in [0,255]
def joint_entropy(img1, img2, crop_shape):

    pixel_count = crop_shape[0] * crop_shape[1]

    concat_img = tf.reshape(tf.concat([img1, img2], axis = -1), [pixel_count, 2])
    scale = tf.constant([0.5], tf.float32)

    # removing half of the pixel values due to a memory issue when creating a 256x256x36864x2 matrix
    resample_factor = 8
    concat_img = tf.reshape(concat_img, [int(pixel_count/resample_factor), resample_factor, 2])
    concat_img = tf.slice(concat_img, [0, 0, 0], [int(pixel_count/resample_factor), 1, 2])
    concat_img = tf.reshape(concat_img, [int(pixel_count/resample_factor), 2])
    pixel_count = int(pixel_count/resample_factor)

    # rounding the intensity values so that the sampling will only occur at the peak of the guassians
    #concat_img = tf.cast(concat_img, tf.int32)
    #concat_img = concat_img - tf.sin(2*np.pi*concat_img) / 2*np.pi

    tfd = tf.contrib.distributions

    gm = tfd.MixtureSameFamily(
        mixture_distribution = tfd.Categorical(
            probs = tf.tile(tf.constant([1 / pixel_count], tf.float32), [pixel_count])),
        components_distribution=tfd.MultivariateNormalDiag(
            loc = concat_img,
            scale_identity_multiplier = scale))

    x_axis = tf.linspace(0., 255., 256)
    y_axis = tf.linspace(0., 255., 256)
    x_grid, y_grid = tf.meshgrid(x_axis, y_axis)
    x_grid = tf.expand_dims(x_grid, -1)
    y_grid = tf.expand_dims(y_grid, -1)
    grid = tf.concat([x_grid, y_grid], -1)

    pdf_samples = gm.prob(grid)

    # can be used to compare with discrete entropy
    #pdf_samples = pdf_samples / tf.reduce_sum(pdf_samples)

    # clipping in order to avoid log(0) = -inf
    entropy = -1 * tf.reduce_sum(pdf_samples * tf.clip_by_value(tf.log(pdf_samples), -1e+10, 1e+10))

    return tf.clip_by_value(entropy, 0, 1e+5)



# An approximation to a lower bound of the joint mutual information of img1 and img2
# img1/img2 are of shape [1, crop_height, crop_width, 1]
def mi_lower_bound(img1, img2, img3, crop_shape, train_mi_est):

    # using the image values as indices for the joint hisotgram
    x = tf.reshape(img1, [crop_shape[0] * crop_shape[1], 1])
    y = tf.reshape(img2, [crop_shape[0] * crop_shape[1], 1])
    y_ = tf.reshape(img3, [crop_shape[0] * crop_shape[1], 1])

    n_hidden = 100

    Wx=tf.Variable(tf.random_normal(stddev=0.1,shape=[1,n_hidden]), train_mi_est)
    Wy=tf.Variable(tf.random_normal(stddev=0.1,shape=[1,n_hidden]), train_mi_est)
    b=tf.Variable(tf.constant(0.1,shape=[n_hidden]), train_mi_est)

    hidden_joint=tf.nn.relu(tf.matmul(x,Wx)+tf.matmul(y,Wy)+b)
    hidden_marg=tf.nn.relu(tf.matmul(x,Wx)+tf.matmul(y_,Wy)+b)

    Wout=tf.Variable(tf.random_normal(stddev=0.1,shape=[n_hidden,1]), train_mi_est)
    bout=tf.Variable(tf.constant(0.1,shape=[1]), train_mi_est)

    out_joint=tf.matmul(hidden_joint,Wout)+bout
    out_marg=tf.matmul(hidden_marg,Wout)+bout

    comp1 = tf.reduce_mean(out_joint)
    #comp2 = tf.log(tf.clip_by_value(tf.reduce_mean(tf.clip_by_value(tf.exp(out_marg), 1e-10, 1e+10)), 1e-10, 1e+10))
    comp2 = tf.log(tf.reduce_mean(tf.exp(out_marg)))
    lower_bound = comp1 - comp2

    return lower_bound
    #train_step = tf.train.AdamOptimizer(0.005).minimize(-lower_bound)


# Zero-Normalized Cross Correlation
def ncc(x, y): 
    mean_x = tf.reduce_mean(x, [1, 2, 3])
    mean_y = tf.reduce_mean(y, [1, 2, 3])
    mean_x2 = tf.reduce_mean(tf.square(x), [1, 2, 3])
    mean_y2 = tf.reduce_mean(tf.square(y), [1, 2, 3])
    std_x = tf.sqrt(mean_x2 - tf.square(mean_x))
    std_y = tf.sqrt(mean_y2 - tf.square(mean_y))
    return tf.reduce_mean(((x - mean_x) * (y - mean_y)) / (std_x * std_y))


# Zero-Normalized Cross Correlation
def zncc(x, y):
    mean_x, var_x = tf.nn.moments(x, [1, 2, 3])
    mean_y, var_y = tf.nn.moments(y, [1, 2, 3])
    x = tf.transpose(x, perm=[1, 2, 3, 0])
    y = tf.transpose(y, perm=[1, 2, 3, 0])
    var_x = tf.where(var_x<=0, tf.zeros_like(var_x)+1e-9, var_x)
    var_y = tf.where(var_y<=0, tf.zeros_like(var_y)+1e-9, var_y)
    den = tf.sqrt(var_x)*tf.sqrt(var_y)
    num = (x - mean_x)*(y - mean_y)
    zncc = tf.reduce_mean(num/den, [0, 1, 2])
    return zncc


def gdl_w(preds, labels, name):

    # Generelized Dice Loss
    # preds [batch, height, width, 4] and labels [batch, height, width] are tf.float32

    with tf.name_scope(name):

        # Prediction
        # Converting preds (shape==[batch, height, width, 4]) into preds_123 (shape==[batch, height, width])
        # The exponent here must be sufficiently large to make the fractions as close to zero as possible
        preds_mask_3 = preds[:,:,:,3]
        preds_mask_2 = preds[:,:,:,2]
        preds_mask_1 = preds[:,:,:,1]
        preds_mask_0 = preds[:,:,:,0]

        # Ground truth
        labels_0123 = labels
        labels_bmask_3 = tf.pow(tf.div(labels_0123, 3.0), 1000000)

        labels_012 = labels_0123 - (labels_bmask_3 * 3.0)
        labels_bmask_2 = tf.pow(tf.div(labels_012, 2.0), 1000000)

        labels_bmask_1 = labels_012 - (labels_bmask_2 * 2.0)

        labels_bmask_0 = 1 - (labels_bmask_3 + labels_bmask_2 + labels_bmask_1)

        inter_0 = tf.reduce_sum(tf.multiply(preds_mask_0, labels_bmask_0), [1,2])
        inter_1 = tf.reduce_sum(tf.multiply(preds_mask_1, labels_bmask_1), [1,2])
        inter_2 = tf.reduce_sum(tf.multiply(preds_mask_2, labels_bmask_2), [1,2])
        inter_3 = tf.reduce_sum(tf.multiply(preds_mask_3, labels_bmask_3), [1,2])

        w_0 = tf.pow((1 / (1 + tf.reduce_sum(labels_bmask_0, [1,2]))), 2)
        w_1 = tf.pow((1 / (1 + tf.reduce_sum(labels_bmask_1, [1,2]))), 2)
        w_2 = tf.pow((1 / (1 + tf.reduce_sum(labels_bmask_2, [1,2]))), 2)
        w_3 = tf.pow((1 / (1 + tf.reduce_sum(labels_bmask_3, [1,2]))), 2)

        numerator = (w_0 * inter_0) + (w_1 * inter_1) + (w_2 * inter_2) + (w_3 * inter_3)

        add_0 = tf.add(tf.reduce_sum(preds_mask_0, [1,2]), tf.reduce_sum(labels_bmask_0, [1,2]))
        add_1 = tf.add(tf.reduce_sum(preds_mask_1, [1,2]), tf.reduce_sum(labels_bmask_1, [1,2]))
        add_2 = tf.add(tf.reduce_sum(preds_mask_2, [1,2]), tf.reduce_sum(labels_bmask_2, [1,2]))
        add_3 = tf.add(tf.reduce_sum(preds_mask_3, [1,2]), tf.reduce_sum(labels_bmask_3, [1,2]))

        denominator = (w_0 * add_0) + (w_1 * add_1) + (w_2 * add_2) + (w_3 * add_3)

        result = 1 - ((2.0 * numerator) / denominator)

        # result is a 1D tensor of size batch_size
        return result

def gdl(preds, labels, name):

    # Generelized Dice Loss
    # preds [batch, height, width, 4] and labels [batch, height, width] are tf.float32

    with tf.name_scope(name):

        # Prediction

        # Converting preds (shape==[batch, height, width, 4]) into preds_123 (shape==[batch, height, width])
        # The exponent here must be sufficiently large to make the fractions as close to zero as possible
        preds_mask_3 = preds[:,:,:,3]
        preds_mask_2 = preds[:,:,:,2]
        preds_mask_1 = preds[:,:,:,1]

        # Ground truth
        labels_123 = labels
        labels_bmask_3 = tf.pow(tf.div(labels_123, 3.0), 1000000)

        labels_12 = labels_123 - (labels_bmask_3 * 3.0)
        labels_bmask_2 = tf.pow(tf.div(labels_12, 2.0), 1000000)

        labels_1 = labels_12 - (labels_bmask_2 * 2.0)
        labels_bmask_1 = labels_1

        inter_1 = tf.reduce_sum(tf.multiply(preds_mask_1, labels_bmask_1), [1,2])
        inter_2 = tf.reduce_sum(tf.multiply(preds_mask_2, labels_bmask_2), [1,2])
        inter_3 = tf.reduce_sum(tf.multiply(preds_mask_3, labels_bmask_3), [1,2])

        numerator_1 = tf.multiply(2.0, inter_1)
        numerator_2 = tf.multiply(2.0, inter_2)
        numerator_3 = tf.multiply(2.0, inter_3)

        denominator_1 = tf.add(tf.reduce_sum(preds_mask_1, [1,2]), tf.reduce_sum(labels_bmask_1, [1,2]))
        denominator_2 = tf.add(tf.reduce_sum(preds_mask_2, [1,2]), tf.reduce_sum(labels_bmask_2, [1,2]))
        denominator_3 = tf.add(tf.reduce_sum(preds_mask_3, [1,2]), tf.reduce_sum(labels_bmask_3, [1,2]))

        epsilon = 0.0001
        # When the denominator is 0, the numerator is also 0 and the result will be 0
        # If it's not, the epsilon is inssugnificant

        class_1 = tf.div(numerator_1, denominator_1 + epsilon)
        class_2 = tf.div(numerator_2, denominator_2 + epsilon)
        class_3 = tf.div(numerator_3, denominator_3 + epsilon)

        result = 3 - (class_1 + class_2 + class_3)

        # result is a 1D tensor of size batch_size
        return result

def dice_loss(preds, labels, name):

    # preds [batch, height, width, 4] and labels [batch, height, width] are tf.float32

    with tf.name_scope(name):

        # Prediction

        # Converting preds (shape==[batch, height, width, 4]) into preds_123 (shape==[batch, height, width])
        # The exponent here must be sufficiently large to make the fractions as close to zero as possible
        preds_123 = tf.reduce_max(tf.add(tf.multiply(tf.pow(tf.div(preds, tf.reduce_max(preds, [3], keep_dims=True)), 1000000), [1.0,2.0,3.0,4.0]), -1), [3])
        preds_bmask_3 = tf.pow(tf.div(preds_123, 3.0), 1000000)

        preds_12 = preds_123 - (preds_bmask_3 * 3.0)
        preds_bmask_2 = tf.pow(tf.div(preds_12, 2.0), 1000000)

        preds_1 = preds_12 - (preds_bmask_2 * 2.0)
        preds_bmask_1 = preds_1

        # Ground truth
        labels_123 = labels
        labels_bmask_3 = tf.pow(tf.div(labels_123, 3.0), 1000000)

        labels_12 = labels_123 - (labels_bmask_3 * 3.0)
        labels_bmask_2 = tf.pow(tf.div(labels_12, 2.0), 1000000)

        labels_1 = labels_12 - (labels_bmask_2 * 2.0)
        labels_bmask_1 = labels_1

        inter_1 = tf.reduce_sum(tf.multiply(preds_bmask_1, labels_bmask_1), [1,2])
        inter_2 = tf.reduce_sum(tf.multiply(preds_bmask_2, labels_bmask_2), [1,2])
        inter_3 = tf.reduce_sum(tf.multiply(preds_bmask_3, labels_bmask_3), [1,2])

        numerator_1 = tf.multiply(2.0, inter_1)
        numerator_2 = tf.multiply(2.0, inter_2)
        numerator_3 = tf.multiply(2.0, inter_3)

        denominator_1 = tf.add(tf.reduce_sum(preds_bmask_1, [1,2]), tf.reduce_sum(labels_bmask_1, [1,2]))
        denominator_2 = tf.add(tf.reduce_sum(preds_bmask_2, [1,2]), tf.reduce_sum(labels_bmask_2, [1,2]))
        denominator_3 = tf.add(tf.reduce_sum(preds_bmask_3, [1,2]), tf.reduce_sum(labels_bmask_3, [1,2]))

        epsilon = 0.0001
        # When the denominator is 0, the numerator is also 0 and the result will be 0
        # If it's not, the epsilon is inssugnificant

        class_1 = tf.div(numerator_1, denominator_1 + epsilon)
        class_2 = tf.div(numerator_2, denominator_2 + epsilon)
        class_3 = tf.div(numerator_3, denominator_3 + epsilon)

        result = 3 - (class_1 + class_2 + class_3)

        # result is a 1D tensor of size batch_size
        return result


def mdsc_loss(preds, labels, name):

    # preds [batch, height, width, 4] and labels [batch, height, width] are tf.float32

    with tf.name_scope(name):

        # Prediction

        # Converting preds (shape==[batch, height, width, 4]) into preds_123 (shape==[batch, height, width])
        # The exponent here must be sufficiently large to make the fractions as close to zero as possible
        preds_123 = tf.reduce_max(tf.add(tf.multiply(tf.pow(tf.div(preds, tf.reduce_max(preds, [3], keep_dims=True)), 1000000), [1.0,2.0,3.0,4.0]), -1), [3])
        preds_bmask_3 = tf.pow(tf.div(preds_123, 3.0), 1000000)

        preds_12 = preds_123 - (preds_bmask_3 * 3.0)
        preds_bmask_2 = tf.pow(tf.div(preds_12, 2.0), 1000000)

        preds_1 = preds_12 - (preds_bmask_2 * 2.0)
        preds_bmask_1 = preds_1

        # Ground truth
        labels_123 = labels
        labels_bmask_3 = tf.pow(tf.div(labels_123, 3.0), 1000000)

        labels_12 = labels_123 - (labels_bmask_3 * 3.0)
        labels_bmask_2 = tf.pow(tf.div(labels_12, 2.0), 1000000)

        labels_1 = labels_12 - (labels_bmask_2 * 2.0)
        labels_bmask_1 = labels_1

        # Merged = Prediction + Ground truth
        preds_sum = preds_bmask_3 + preds_bmask_2 + preds_bmask_1
        labels_sum = labels_bmask_3 + labels_bmask_2 + labels_bmask_1

        merged_12 = preds_sum + labels_sum
        merged_bmask_2 = tf.pow(tf.div(merged_12, 2.0), 1000000)

        merged_1 = merged_12 - (merged_bmask_2 * 2.0)
        merged_bmask_1 = merged_1

        N = tf.reduce_sum(tf.add(merged_bmask_2, merged_bmask_1), [1,2])

        inter_1 = tf.reduce_sum(tf.multiply(preds_bmask_1, labels_bmask_1), [1,2])
        inter_2 = tf.reduce_sum(tf.multiply(preds_bmask_2, labels_bmask_2), [1,2])
        inter_3 = tf.reduce_sum(tf.multiply(preds_bmask_3, labels_bmask_3), [1,2])

        numerator_1 = tf.div(tf.multiply(2.0, inter_1), N)
        numerator_2 = tf.div(tf.multiply(2.0, inter_2), N)
        numerator_3 = tf.div(tf.multiply(2.0, inter_3), N)

        denominator_1 = tf.add(tf.reduce_sum(preds_bmask_1, [1,2]), tf.reduce_sum(labels_bmask_1, [1,2]))
        denominator_2 = tf.add(tf.reduce_sum(preds_bmask_2, [1,2]), tf.reduce_sum(labels_bmask_2, [1,2]))
        denominator_3 = tf.add(tf.reduce_sum(preds_bmask_3, [1,2]), tf.reduce_sum(labels_bmask_3, [1,2]))

        class_1 = tf.div(numerator_1, denominator_1)
        class_2 = tf.div(numerator_2, denominator_2)
        class_3 = tf.div(numerator_3, denominator_3)

        result = tf.multiply(tf.add(tf.add(class_1, class_2), class_3), -1)

        '''
        N = tf.cast(tf.count_nonzero(tf.add(tf.cast(tf.greater(preds, 0), tf.float32), tf.cast(tf.greater(labels, 0), tf.float32)), [1,2]), tf.float32)

        preds_eq_1 = tf.cast(tf.equal(preds, 1), tf.float32)
        preds_eq_2 = tf.cast(tf.equal(preds, 2), tf.float32)
        preds_eq_3 = tf.cast(tf.equal(preds, 3), tf.float32)

        labels_eq_1 = tf.cast(tf.equal(labels, 1), tf.float32)
        labels_eq_2 = tf.cast(tf.equal(labels, 2), tf.float32)
        labels_eq_3 = tf.cast(tf.equal(labels, 3), tf.float32)

        inter_1 = tf.cast(tf.count_nonzero(tf.cast(tf.equal(preds_eq_1, labels_eq_1), tf.float32), [1,2]), tf.float32)
        inter_2 = tf.cast(tf.count_nonzero(tf.cast(tf.equal(preds_eq_2, labels_eq_2), tf.float32), [1,2]), tf.float32)
        inter_3 = tf.cast(tf.count_nonzero(tf.cast(tf.equal(preds_eq_3, labels_eq_3), tf.float32), [1,2]), tf.float32)

        numerator_1 = tf.div(tf.multiply(2.0, inter_1), N)
        numerator_2 = tf.div(tf.multiply(2.0, inter_2), N)
        numerator_3 = tf.div(tf.multiply(2.0, inter_3), N)

        denominator_1 = tf.cast(tf.add(tf.count_nonzero(preds_eq_1, [1,2]), tf.count_nonzero(labels_eq_1, [1,2])), tf.float32)
        denominator_2 = tf.cast(tf.add(tf.count_nonzero(preds_eq_2, [1,2]), tf.count_nonzero(labels_eq_2, [1,2])), tf.float32)
        denominator_3 = tf.cast(tf.add(tf.count_nonzero(preds_eq_3, [1,2]), tf.count_nonzero(labels_eq_3, [1,2])), tf.float32)

        class_1 = tf.div(numerator_1, denominator_1)
        class_2 = tf.div(numerator_2, denominator_2)
        class_3 = tf.div(numerator_3, denominator_3)

        result = tf.multiply(tf.add(tf.add(class_1, class_2), class_3), -1)
        '''

        # result is a 1D tensor of size batch_size
        return result

    def euclidian_distance(preds, labels):
        return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(preds, labels))))
