# tf layers to be used as a front end for tensorflow

import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow_addons import rnn

import utils.stn3d as stn3d
import utils.stn as stn

############################### General Operations #############################


def extract_image_patches(img, patch_size, patch_stride, patch_count):

    ksizes = [1, patch_size, patch_size, 1]
    strides = [1, patch_stride, patch_stride, 1]
    rates = [1, 1, 1, 1]

    # e.g. [1, 192, 192, 1] => [1, 8, 8, 32*32]
    img_patches = tf.extract_image_patches(img, ksizes, strides, rates, "SAME")
    # e.g. [1, 8, 8, 32*32] => [64, 32, 32, 1]
    img_patches = tf.reshape(
        img_patches, [patch_count, patch_size, patch_size, 1])

    return img_patches


def get_transform(shape_in, rotate_degree, shift_y, shift_x):
    """tensorflow implementation of getting the transformation matrix."""

    #shift = tf.Variable([shift_y, shift_x], dtype=tf.float32, trainable=False, name='var_a')
    shift = [shift_y, shift_x]

    shape_in = tf.cast(tf.reshape(shape_in, [1, 2]), tf.float32)
    centre_in = 0.5 * shape_in
    centre_out = centre_in

    theta = rotate_degree * np.pi / 180

    #transform = tf.Variable([[tf.cos(theta),-tf.sin(theta)],[tf.sin(theta),tf.cos(theta)]], dtype=tf.float32, trainable=False, name='var_b')
    transform = [[tf.cos(theta), -tf.sin(theta)],
                 [tf.sin(theta), tf.cos(theta)]]

    offset = centre_in - tf.matmul(centre_out, transform) + shift

    trans_vec = [transform[1][1],
                 transform[0][1],
                 offset[0][1],
                 transform[1][0],
                 transform[0][0],
                 offset[0][0],
                 tf.constant(0, tf.float32),
                 tf.constant(0, tf.float32)]

    return trans_vec


# x is of shape (None, None, None, 1)
def preprocess_img(x, canvas_shape, trans, crop_shape, name='preprocess_img', hist=True):

    # Normalizing the range to [0,255]
    #x_norm = tf.cond(tf.reduce_sum(tf.reshape(x,[-1])) > 0, lambda: norm_range(x), lambda: x)
    x_norm = norm_range(x)

    if hist:
        # Histogram Equalization
        x_hist = hist_equal(x_norm)
    else:
        x_hist = x_norm

    # Affine Transformation
    x_cropped = affine_transform(
        x_hist, canvas_shape, trans, 'BILINEAR', crop_shape)

    return tf.identity(x_cropped, name)

# x is of shape (None, None, None, 1)
def preprocess_img_stn(x, trans, crop_shape, name='preprocess_img', hist=True):

    # Normalizing the range to [0,255]
    x_norm = norm_range(x)

    if hist:
        # Histogram Equalization
        with tf.variable_scope('Histogram'):
            x_hist = hist_equal(x_norm)
    else:
        x_hist = x_norm

    # Affine Transformation
    with tf.variable_scope('Transformer'):
        with tf.control_dependencies([x_hist]):
            x_cropped = stn.spatial_transformer_network(x_hist, trans, inter='bilinear', out_dims=crop_shape)

    return tf.identity(x_cropped, name)

# y_seg is of shape (None, None, None) representing (batch, height, width)
def preprocess_seg(y_seg, canvas_shape, trans, crop_shape, name='preprocess_seg'):
    y_seg_cropped = affine_transform(tf.expand_dims(
        y_seg, -1), canvas_shape, trans, 'NEAREST', crop_shape)
    return tf.identity(tf.squeeze(y_seg_cropped, -1), name)

# y_seg is of shape (None, None, None) representing (batch, height, width)
def preprocess_seg_stn(y_seg, trans, crop_shape, name='preprocess_seg'):
    y_seg_cropped = stn.spatial_transformer_network(
        tf.expand_dims(y_seg, -1), trans, inter='nearest', out_dims=crop_shape)
    return tf.identity(tf.squeeze(y_seg_cropped, -1), name)


# x is of shape (None, None, None, None, 1)
def preprocess_img_3d(x, trans, crop_shape, name='preprocess_img_3d'):
    # Normalizing the range to [0,255]
    x_norm = norm_range3d(x)

    # Affine Transformation
    x_cropped = stn3d.spatial_transformer_network(x_norm, trans, crop_shape)

    return tf.identity(x_cropped, name)


def preprocess_seg_3d(y_seg, trans, crop_shape, name='preprocess_seg_3d'):
    y_seg = tf.expand_dims(y_seg, -1)
    y_seg_cropped = stn3d.spatial_transformer_network(y_seg, trans, out_dims=crop_shape, inter='nearest')
    y_seg_cropped = tf.squeeze(y_seg_cropped, -1)
    return tf.identity(y_seg_cropped, name)


# y_reg is of shape (None, reg_out_size) containing indices of x
def preprocess_reg(y_reg, src_shape, canvas_shape, trans, crop_shape, name):
    y_reg_cropped = affine_transform_reg(
        y_reg, src_shape, canvas_shape, trans, crop_shape)
    y_reg_cropped = tf.divide(y_reg_cropped, tf.tile(
        crop_shape, [tf.div(tf.shape(y_reg)[1], 2)]))
    return tf.identity(y_reg_cropped, name)

# y_reg is of shape (None, reg_out_size) containing indices of x
# trans: (B,2,3)
def preprocess_reg_stn(y_reg, src_shape, trans, crop_shape, name):
    y_reg_shape = tf.shape(y_reg)
    num_batch = y_reg_shape[0]
    max_y = tf.cast(crop_shape[0]-1, 'float32')
    max_x = tf.cast(crop_shape[1]-1, 'float32')
    zero = tf.zeros([], dtype='float32')
    # y_reg: (B,N,2)
    y_reg = tf.reshape(y_reg, [-1, tf.div(y_reg_shape[1], 2), 2])

    # bring to normalize space
    y_reg = normalize_reg_points(y_reg, src_shape)
    ones = tf.ones_like(y_reg[:,:,1:])
    y_reg_full = tf.concat([y_reg, ones], axis=-1)

    # trans_full: (B,3,3)
    pad = tf.tile(tf.expand_dims(tf.constant([[0, 0, 1]], dtype=tf.float32), axis=0), (num_batch, 1, 1))
    trans_full = tf.concat([trans, pad], axis=1)

    # inverse solver
    y_reg_cropped_full = tf.linalg.solve(trans_full, tf.transpose(y_reg_full, [0, 2, 1]))
    y_reg_cropped_full = tf.transpose(y_reg_cropped_full, [0, 2, 1])

    # bring back to crop space
    y_reg_cropped = y_reg_cropped_full[:,:,:2]
    y_reg_cropped = normalize_reg_points(y_reg_cropped, crop_shape, reverse=True)

    # clipping value when out of range
    y = y_reg_cropped[:, :, 0]
    x = y_reg_cropped[:, :, 1]
    y = tf.clip_by_value(y, zero, max_y)
    x = tf.clip_by_value(x, zero, max_x)
    y_reg_cropped_clip = tf.stack([y, x], axis=-1)

    y_reg_cropped_clip = tf.reshape(y_reg_cropped_clip, [-1, y_reg_shape[1]])
    return tf.identity(y_reg_cropped_clip, name)

def normalize_reg_points(points, axes_size, reverse=False):
    if not reverse:
        out = (2 * (points / (axes_size - 1))) - 1
    else:
        out = tf.round(((points + 1) / 2) * (axes_size - 1))

    return out

def norm_range3d(x):
    min_vals = tf.reduce_min(x, [1, 2, 3, 4], keepdims=True)
    max_vals = tf.reduce_max(x, [1, 2, 3, 4], keepdims=True)
    zeros = tf.zeros_like(x)
    num = (x - min_vals)
    den = zeros+(max_vals - min_vals)
    norm = tf.where(tf.equal(den, 0.), zeros, num/den*255)
    return norm


def norm_range(x):
    min_vals = tf.reduce_min(x, [1, 2, 3], keepdims=True)
    max_vals = tf.reduce_max(x, [1, 2, 3], keepdims=True)
    zeros = tf.zeros_like(x)
    num = (x - min_vals)
    den = zeros+(max_vals - min_vals)
    norm = tf.where(tf.equal(den, 0.), zeros, num/den*255)
    return norm


def loop_body(i, x):
    x = tf.concat([x, tf.expand_dims(hist_equal_single(x[0]), 0)], axis=0)
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    i = tf.add(i, 1)
    return i, x


def loop_condition(i, x):
    return i < tf.shape(x)[0]


def hist_equal(x):
    # x is of shape (None, None, None, 1)
    i = tf.constant(0)
    return tf.while_loop(loop_condition, loop_body, [i, x])[1]


def hist_equal_single(img):
    # img is of shape (None, None, 1) i.e. a single greyscale image
    values_range = tf.constant([0, 255], dtype=tf.float32)
    histogram = tf.histogram_fixed_width(tf.to_float(img), values_range, 256)
    cdf = tf.cumsum(histogram)
    cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]

    img_shape = tf.shape(img)
    pix_cnt = img_shape[-3] * img_shape[-2]
    px_map = tf.round(tf.to_float(cdf - cdf_min) *
                      255 / tf.to_float(pix_cnt - 1))
    px_map = tf.cast(px_map, tf.uint8)

    eq_hist = tf.expand_dims(tf.gather_nd(px_map, tf.cast(img, tf.int32)), -1)
    return tf.cast(eq_hist, tf.float32)


def affine_transform(x, canvas_shape, trans, inter, target_shape):
    # x is a Tensor of shape (None, None, None, None)
    canvas_shape = tf.cast(canvas_shape, tf.int32)
    target_shape = tf.cast(target_shape, tf.int32)
    x_resized = tf.image.resize_image_with_crop_or_pad(
        x, canvas_shape[0], canvas_shape[1])
    x_transformed = tf.contrib.image.transform(x_resized, trans, inter)
    x_cropped = tf.image.resize_image_with_crop_or_pad(
        x_transformed, target_shape[0], target_shape[1])

    return x_cropped


def affine_transform_reg(y_reg, source_shape, canvas_shape, trans, target_shape):
    # y_reg is a Tensor of shape (None, reg_out_size)
    # each entry is of form [y1,x1,y2,x2,y3,x3,....]  containing indices of x
    # return indices of x (not ratios)

    transform = tf.reshape(tf.gather(trans, [4, 1, 3, 0]), [2, 2])
    offset = tf.gather(trans, [5, 2])

    y_reg_shape = tf.shape(y_reg)
    y_reg = tf.reshape(y_reg, [-1, tf.div(y_reg_shape[1], 2), 2])

    y_reg = adjust_points(y_reg, source_shape, canvas_shape)
    inv_transform = tf.matrix_inverse(transform)
    inv_transform = tf.tile(tf.expand_dims(
        inv_transform, 0), [y_reg_shape[0], 1, 1])
    y_reg = tf.matmul((y_reg - offset), inv_transform)
    y_reg = adjust_points(y_reg, canvas_shape, target_shape)

    y_reg = tf.reshape(y_reg, [-1, y_reg_shape[1]])

    return y_reg


def adjust_points(y_reg, shape_in, shape_out):
    # broadcasting is applied
    return (y_reg + ((shape_out - shape_in) / 2))


def batch_normalization(x, training, trainable=True, fused=None):
    x = tf.layers.batch_normalization(inputs=x, training=training, trainable=trainable, fused=fused)
    return x

def layer_normalization(x, training, trainable=True, fused=None):
    x = tf.layers.layer_normalization(inputs=x, training=training, trainable=trainable, fused=fused)
    return x

def batch_normalization_scope(x, training, trainable=True, name=None, reuse=False):
    with tf.variable_scope(name, default_name='batchnorm', reuse=reuse):
        return tf.layers.batch_normalization(inputs=x, training=training, trainable=trainable)

def dropout(x, drop_rate, training):
    return tf.layers.dropout(inputs=x, rate=drop_rate, training=training)


def activation(x, function, name=None):
    try:
        F = getattr(tf.nn, function)
    except:
        raise Exception("Unknown activation ({}).".format(function))
    if name is not None:
        return F(x, name=name)
    else:
        return F(x)


def concatenate(x, y, axis=-1):
    return tf.concat([x, y], axis)


def flatten(input):
    # Get the shape of the input layer.
    layer_shape = tf.shape(input)

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    num_features = layer_shape[1] + layer_shape[2] + layer_shape[3]
    #num_features = tf.reduce_sum(tf.gather(layer_shape, [1,2,3]))
    #num_features = layer_shape[-3] + layer_shape[-2] + layer_shape[-1]

    # Reshape the input to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(input, [-1, num_features])

    # The shape of the flattened input is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened input and the number of features.
    return layer_flat, num_features


def fully_connect(input, num_outputs, non_linearity, name=None, trainable=True):

    if non_linearity is None:
        layer = tf.contrib.layers.fully_connected(
            input, num_outputs, None, trainable=trainable)
    elif non_linearity == 'relu':
        layer = tf.contrib.layers.fully_connected(
            input, num_outputs, tf.nn.relu, trainable=trainable)
    elif non_linearity == 'sigmoid':
        layer = tf.contrib.layers.fully_connected(
            input, num_outputs, tf.nn.sigmoid, trainable=trainable)

    return tf.identity(layer, name)


################################# 2D Operations ################################


def conv2d_scope(x, filters, kernel_size, padding='same', dilation_rate=1, activation=None, trainable=True, strides=(1, 1), name=None, reuse=False):
    with tf.variable_scope(name, default_name='conv', reuse=reuse):
        return tf.layers.conv2d(
            inputs=x,
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            dilation_rate=dilation_rate,
            activation=activation,
            trainable=trainable,
            strides=strides)


def conv2d(x, filters, kernel_size, padding='same', dilation_rate=1, activation=None, trainable=True, strides=(1, 1)):
    return tf.layers.conv2d(
        inputs=x,
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        dilation_rate=dilation_rate,
        activation=activation,
        trainable=trainable,
        strides=strides)


def conv2d_transpose(x, filters, kernel_size, strides=(2, 2), padding='valid', activation=None):
    return tf.layers.conv2d_transpose(
        inputs=x,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation)


def max_pooling2d(x, pool_size, strides=2):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=strides)


def avg_pooling2d(x, pool_size, strides=2):
    return tf.layers.AveragePooling2D(pool_size=pool_size, strides=strides)(x)


def cropping2d(x, y):
    x_shape = tf.shape(x)
    y_shape = tf.shape(y)
    offsets = [0,
               (x_shape[1] - y_shape[1]) // 2,
               (x_shape[2] - y_shape[2]) // 2,
               0]
    size = [-1, y_shape[1], y_shape[2], -1]
    return tf.slice(x, offsets, size)


def down_op(input, filters, name, norm_type, training, padding='same', pool_type='max'):
    with tf.name_scope(name):
        x = conv2d(input, filters, kernel_size=(3, 3), padding=padding)
        if norm_type == 'batch':
            x = batch_normalization(x, training)
        elif norm_type == 'instance':
            x = tf.contrib.layers.instance_norm(x)
        x = activation(x, 'relu')
        if pool_type == 'max':
            return max_pooling2d(x, pool_size=(2, 2)), x
        elif pool_type == 'avg':
            return avg_pooling2d(x, pool_size=(2, 2)), x


def down_block_nonorm(input_tensor, filters, training, name, drop_rate=0.0, padding='same', pool_type='max'):
    with tf.name_scope(name):
        x = conv2d(input_tensor, filters, kernel_size=(3, 3), padding=padding)
        x = activation(x, 'relu')
        x = dropout(x, drop_rate, training)
        x = conv2d(x, filters, kernel_size=(3, 3), padding=padding)
        x = activation(x, 'relu')
        x = dropout(x, drop_rate, training)
        if pool_type == 'max':
            return max_pooling2d(x, pool_size=(2, 2)), x
        elif pool_type == 'avg':
            return avg_pooling2d(x, pool_size=(2, 2)), x
        else:
            print('unsupported pooling type')
            exit()


def down_block(input_tensor, filters, training, name, drop_rate=0.0, padding='same', pool_type='max'):
    with tf.name_scope(name):
        x = conv2d(input_tensor, filters, kernel_size=(3, 3), padding=padding)
        x = batch_normalization(x, training)
        x = activation(x, 'relu')
        x = dropout(x, drop_rate, training)
        x = conv2d(x, filters, kernel_size=(3, 3), padding=padding)
        x = batch_normalization(x, training)
        x = activation(x, 'relu')
        x = dropout(x, drop_rate, training)
        if pool_type == 'max':
            return max_pooling2d(x, pool_size=(2, 2)), x
        elif pool_type == 'avg':
            return avg_pooling2d(x, pool_size=(2, 2)), x
        else:
            print('unsupported pooling type')
            exit()


def up_block_nonorm(input_tensor, skip_tensor, filters, training,
                    name, drop_rate=0.0, padding='same'):
    with tf.name_scope(name):
        x = conv2d_transpose(input_tensor, filters, kernel_size=(2, 2))
        x = concatenate(x, skip_tensor)
        x = conv2d(x, filters, kernel_size=(3, 3), padding=padding)
        x = activation(x, 'relu')
        x = dropout(x, drop_rate, training)
        x = conv2d(x, filters, kernel_size=(3, 3), padding=padding)
        x = activation(x, 'relu')
        x = dropout(x, drop_rate, training)
        return x


def up_block(input_tensor, skip_tensor, filters, training,
             name, drop_rate=0.0, padding='same'):
    with tf.name_scope(name):
        x = conv2d_transpose(input_tensor, filters, kernel_size=(2, 2))
        x = concatenate(x, skip_tensor)
        x = conv2d(x, filters, kernel_size=(3, 3), padding=padding)
        x = batch_normalization(x, training)
        x = activation(x, 'relu')
        x = dropout(x, drop_rate, training)
        x = conv2d(x, filters, kernel_size=(3, 3), padding=padding)
        x = batch_normalization(x, training)
        x = activation(x, 'relu')
        x = dropout(x, drop_rate, training)
        return x


def unet(input_tensor, training, kernel_size=(3, 3), filters=8, layers=2, depth=6,
         drop_rate=0.0, name='unet'):
    with tf.variable_scope(name):
        net = {}
        x = input_tensor
        for d in range(depth):
            with tf.variable_scope('down_{}'.format(d+1)):
                strides = 1 if d == 0 else 2
                x = conv2d(x, filters << d, kernel_size, strides=strides)
                x = batch_normalization(x, training)
                x = activation(x, 'relu')
                x = dropout(x, drop_rate, training)
                for l in range(layers-1):
                    x = conv2d(x, filters << d, kernel_size)
                    x = batch_normalization(x, training)
                    x = activation(x, 'relu')
                    x = dropout(x, drop_rate, training)
                net['down_{}'.format(d+1)] = x

        with tf.variable_scope('up_{}'.format(depth)):
            net['up_{}'.format(depth)] = net['down_{}'.format(depth)]

        for d in range(depth-1, 0, -1):
            with tf.variable_scope('up_{}'.format(d)):
                x = conv2d_transpose(net['up_{}'.format(d+1)], filters <<
                                     (d-1), kernel_size, padding='same')
                x = batch_normalization(x, training)
                x = activation(x, 'relu')
                x = concatenate(x, net['down_{}'.format(d)])
                x = dropout(x, drop_rate, training)
                for l in range(layers-1):
                    x = conv2d(x, filters << (d-1), kernel_size)
                    x = batch_normalization(x, training)
                    x = activation(x, 'relu')
                    x = dropout(x, drop_rate, training)
                net['up_{}'.format(d)] = x

        return net['up_1']


def unet_nobn(input_tensor, kernel_size=(3, 3), filters=8, layers=2, depth=6,
              drop_rate=0.0, name='unet'):
    with tf.variable_scope(name):
        net = {}
        x = input_tensor
        for d in range(depth):
            with tf.variable_scope('down_{}'.format(d+1)):
                strides = 1 if d == 0 else 2
                x = conv2d(x, filters << d, kernel_size, strides=strides)
                x = activation(x, 'relu')
                for l in range(layers-1):
                    x = conv2d(x, filters << d, kernel_size)
                    x = activation(x, 'relu')
                net['down_{}'.format(d+1)] = x

        with tf.variable_scope('up_{}'.format(depth)):
            net['up_{}'.format(depth)] = net['down_{}'.format(depth)]

        for d in range(depth-1, 0, -1):
            with tf.variable_scope('up_{}'.format(d)):
                x = conv2d_transpose(net['up_{}'.format(d+1)], filters <<
                                     (d-1), kernel_size, padding='same')
                x = activation(x, 'relu')
                x = concatenate(x, net['down_{}'.format(d)])
                for l in range(layers-1):
                    x = conv2d(x, filters << (d-1), kernel_size)
                    x = activation(x, 'relu')
                net['up_{}'.format(d)] = x

        return net['up_1']


def unet3d(input_tensor, training, kernel_size=(3, 3, 3), filters=8, layers=2,
           depth=6, drop_rate=0.0, name='unet', dilation=False):
    with tf.variable_scope(name):
        net = {}
        x = input_tensor
        if not isinstance(filters, list):
            filters = [filters<<x for x in range(depth)]
        for d in range(depth):
            with tf.variable_scope('down_{}'.format(d+1)):
                strides = 1 if d == 0 else 2
                x = conv3d(x, filters[d], kernel_size, strides=strides)
                #x = batch_normalization(x, training, fused=True)
                x = activation(x, 'relu')
                x = dropout(x, drop_rate, training)
                for l in range(layers-1):
                    dia = l+2 if dilation else 1
                    x = conv3d(x, filters[d], kernel_size, dilation_rate=dia)
                    #x = batch_normalization(x, training, fused=True)
                    x = activation(x, 'relu')
                    x = dropout(x, drop_rate, training)
                x = batch_normalization(x, training, fused=True)
                net['down_{}'.format(d+1)] = x

        with tf.variable_scope('up_{}'.format(depth)):
            net['up_{}'.format(depth)] = net['down_{}'.format(depth)]

        for d in range(depth-1, 0, -1):
            with tf.variable_scope('up_{}'.format(d)):
                x = conv3d_transpose(net['up_{}'.format(d+1)], filters[d-1],
                                     kernel_size, padding='same')
                #x = batch_normalization(x, training, fused=True)
                x = activation(x, 'relu')
                x = concatenate(x, net['down_{}'.format(d)])
                x = dropout(x, drop_rate, training)
                for l in range(layers-1):
                    dia = l+2 if dilation else 1
                    x = conv3d(x, filters[d-1], kernel_size, dilation_rate=dia)
                    #x = batch_normalization(x, training, fused=True)
                    x = activation(x, 'relu')
                    x = dropout(x, drop_rate, training)
                x = batch_normalization(x, training, fused=True)
                net['up_{}'.format(d)] = x

        return net['up_1']

def flatten(x) :
    return tf.layers.flatten(x)

def hw_flatten(x) :
    #[b, d, h, w, c]
    x = tf.transpose(x, [0, 4, 1, 2, 3])
    #[b, c, d, h, w]
    return tf.reshape(x, shape=[-1, x.shape[1]*x.shape[2], x.shape[3]*x.shape[4]])

def d_flatten(x) :
    #[b, d, h, w, c]
    x = tf.transpose(x, [0, 2, 3, 4, 1])
    #[b, h, w, c, d]
    return tf.reshape(x, shape=[-1, x.shape[1]*x.shape[2]*x.shape[3], x.shape[4]])

def inplane_attention(x, ch):
    #plane attention
    f = conv3d(x, ch // 8, kernel_size=(1,1,1), strides=1) # [b, d, h, w, c]
    g = conv3d(x, ch // 8, kernel_size=(1,1,1), strides=1) # [b, d, h, w, c]
    h = conv3d(x, ch, kernel_size=(1,1,1), strides=1) # [b, d, h, w, c]

    # N = h * w
    g_flat = hw_flatten(g)
    f_flat = hw_flatten(f)
    s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_a=True) # # [b, N, N]

    beta = tf.nn.softmax(s)  # attention map

    o = tf.matmul(hw_flatten(h), beta) # [b, c*d, N]
    gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

    o = tf.reshape(o, shape=[-1, ch, x.shape[1], x.shape[2], x.shape[3]]) # [b, c, d, h, w]
    o = tf.transpose(o, [0, 2, 3, 4, 1])

    x = gamma * o + x

    return x


def inplane_depth_attention(x, ch, training):
    with tf.variable_scope('attention'):
        f = conv3d(x, ch // 6, kernel_size=(1,1,1), strides=1) # [b, d, h, w, c]
        g = conv3d(x, ch // 6, kernel_size=(1,1,1), strides=1) # [b, d, h, w, c]
        h = conv3d(x, ch, kernel_size=(1,1,1), strides=1) # [b, d, h, w, c]

        #plane attention
        # N = h * w
        g_flat = hw_flatten(g)
        f_flat = hw_flatten(f)
        s_hw = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_a=True) # # [b, N, N]
        beta_hw = tf.nn.softmax(s_hw)  # attention map

        o_hw = tf.matmul(hw_flatten(h), beta_hw) # [b, c*d, N]

        o_hw = tf.reshape(o_hw, shape=[-1, ch, x.shape[1], x.shape[2], x.shape[3]]) # [b, c, d, h, w]
        o_hw = tf.transpose(o_hw, [0, 2, 3, 4, 1])
        o_hw = batch_normalization(o_hw, training, fused=True)
        o_hw = activation(o_hw, 'relu')

        #depth attention
        g_flat = d_flatten(g)
        f_flat = d_flatten(f)
        s_d = tf.matmul(d_flatten(g), d_flatten(f), transpose_a=True) # # [b, d, d]
        beta_d = tf.nn.softmax(s_d)  # attention map

        o_d = tf.matmul(d_flatten(h), beta_d) # [b, h*w*c, d]

        o_d = tf.reshape(o_d, shape=[-1, x.shape[2], x.shape[3], ch, x.shape[1]]) # [b, c, d, h, w]
        o_d = tf.transpose(o_d, [0, 4, 1, 2, 3])
        o_d = batch_normalization(o_d, training, fused=True)
        o_d = activation(o_d, 'relu')

        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        x = gamma * (o_d+o_hw) + x

    return x

def unet3d_sa(input_tensor, training, kernel_size=(3, 3, 3), filters=8, layers=2,
           depth=6, drop_rate=0.0, name='unet', dilation=False):
    with tf.variable_scope(name):
        net = {}
        x = input_tensor
        if not isinstance(filters, list):
            filters = [filters*(x+1) for x in range(depth)]
        for d in range(depth):
            with tf.variable_scope('down_{}'.format(d+1)):
                strides = 1 if d == 0 else 2
                x = conv3d(x, filters[d], kernel_size, strides=strides)
                x = batch_normalization(x, training, fused=True)
                x = activation(x, 'relu')
                x = dropout(x, drop_rate, training)
                for l in range(layers-1):
                    dia = l+2 if dilation else 1
                    x = conv3d(x, filters[d], kernel_size, dilation_rate=dia)
                    if (l == layers-2) and (d== 1 or d == 2 or d == 3):
                        with tf.variable_scope('attention_{}'.format(d+1)):
                            x = inplane_depth_attention(x, ch=filters[d], training=training)
                    x = batch_normalization(x, training, fused=True)
                    x = activation(x, 'relu')
                    x = dropout(x, drop_rate, training)

                # x = batch_normalization(x, training, fused=True)
                net['down_{}'.format(d+1)] = x

        with tf.variable_scope('up_{}'.format(depth)):
            net['up_{}'.format(depth)] = net['down_{}'.format(depth)]

        for d in range(depth-1, 0, -1):
            with tf.variable_scope('up_{}'.format(d)):
                x = conv3d_transpose(net['up_{}'.format(d+1)], filters[d-1],
                                     kernel_size, padding='same')
                x = batch_normalization(x, training, fused=True)
                x = activation(x, 'relu')
                x = concatenate(x, net['down_{}'.format(d)])
                x = dropout(x, drop_rate, training)
                for l in range(layers-1):
                    dia = l+2 if dilation else 1
                    x = conv3d(x, filters[d-1], kernel_size, dilation_rate=dia)
                    x = batch_normalization(x, training, fused=True)
                    x = activation(x, 'relu')
                    x = dropout(x, drop_rate, training)
                # x = batch_normalization(x, training, fused=True)
                net['up_{}'.format(d)] = x

        return net['up_1']


def simple_discriminator(x, n_layers=3, reuse=False):
    with tf.variable_scope('Classifier', reuse=reuse):
        # filters = 8
        # filters = [filters<<i for i in range(n_layers)]
        filters = [8, 4, 2, 1, 1, 1]
        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (i)):
                x = conv3d(x, filters[i], kernel_size=(3,3,3), strides=2)
                x = batch_normalization(x, 1, fused=True)
                x = activation(x, 'leaky_relu', alpha=0.2)

        logits = conv3d(x, 1, kernel_size=(1,1,1), strides=1)
        probability=tf.sigmoid(logits)

        return probability, logits



def tracker(x, training, name='Tracker'):
    with tf.variable_scope(name):
        filters = [48, 48, 48, 48, 64, 64]
        dias = [1, 1, 2, 4, 1, 1]
        kernel_size = [3, 3, 3, 3, 3, 1]
        for i in range(6):
            with tf.variable_scope("layer_%d" % (i)):
                x = conv3d(x, filters=filters[i], kernel_size=kernel_size[i], strides=1, dilation_rate=dias[i], padding='valid')
                x = batch_normalization(x, training, fused=True)
                x = activation(x, 'relu')


        temp_y_directions_out = conv3d(x,
                                  filters = 500 ,
                                  kernel_size = 1,
                                  strides = 1,
                                  padding='valid')

        temp_y_radius_out = conv3d (x,
                              filters = 1 ,
                              kernel_size = 1,
                              strides = 1,
                              padding='valid'
                              )

        y_directions_out = tf.reshape(temp_y_directions_out, [-1, 500], name="y_directions")
        y_radius_out =  tf.reshape(temp_y_radius_out, [-1, 1], name = "y_radius")

        return y_directions_out, y_radius_out


def classifier(x, n_outs, n_layers=2, is_training=True, reuse=False, scope='scope'):
    channel = 32
    with tf.variable_scope(scope, reuse=reuse):
        x = conv2d_scope(x, channel, kernel_size=4, strides=2,
                   name='discrim_conv00', reuse=reuse)
        x = activation(x, 'leaky_relu', name='discrim_actv0')

        for i in range(n_layers):
            x = conv2d_scope(x, channel * 2, kernel_size=4, strides=2,
                       name='discrim_conv'+str(i), reuse=reuse)
            x = batch_normalization_scope(
                x, is_training, name='discrim_batchnorm'+str(i), reuse=reuse)
            x = activation(x, 'leaky_relu', name='discrim_actv'+str(i))
            channel = channel * 2

        x = conv2d_scope(x, channel, kernel_size=3, strides=1,
                   name='discrim_conv'+str(i+1), reuse=reuse)
        x = batch_normalization_scope(
            x, is_training, name='discrim_batchnorm'+str(i+1), reuse=reuse)
        x = activation(x, 'leaky_relu', name='discrim_activ'+str(i+1))

        x = conv2d_scope(x, n_outs, kernel_size=3, strides=1,
                   name='discrim_conv_end', reuse=reuse)

        return x


################################# 3D Operations ################################


def conv3d(x, filters, kernel_size, padding='same', dilation_rate=(1, 1, 1),
           activation=None, strides=(1, 1, 1)):
    return tf.layers.conv3d(
        inputs=x,
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        dilation_rate=dilation_rate,
        activation=activation,
        strides=strides)


def conv3d_transpose(x, filters, kernel_size, strides=(2, 2, 2),
                     padding='valid', activation=None):
    return tf.layers.conv3d_transpose(
        inputs=x,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation)


def max_pooling3d(x, pool_size, strides=(2, 2, 2)):
    return tf.layers.max_pooling3d(inputs=x, pool_size=pool_size, strides=strides)


def down_block3d(input_tensor, filters, training, name,
                 drop_rate=0.0, padding='same', kernel_size=(3, 3, 3),
                 pool_size=(2, 2, 2), pool_stride=(2, 2, 2),
                 layers=1, batch_norm=True):
    with tf.name_scope(name):
        x = conv3d(input_tensor, filters, kernel_size=kernel_size, padding=padding)
        if batch_norm:
            x = batch_normalization(x, training)
        x = activation(x, 'relu')
        x = dropout(x, drop_rate, training)
        for i in range(layers - 1):
            x = conv3d(x, filters, kernel_size=kernel_size, padding=padding)
            if batch_norm:
                x = batch_normalization(x, training)
            x = activation(x, 'relu')
            x = dropout(x, drop_rate, training)

        return max_pooling3d(x, pool_size=pool_size, strides=pool_stride), x


def up_block3d(input_tensor, skip_tensor, filters, training, name,
               drop_rate=0.0, padding='same', kernel_size=(3, 3, 3),
               unpool_size=(2, 2, 2), unpool_stride=(2, 2, 2),
               layers=1, batch_norm=True):
    with tf.name_scope(name):
        x = conv3d_transpose(input_tensor, filters, kernel_size=unpool_size, strides=unpool_stride)
        x = concatenate(x, skip_tensor, axis=4)
        x = conv3d(x, filters, kernel_size=kernel_size, padding=padding)
        if batch_norm:
            x = batch_normalization(x, training)
        x = activation(x, 'relu')
        x = dropout(x, drop_rate, training)
        for i in range(layers - 1):
            x = conv3d(x, filters, kernel_size=kernel_size, padding=padding)
            if batch_norm:
                x = batch_normalization(x, training)
            x = activation(x, 'relu')
            x = dropout(x, drop_rate, training)

        return x


def extract_regions_loop_condition(i, regions, image_p, mask, labels, region_dims):

    return i < tf.shape(labels)[0]


def extract_regions_loop_body(i, regions, image, mask, labels, region_dims):

    region_mask = tf.equal(mask, labels[i])  # [None, None, None]
    true_indices = tf.where(region_mask)  # [num_of_true_elements, 3]

    min_indices = tf.reduce_min(true_indices, axis=0)  # [3]
    max_indices = tf.reduce_max(true_indices, axis=0)  # [3]

    center = tf.cast((min_indices + max_indices) / 2, tf.int32)

    start = center - tf.cast(region_dims / 2, 'int32')
    end = start + region_dims

    # Making sure that start (inclusive) and end (exclusive) are valid, and if not make sure to slice and pad accordingly
    image_dims = tf.shape(image)[0:3]

    start_valid = tf.clip_by_value(start, 0, image_dims - 1)
    pad_left = tf.clip_by_value(0 - start, 0, tf.cast(region_dims / 2, 'int32'))
    end_valid = tf.clip_by_value(end, 1, image_dims)
    pad_right = tf.clip_by_value(end - image_dims, 0, tf.cast(region_dims / 2, 'int32'))

    # region_image = tf.squeeze(image, -1) * tf.cast(region_mask, tf.float32)  # [None, None, None]
    region_image = tf.squeeze(image, -1)  # [None, None, None]

    region = tf.slice(
        region_image,
        [start_valid[0], start_valid[1], start_valid[2]],
        [end_valid[0] - start_valid[0], end_valid[1] - start_valid[1], end_valid[2] - start_valid[2]])
    region = tf.pad(
        region,
        [[pad_left[0], pad_right[0]], [pad_left[1], pad_right[1]], [pad_left[2], pad_right[2]]])

    region = tf.expand_dims(region, -1)  # [None, None, None, 1]

    # Volume is of shape [height, width, depth, channels]
    # Transposing the volume shape to [depth, height, width, channels]
    region = tf.transpose(region, [2, 0, 1, 3])

    # Flipping regions in one of the hemispheres
    region = tf.cond(
        labels[i] >= 11,
        lambda: tf.image.flip_up_down(region),
        lambda: tf.identity(region))

    # Transposing the volume shape back to [height, width, depth, channels]
    region = tf.transpose(region, [1, 2, 0, 3])

    # [1, region_dims[0], region_dims[1], region_dims[2], 1]
    region = tf.expand_dims(region, 0)

    # [num_of_labels + 1, region_dims[0], region_dims[1], region_dims[2], 1]
    regions = tf.concat([regions, region], axis=0)
    # [num_of_labels, region_dims[0], region_dims[1], region_dims[2], 1]
    regions = tf.slice(regions, [1, 0, 0, 0, 0], [-1, -1, -1, -1, -1])

    i = tf.add(i, 1)

    return i, regions, image, mask, labels, region_dims


def extract_regions(image, mask, labels, region_dims):
    """ Extract num_of_labels regions of shape region_dims from image based on mask

    :param image: a 3D image (shape = [None, None, None, 1])
    :param mask: a 3D mask (shape = [None, None, None])
    :param labels: a tensor of label values (shape = [num_of_labels])
    :param region_dims: the shape of the extracted regions ([height, width, depth])
    :returns regions: a tensor with num_of_labels fixed size boxes, containing the regions (shape = [num_of_labels, region_dims[0], region_dims[1], region_dims[2], 1])

    """

    i = tf.constant(0, dtype=tf.int32)
    region_dims = tf.constant(region_dims, dtype=tf.int32)
    regions = tf.zeros(shape=[tf.shape(labels)[0], region_dims[0], region_dims[1], region_dims[2], 1])
    return tf.while_loop(extract_regions_loop_condition, extract_regions_loop_body, [i, regions, image, mask, labels, region_dims])[1]


def augment_regions_loop_condition(i, augmented_regions, regions, transform_mat):

    return i < tf.shape(regions)[0]


def augment_regions_loop_body(i, augmented_regions, regions, transform_mat):

    augmented_region = stn3d.spatial_transformer_network(tf.expand_dims(regions[i], 0), tf.expand_dims(transform_mat[i], 0))

    augmented_regions = tf.concat([augmented_regions, augmented_region], axis=0)
    augmented_regions = tf.slice(augmented_regions, [1, 0, 0, 0, 0], [-1, -1, -1, -1, -1])
    augmented_regions.set_shape([regions.shape[0], regions.shape[1], regions.shape[2], regions.shape[3], regions.shape[4]])

    i = tf.add(i, 1)

    return i, augmented_regions, regions, transform_mat


def augment_regions(regions, transform_mat):
    """ Augment each region in regions independently based on transform_mat

    :param regions: a 5D tensor of shape [#regions==10, box_height==256, box_width==256, box_depth==16, channels==2]
    :param transform_mat: a transformation tensor of shape [#regions, 3, 4]
    :returns augmented_regions: a tensor of shape [10, 256, 256, 16, 2]

    """

    i = tf.constant(0, dtype=tf.int32)
    augmented_regions = tf.zeros_like(regions)
    return tf.while_loop(augment_regions_loop_condition, augment_regions_loop_body, [i, augmented_regions, regions, transform_mat])[1]


def extract_aspects_union(image, mask, out_dims):
    """ Extracts a 3D box of shape out_dims from image and mask, with its center matching the center of the 3D object in mask

    :param image is of shape [D, H, W]
    :param mask is of shape [D, H, W]
    """

    true_indices = tf.where(mask > 0)  # [num_of_true_elements, 3]

    min_indices = tf.reduce_min(true_indices, axis=0)  # [3]
    max_indices = tf.reduce_max(true_indices, axis=0)  # [3]

    center = tf.cast((min_indices + max_indices) / 2, tf.int32)

    start = center - tf.cast(out_dims / 2, 'int32')
    end = start + out_dims

    # Making sure that start (inclusive) and end (exclusive) are valid, and if not, make sure to slice and pad accordingly
    image_dims = tf.shape(image)[0:3]

    start_valid = tf.clip_by_value(start, 0, image_dims - 1)
    pad_left = tf.clip_by_value(0 - start, 0, tf.cast(out_dims / 2, 'int32'))
    end_valid = tf.clip_by_value(end, 1, image_dims)
    pad_right = tf.clip_by_value(end - image_dims, 0, tf.cast(out_dims / 2, 'int32'))

    union_image = tf.slice(
        image,
        [start_valid[0], start_valid[1], start_valid[2]],
        [end_valid[0] - start_valid[0], end_valid[1] - start_valid[1], end_valid[2] - start_valid[2]])
    union_image = tf.pad(
        union_image,
        [[pad_left[0], pad_right[0]], [pad_left[1], pad_right[1]], [pad_left[2], pad_right[2]]])

    union_mask = tf.slice(
        mask,
        [start_valid[0], start_valid[1], start_valid[2]],
        [end_valid[0] - start_valid[0], end_valid[1] - start_valid[1], end_valid[2] - start_valid[2]])
    union_mask = tf.pad(
        union_mask,
        [[pad_left[0], pad_right[0]], [pad_left[1], pad_right[1]], [pad_left[2], pad_right[2]]])

    union_image.set_shape(out_dims)
    union_mask.set_shape(out_dims)

    return union_image, union_mask, start_valid, end_valid, pad_left, pad_right


############################### Atrous Operations ###############################


def artous_pyramid(inputs, training, depth=3, atrous_filters=8, conv_filters=8,
                   scope='atrous_pyramid'):
    with tf.variable_scope(scope):
        net = {}
        depth = max(depth, 0)

        with tf.variable_scope('atrous_0'):
            x = conv2d(inputs, filters=atrous_filters, kernel_size=(1, 1))
            x = activation(x, 'relu')
            x = conv2d(x, filters=conv_filters, kernel_size=(3, 3))
            x = batch_normalization(x, training)
            x = activation(x, 'relu')
            net['atrous_0'] = x

        for l in range(depth):
            with tf.variable_scope('atrous_{}'.format(l+1)):
                x = conv2d(inputs, filters=atrous_filters,
                           kernel_size=(3, 3), dilation_rate=l+1)
                x = activation(x, 'relu')
                x = conv2d(x, filters=conv_filters, kernel_size=(3, 3))
                x = batch_normalization(x, training)
                x = activation(x, 'relu')
                net['atrous_{}'.format(l+1)] = x

        with tf.variable_scope('synthesis'):
            x = net['atrous_{}'.format(depth)]
            for l in range(depth):
                x += net['atrous_{}'.format(l)]
            x = conv2d(x, filters=conv_filters, kernel_size=(1, 1))
            x = activation(x, 'relu')
            x = batch_normalization(x, training)

        return x


def context_module(inputs, training, channels=1, depth=4, name='context_module'):
    with tf.variable_scope(name):
        net = {}

        with tf.variable_scope('atrous_0'):
            x = conv2d(inputs, filters=channels, kernel_size=(3, 3))
            x = activation(x, 'relu')
            x = batch_normalization(x, training)
            net['atrous_0'] = x

        with tf.variable_scope('atrous_1'):
            x = conv2d(net['atrous_0'], filters=channels, kernel_size=(3, 3))
            x = activation(x, 'relu')
            x = batch_normalization(x, training)
            net['atrous_1'] = x

        for l in range(depth):
            with tf.variable_scope('atrous_{}'.format(l+2)):
                x = conv2d(net['atrous_{}'.format(l+1)], filters=channels,
                           kernel_size=(3, 3), dilation_rate=1 << (l+1))
                x = activation(x, 'relu')
                x = batch_normalization(x, training)
                net['atrous_{}'.format(l+2)] = x

        with tf.variable_scope('atrous_6'):
            x = conv2d(net['atrous_5'], filters=channels, kernel_size=(3, 3))
            x = activation(x, 'relu')
            x = batch_normalization(x, training)
            net['atrous_6'] = x

        with tf.variable_scope('atrous_7'):
            x = conv2d(net['atrous_6'], filters=channels, kernel_size=(1, 1))
            net['atrous_7'] = x

        return net['atrous_7']


################################# RNN Operations ################################


def down_block_lstm_atrous(inputs, input_shape, max_sequence_length,
                           batch_size, output_channels=8, scope='down'):
    with tf.variable_scope(scope):
        strided_input_shape = [input_shape[0] >> 1,
                               input_shape[1] >> 1,
                               input_shape[2]]
        cell = rnn.Conv2DLSTMCell(input_shape=strided_input_shape,
                                  output_channels=output_channels,
                                  kernel_shape=[3, 3])
        state = cell.zero_state(batch_size*4, tf.float32)
        outputs = []
        skip = []
        for t in range(max_sequence_length):
            if t > 0:
                tf.get_variable_scope().reuse_variables()
            x = dilate_input(inputs[t], strided_input_shape)
            # for stride in range(4):
            #     x_temp, state = cell(x[stride], state)
            #     x[stride] = x_temp  # [:,1:-1,1:-1,:]
            x, state = cell(x, state)
            x = undilate_input(x, input_shape, output_channels)
            skip += [x]
            x = max_pooling2d(x, (2, 2))
            outputs += [x]
        return outputs, skip


def dense_block_lstm_atrous(inputs, input_shape, max_sequence_length,
                            batch_size, output_channels=8, scope='dense'):
    with tf.variable_scope(scope):
        strided_input_shape = [input_shape[0] >> 1,
                               input_shape[1] >> 1,
                               input_shape[2]]
        cell = rnn.Conv2DLSTMCell(input_shape=strided_input_shape,
                                  output_channels=output_channels,
                                  kernel_shape=[3, 3])
        state = cell.zero_state(batch_size*4, tf.float32)
        outputs = []
        for t in range(max_sequence_length):
            if t > 0:
                tf.get_variable_scope().reuse_variables()
            x = dilate_input(inputs[t], strided_input_shape)
            # for stride in range(4):
            #     x_temp, state = cell(x[stride], state)
            #     x[stride] = x_temp  # [:,1:-1,1:-1,:]
            x, state = cell(x, state)
            x = undilate_input(x, input_shape, output_channels)
            outputs += [x]
        return outputs


def down_block_lstm_manual(inputs, input_shape, max_sequence_length,
                           batch_size, training=False, kernel_shape=[3,3],
                           output_channels=8, drop_rate=0.0, scope='down'):
    with tf.variable_scope(scope):
        cell = rnn.Conv2DLSTMCell(input_shape=input_shape,
                                  output_channels=output_channels,
                                  kernel_shape=kernel_shape)
        state = cell.zero_state(batch_size, tf.float32)
        outputs = []
        skip = []
        for t in range(max_sequence_length):
            if t > 0:
                tf.get_variable_scope().reuse_variables()
            x, state = cell(inputs[t], state)
            x = activation(x, 'relu')
            x = dropout(x, drop_rate, training)
            skip += [x]
            x = max_pooling2d(x, (2, 2))
            outputs += [x]
        return outputs, skip


def dense_block_lstm_manual(inputs, input_shape, max_sequence_length,
                            batch_size, training=False, kernel_shape=[3,3],
                            output_channels=8, drop_rate= 0.0, scope='dense'):
    with tf.variable_scope(scope):
        cell = rnn.Conv2DLSTMCell(input_shape=input_shape,
                                  output_channels=output_channels,
                                  kernel_shape=kernel_shape)
        state = cell.zero_state(batch_size, tf.float32)
        outputs = []
        for t in range(max_sequence_length):
            if t > 0:
                tf.get_variable_scope().reuse_variables()
            x, state = cell(inputs[t], state)
            x = activation(x, 'relu')
            x = dropout(x, drop_rate, training)
            outputs += [x]
        return outputs


def up_block_lstm_manual(inputs, input_shape, max_sequence_length, batch_size,
                         training=False, kernel_shape=[3,3], output_channels=8,
                         scope='up', drop_rate=0.0, skip=None):
    with tf.variable_scope(scope):
        cell = rnn.Conv2DLSTMCell(input_shape=input_shape,
                                  output_channels=output_channels,
                                  kernel_shape=kernel_shape)
        state = cell.zero_state(batch_size, tf.float32)
        outputs = []
        for t in range(max_sequence_length):
            if t > 0:
                tf.get_variable_scope().reuse_variables()
            x = tf.layers.conv2d_transpose(inputs=inputs[t],
                                           filters=output_channels,
                                           kernel_size=(2, 2), strides=(2, 2),
                                           padding='valid', activation=None,
                                           name='transpose')
            if skip:
                x = concatenate(x, skip[t])
            x, state = cell(x, state)
            x = activation(x, 'relu')
            x = dropout(x, drop_rate, training)
            outputs += [x]
        return outputs


def ulstm_atrous(rnnin, input_shape, max_sequence_length, kernel_shape=[3, 3],
                 output_channels=8, scope='rnn', depth=4):
    with tf.variable_scope(scope):
        batch_size = tf.shape(rnnin[0])[0]
        net = {}
        shape = [input_shape[0], input_shape[1], input_shape[2]]
        out = down_block_lstm_atrous(rnnin, shape, max_sequence_length,
                                     batch_size, scope='down1',
                                     output_channels=output_channels)
        net['down1'], net['skip1'] = out

        for l in range(1, depth):
            shape = [input_shape[0] >> l, input_shape[1] >> l,
                     output_channels*(2**(l-1))]
            out = down_block_lstm_atrous(net['down{}'.format(l)], shape,
                                         max_sequence_length, batch_size,
                                         scope='down{}'.format(l+1),
                                         output_channels=output_channels*(2**l))
            net['down{}'.format(l+1)], net['skip{}'.format(l+1)] = out

        l = depth
        shape = [input_shape[0] >> l, input_shape[1] >> l,
                 output_channels*(2**(l-1))]
        out = dense_block_lstm_atrous(net['down{}'.format(l)], shape,
                                      max_sequence_length, batch_size,
                                      scope='dense1',
                                      output_channels=output_channels*(2**l))
        net['up{}'.format(l+1)] = out

        for l in range(depth, 0, -1):
            shape = [input_shape[0] >> (l-1), input_shape[1] >> (l-1),
                     output_channels*(2**(l-1))*3]
            out = up_block_lstm_manual(net['up{}'.format(l+1)], shape,
                                       max_sequence_length, batch_size,
                                       scope='up{}'.format(l),
                                       output_channels=output_channels *
                                       (2**(l-1)),
                                       kernel_shape=kernel_shape,
                                       skip=net['skip{}'.format(l)])
            net['up{}'.format(l)] = out

        return net['up1']


def ulstm(rnnin, input_shape, max_sequence_length, training=False,
          kernel_shape=[3,3], output_channels=8, scope='rnn', depth=4,
          drop_rate=0.0):
    with tf.variable_scope(scope):
        batch_size = tf.shape(rnnin[0])[0]
        net = {}
        shape = [input_shape[0], input_shape[1], input_shape[2]]
        out = down_block_lstm_manual(rnnin, shape, max_sequence_length,
                                     batch_size, training=training,
                                     scope='down1',
                                     output_channels=output_channels,
                                     kernel_shape=kernel_shape,
                                     drop_rate=drop_rate)
        net['down1'], net['skip1'] = out

        for l in range(1, depth):
            shape = [input_shape[0] >> l, input_shape[1] >> l,
                     output_channels*(2**(l-1))]
            out = down_block_lstm_manual(net['down{}'.format(l)], shape,
                                         max_sequence_length, batch_size,
                                         training=training,
                                         scope='down{}'.format(l+1),
                                         output_channels=output_channels*(2**l),
                                         kernel_shape=kernel_shape,
                                         drop_rate=drop_rate)
            net['down{}'.format(l+1)], net['skip{}'.format(l+1)] = out

        l = depth
        shape = [input_shape[0] >> l, input_shape[1] >> l,
                 output_channels*(2**(l-1))]
        out = dense_block_lstm_manual(net['down{}'.format(l)], shape,
                                      max_sequence_length, batch_size,
                                      training=training, scope='dense1',
                                      output_channels=output_channels*(2**l),
                                      kernel_shape=kernel_shape,
                                      drop_rate=drop_rate)
        # out = dense_block_lstm_manual(out, shape,
        #                               max_sequence_length, batch_size,
        #                               scope='dense2',
        #                               output_channels=output_channels*(2**l),
        #                               kernel_shape=kernel_shape)
        net['up{}'.format(l+1)] = out

        for l in range(depth, 0, -1):
            shape = [input_shape[0] >> (l-1), input_shape[1] >> (l-1),
                     output_channels*(2**(l-1))*3]
            out = up_block_lstm_manual(net['up{}'.format(l+1)], shape,
                                       max_sequence_length, batch_size,
                                       training=training,
                                       scope='up{}'.format(l),
                                       output_channels=output_channels *
                                       (2**(l-1)),
                                       kernel_shape=kernel_shape,
                                       drop_rate=drop_rate,
                                       skip=net['skip{}'.format(l)])
            net['up{}'.format(l)] = out

        return net['up1']


def ulstm_bidirectional(x, input_shape, max_sequence_length, training=False,
                        kernel_shape=[3, 3], output_channels=8,
                        time_major=False, scope='rnn', depth=4, atrous=False,
                        drop_rate=0.0):
    with tf.variable_scope(scope):
        with tf.variable_scope('unstack'):
            if time_major:
                rnnin = tf.reshape(
                    x, [max_sequence_length, -1, input_shape[0], input_shape[1], input_shape[2]])
                rnnin = tf.unstack(rnnin, axis=0)
            else:
                rnnin = tf.reshape(
                    x, [-1, max_sequence_length, input_shape[0], input_shape[1], input_shape[2]])
                rnnin = tf.unstack(rnnin, axis=1)

        if atrous:
            out_fw = ulstm_atrous(rnnin, input_shape, max_sequence_length,
                                  scope="forward",
                                  output_channels=output_channels,
                                  kernel_shape=kernel_shape, depth=depth)

            out_bw = ulstm_atrous(rnnin, input_shape, max_sequence_length,
                                  scope="backward",
                                  output_channels=output_channels,
                                  kernel_shape=kernel_shape, depth=depth)
        else:
            out_fw = ulstm(rnnin, input_shape, max_sequence_length,
                           training=training, scope="forward",
                           output_channels=output_channels,
                           kernel_shape=kernel_shape, depth=depth,
                           drop_rate=drop_rate)

            out_bw = ulstm(rnnin, input_shape, max_sequence_length,
                           training=training, scope="backward",
                           output_channels=output_channels,
                           kernel_shape=kernel_shape, depth=depth,
                           drop_rate=drop_rate)

        with tf.variable_scope('stack', reuse=tf.AUTO_REUSE):
            outputs = []
            for t in range(max_sequence_length):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                x = tf.concat(
                    [out_fw[t], out_bw[max_sequence_length-1-t]], axis=-1)
                outputs += [x]
            if time_major:
                rnnout = tf.stack(outputs, axis=0)
            else:
                rnnout = tf.stack(outputs, axis=1)
            rnnout = tf.reshape(
                rnnout, [-1, input_shape[0], input_shape[1], output_channels*2])
    return rnnout


def bi2dconvlstm(x, input_shape, max_sequence_length, sequence_length,
                 num_layers=1, kernel_shape=[3, 3], output_channels=8,
                 is_symmetric=True, time_major=False, scope='rnn'):
    with tf.variable_scope(scope):
        with tf.variable_scope(scope+'_stack'):
            if time_major:
                rnnin = tf.reshape(
                    x, [max_sequence_length, -1, input_shape[0], input_shape[1], input_shape[2]])
            else:
                rnnin = tf.reshape(
                    x, [-1, max_sequence_length, input_shape[0], input_shape[1], input_shape[2]])
        with tf.variable_scope(scope+'_core'):
            if num_layers < 2:
                rnnfw = rnn.Conv2DLSTMCell(input_shape=[input_shape[0], input_shape[1], input_shape[2]],
                                           output_channels=output_channels, kernel_shape=kernel_shape, name='Conv2dlstmcell_fw')
                rnnbw = rnn.Conv2DLSTMCell(input_shape=[input_shape[0], input_shape[1], input_shape[2]],
                                           output_channels=output_channels, kernel_shape=kernel_shape, name='Conv2dlstmcell_bw')
            else:
                rnnfw = [rnn.Conv2DLSTMCell(input_shape=[input_shape[0], input_shape[1], input_shape[2]], output_channels=output_channels,
                                            kernel_shape=kernel_shape, name='Conv2dlstmcell_fw_{}'.format(n)) for n in range(num_layers)]
                rnnbw = [rnn.Conv2DLSTMCell(input_shape=[input_shape[0], input_shape[1], input_shape[2]], output_channels=output_channels,
                                            kernel_shape=kernel_shape, name='Conv2dlstmcell_bw_{}'.format(n)) for n in range(num_layers)]
                rnnfw = rnn.MultiRNNCell(rnnfw)
                rnnbw = rnn.MultiRNNCell(rnnbw)
            if is_symmetric:
                rnnout, _ = tf.nn.bidirectional_dynamic_rnn(
                    rnnfw, rnnfw, rnnin, dtype=tf.float32, time_major=time_major, sequence_length=sequence_length, scope='Symmetric_bidirectional_rnn')
            else:
                rnnout, _ = tf.nn.bidirectional_dynamic_rnn(
                    rnnfw, rnnbw, rnnin, dtype=tf.float32, time_major=time_major, sequence_length=sequence_length, scope='Asymmetric_bidirectional_rnn')
        with tf.variable_scope(scope+'_unstack'):
            rnnout = tf.concat(rnnout, -1)
            rnnout = tf.reshape(
                rnnout, [-1, input_shape[0], input_shape[1], output_channels*2])
        return rnnout


def down_block_lstm(input_tensor, input_shape, max_sequence_length,
                    sequence_length, filters, training, name, padding='same',
                    pool_type='max', is_symmetric=True, time_major=False):
    with tf.variable_scope(name):
        x = bi2dconvlstm(input_tensor, input_shape, max_sequence_length, sequence_length,
                         num_layers=2, output_channels=(filters >> 1),
                         is_symmetric=is_symmetric, time_major=time_major,
                         scope=name)
        x = batch_normalization(x, training)
        x = activation(x, 'relu')
        if pool_type == 'max':
            return max_pooling2d(x, pool_size=(2, 2)), x
        elif pool_type == 'avg':
            return avg_pooling2d(x, pool_size=(2, 2)), x
        else:
            print('unsupported pooling type')
            exit()


def up_block_lstm(input_tensor, input_shape, max_sequence_length,
                  sequence_length, skip_tensor, filters, training, name,
                  padding='same', is_symmetric=True, time_major=False):
    with tf.variable_scope(name):
        x = conv2d_transpose(input_tensor, filters, kernel_size=(2, 2))
        x = concatenate(x, skip_tensor)
        x = bi2dconvlstm(x, input_shape, max_sequence_length, sequence_length,
                         num_layers=2, output_channels=(filters >> 1),
                         is_symmetric=is_symmetric, time_major=time_major,
                         scope=name)
        x = batch_normalization(x, training)
        x = activation(x, 'relu')
        return x


def dense_lstm(input_tensor, input_shape, max_sequence_length, sequence_length,
               filters, training, name, num_layers=2, padding='same', is_symmetric=True,
               time_major=False):
    with tf.variable_scope(name):
        x = bi2dconvlstm(input_tensor, input_shape, max_sequence_length, sequence_length,
                         num_layers=num_layers, output_channels=(filters >> 1),
                         is_symmetric=is_symmetric, time_major=time_major,
                         scope=name)
        x = batch_normalization(x, training)
        x = activation(x, 'relu')
        return x


def get_input_len_rnn(x, crop_shape, max_sequence_length, time_major=False):
    if time_major:
        unwrapped_x = tf.reshape(
            x, [max_sequence_length, -1, crop_shape[0], crop_shape[1], 1])
        length_sequence = tf.sign(tf.reduce_max(
            tf.reduce_max(tf.abs(tf.squeeze(unwrapped_x, 4)), 3), 2))
        length_sequence = tf.reduce_sum(length_sequence, 0)
    else:
        unwrapped_x = tf.reshape(
            x, [-1, max_sequence_length, crop_shape[0], crop_shape[1], 1])
        length_sequence = tf.sign(tf.reduce_max(
            tf.reduce_max(tf.abs(tf.squeeze(unwrapped_x, 4)), 3), 2))
        length_sequence = tf.reduce_sum(length_sequence, 1)
    return tf.cast(length_sequence, tf.int32)


def rnn_softmax_cross_entropy(logits, y, x):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=y)
    cross_entropy_loss = tf.reduce_mean(cross_entropy, axis=(1, 2))
    mask = tf.sign(tf.reduce_max(
        tf.reduce_max(tf.abs(tf.squeeze(x, 3)), 2), 1))
    cross_entropy_loss *= mask
    return tf.reduce_sum(cross_entropy_loss)/tf.reduce_sum(mask)


def dilate_input(inputs, strided_input_shape):
    input_shape = inputs.get_shape()
    output = tf.stack([inputs[:, 0:-1:2, 0:-1:2, :],
                       inputs[:, 1::2, 0:-1:2, :],
                       inputs[:, 0:-1:2, 1::2, :],
                       inputs[:, 1::2, 1::2, :]], 0)
    output = tf.reshape(output, [-1, strided_input_shape[0],
                                 strided_input_shape[1], strided_input_shape[2]])
    return output


def undilate_input(inputs, input_shape, output_channels):
    input_shape_temp = [output_channels, -1,
                        input_shape[0] >> 1, input_shape[1], 1]
    inputs = tf.reshape(
        inputs, [4, -1, input_shape[0] >> 1, input_shape[1] >> 1, output_channels])
    output = tf.reshape(tf.concat([tf.reshape(tf.transpose(tf.stack([inputs[0, :, :, :, :], inputs[2, :, :, :, :]], -1), [3, 0, 1, 2, 4]), input_shape_temp),
                                   tf.reshape(tf.transpose(tf.stack([inputs[1, :, :, :, :], inputs[3, :, :, :, :]], -1), [3, 0, 1, 2, 4]), input_shape_temp)],
                                  -2), (output_channels, -1, input_shape[0], input_shape[1]))
    output = tf.transpose(output, [1, 2, 3, 0])
    return output
