import tensorflow as tf
import numpy as np


def spatial_transformer_network(input_fmap, theta, inter='bilinear', out_dims=None, **kwargs):
    """
    Spatial Transformer Network layer implementation as described in [1].

    The layer is composed of 3 elements:

    - localization_net: takes the original image as input and outputs
      the parameters of the affine transformation that should be applied
      to the input image.

    - affine_grid_generator: generates a grid of (y,x) coordinates that
      correspond to a set of points where the input should be sampled
      to produce the transformed output.

    - sampler: takes as input the original image and the grid
      and produces the output image using bilinear/nearest interpolation.

    Input
    -----
    - input_fmap: output of the previous layer. Can be input if spatial
      transformer layer is at the beginning of architecture. Should be
      a tensor of shape (B, H, W, C).

    - theta: affine transform tensor of shape (B, 2, 3). Permits scaling,
      flipping, rotation, and translation. Initialize to identity matrix.
      It is the output of the localization network. Note that this transform
      matrix is applied to the regular sampling grid rather than on the original
      image.

    Returns
    -------
    - out_fmap: transformed input feature map. Tensor of size (B, H, W, C).

    Notes
    -----
    [1]: 'Spatial Transformer Networks', Jaderberg et. al,
         (https://arxiv.org/abs/1506.02025)

    """
    # grab input dimensions
    H = tf.shape(input_fmap)[1]
    W = tf.shape(input_fmap)[2]

    # generate grids of same size or upsample/downsample if specified
    if out_dims is None:
        batch_grids = affine_grid_generator(H, W, theta)
    else:
        out_H = out_dims[0]
        out_W = out_dims[1]
        batch_grids = affine_grid_generator(out_H, out_W, theta)

    y_s = batch_grids[:, 0, :, :]
    x_s = batch_grids[:, 1, :, :]

    # sample input with grid to get output
    out_fmap = sampler(input_fmap, x_s, y_s, inter)

    return out_fmap


def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.

    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: int tensor of shape (B, H, W)
    - y: int tensor of shape (B, H, W)

    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices_bhw = tf.stack([b, y, x], 3)

    output = tf.gather_nd(img, indices_bhw)
    return output


def affine_grid_generator(height, width, theta):
    """
    This function returns a sampling grid, which when
    used with the bilinear/nearest sampler on the input feature
    map, will create an output feature map that is an
    affine transformation [1] of the input feature map.

    Input
    -----
    - height: desired height of grid/output. Used
      to downsample or upsample.

    - width: desired width of grid/output. Used
      to downsample or upsample.

    - theta: affine transform matrices of shape (num_batch, 2, 3).
      For each image in the batch, we have 6 theta parameters of
      the form (2x3) that define the affine transformation T.

    Returns
    -------
    - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
      The 2nd dimension has 2 components: (x, y) which are the
      sampling points of the original image for each point in the
      target image.

    Note
    ----
    [1]: the affine transformation allows cropping, translation,
         and isotropic scaling.
    """
    num_batch = tf.shape(theta)[0]

    # create normalized 2D grid
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    x_t, y_t = tf.meshgrid(x, y)

    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([y_t_flat, x_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')

    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, 2, H, W)
    batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])

    return batch_grids


def sampler(img, x, y, inter):
    """
    Performs bilinear/nearest sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.

    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.

    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.(B,H,W)
    - inter: interpolation used - bilinear or nearest

    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    img_shape = tf.shape(img)
    H = img_shape[1]
    W = img_shape[2]
    num_channel = img_shape[3]
    max_y = tf.cast(H - 1, 'float32')
    max_x = tf.cast(W - 1, 'float32')
    zero = tf.zeros([], dtype='float32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.floor(x)
    x1 = x0 + 1.0
    y0 = tf.floor(y)
    y1 = y0 + 1.0

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    if inter == 'nearest':
        wa = tf.cast((wa >= wb) & (wa >= wc) & (wa >= wd), 'float32')
        wb = tf.cast((wb > wa) & (wb >= wc) & (wb >= wd), 'float32')
        wc = tf.cast((wc > wa) & (wc > wb) & (wc >= wd), 'float32')
        wd = tf.cast((wd > wa) & (wd > wb) & (wd > wc), 'float32')


    x0_c = tf.clip_by_value(x0, zero, max_x)
    y0_c = tf.clip_by_value(y0, zero, max_y)
    x1_c = tf.clip_by_value(x1, zero, max_x)
    y1_c = tf.clip_by_value(y1, zero, max_y)

    x0_c = tf.cast(x0_c, 'int32')
    y0_c = tf.cast(y0_c, 'int32')
    x1_c = tf.cast(x1_c, 'int32')
    y1_c = tf.cast(y1_c, 'int32')

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0_c, y0_c)
    Ib = get_pixel_value(img, x0_c, y1_c)
    Ic = get_pixel_value(img, x1_c, y0_c)
    Id = get_pixel_value(img, x1_c, y1_c)

    # Ia has the same size as the output image rather than input image
    # setting out of bounds values to 0 and x, y coords have to be unclipped float values
    x0 = tf.tile(tf.expand_dims(x0, axis=3), [1,1,1,num_channel])
    y0 = tf.tile(tf.expand_dims(y0, axis=3), [1,1,1,num_channel])
    x1 = tf.tile(tf.expand_dims(x1, axis=3), [1,1,1,num_channel])
    y1 = tf.tile(tf.expand_dims(y1, axis=3), [1,1,1,num_channel])

    Ia = tf.where(tf.logical_or(tf.logical_or(tf.less(x0, 0.), tf.greater(x0, max_x)),
                                    tf.logical_or(tf.less(y0, 0.), tf.greater(y0, max_y))),
                                    tf.zeros_like(x0),
                                    Ia)
    Ib = tf.where(tf.logical_or(tf.logical_or(tf.less(x0, 0.), tf.greater(x0, max_x)),
                                    tf.logical_or(tf.less(y1, 0.), tf.greater(y1, max_y))),
                                    tf.zeros_like(x0),
                                    Ib)
    Ic = tf.where(tf.logical_or(tf.logical_or(tf.less(x1, 0.), tf.greater(x1, max_x)),
                                    tf.logical_or(tf.less(y0, 0.), tf.greater(y0, max_y))),
                                    tf.zeros_like(x0),
                                    Ic)
    Id = tf.where(tf.logical_or(tf.logical_or(tf.less(x1, 0.), tf.greater(x1, max_x)),
                                    tf.logical_or(tf.less(y1, 0.), tf.greater(y1, max_y))),
                                    tf.zeros_like(x0),
                                    Id)
    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out

def get_transform(in_dims, in_res=None, out_dims=None, out_res=None, scale=None, flip=None, rotate=None, translate=None):
    """
    Computing a 2D transformation matrix that applies to the sampling grid

    Input
    -----
    - in_dims: tuple containing the input image size [height, width]
    - in_res: tuple containing the resolution of input image [height, width]
    - out_res: tuple containing the resolution of input image [height, width]
    - scale: tuple containing the scale factor along the (y, x) axis, by a factor of 1/scale.[height, width]
            scale>1 is zoom out and scale<1 is zoom in
    - flip: tuple containing the boolean flip indicator, whether to flip the image along the (y, x) axis.[height, width]
            False is not flip and True is flip
    - rotate: scalar, rotating rotate degrees clockwise across the xy plane.
            negative number is counter-clockwise and positive number is clockwise
    - translate: tuple containing the translating voxels along the (y, x) axis. Already in the [-1,1] space.
            negative number is right/bottom and positive number is left/top

    Returns
    -------
    - theta: the 2D transformation matrix of shape [2, 3]
    """

    # identity transform
    theta = np.eye(2, 3, dtype=np.float32)

    if in_res is None:
        in_res = np.ones(2)

    if out_dims is None:
        out_dims = in_dims

    if out_res is None:
        out_res = in_res

    t_dims = np.ones(2)
    t_res = 1 / t_dims

    # moving to normalized transformation space
    n_tform = get_transform_crop(in_dims, in_res, t_dims, t_res)
    theta = compose_transforms(theta, n_tform)

    # scaling transform
    if scale is not None:
        s_tform = np.array([
            [scale[0], 0, 0],
            [0, scale[1], 0]],
            dtype=np.float32)
        theta = compose_transforms(theta, s_tform)

    # flipping transform
    if flip is not None:
        flip_val = [1, 1]
        if flip[0]:
            flip_val[0] = -1
        if flip[1]:
            flip_val[1] = -1

        f_tform = np.array([
            [flip_val[0], 0, 0],
            [0, flip_val[1], 0]],
            dtype=np.float32)
        theta = compose_transforms(theta, f_tform)

    if rotate is not None:
        rotate_xy = rotate

        # rotation transform - across the yz axis
        if rotate_xy != 0:
            rxy_tform = np.array([
                [np.cos(np.deg2rad(rotate_xy)), -np.sin(np.deg2rad(rotate_xy)), 0],
                [np.sin(np.deg2rad(rotate_xy)), np.cos(np.deg2rad(rotate_xy)), 0]],
                dtype=np.float32)
            theta = compose_transforms(theta, rxy_tform)

    # moving back from normalized transformation space
    o_tform = get_transform_crop(t_dims, t_res, out_dims, out_res)
    theta = compose_transforms(theta, o_tform)

    # translation transform
    if translate is not None:
        t_tform = np.array([
            [1, 0, translate[0]*2],
            [0, 1, translate[1]*2]],
            dtype=np.float32)
        theta = compose_transforms(theta, t_tform)

    return theta


def get_transform_crop(in_dims, in_res, out_dims, out_res):
    """
    Computes the 2D transformation from one resolution to another
    assuming that the center of the image is (0,0) position.

    Input
    -----
    - in_dims: tuple containing the input image size (height, width)
    - in_res: tuple containing the input image pixel spacing (height, width)
    - out_dims: tuple containing the output image size (height, width)
    - out_res: tuple containing the output image pixel spacing (height, width)

    Returns
    -------
    - theta: the 2D transformation matrix of shape [2, 3]
    """

    scale = np.array(out_res) * np.array(out_dims) / np.array(in_res) / np.array(in_dims)
    theta = np.eye(2, 3, dtype=np.float32)
    theta[0, 0] = scale[0]
    theta[1, 1] = scale[1]
    return theta

def compose_transforms(trans1, trans2):
    """
    Combines two transformations into one

    Input
    -----
    - trans1: first transform matrix of shape [2 ,3] or [3, 3]
    - trans2: second transform matrix of shape [2 ,3] or [3, 3]

    Returns
    -------
    - theta: the 2D transformation matrix of shape [2, 3]
    """

    if (trans1.shape[0] == 2):
        trans1 = np.vstack([trans1, np.array([0, 0, 1])])
    if (trans2.shape[0] == 2):
        trans2 = np.vstack([trans2, np.array([0, 0, 1])])

    output = np.matmul(trans1, trans2)

    return output[0:2, :]

def normalize_translation(num_of_pixels, axis_size):
    """
    Convert the value of num_of_pixels from the image space to the [-1, 1] space (where the transformation matrix is applied)

    Input
    -----
    - num_of_pixels: the number of pixels that will be translated across an axis
    - axis_size: the size of the axis where the translation is applied

    Returns
    -------
    - out: the converted num_of_pixels
    """

    out = num_of_pixels * (2 / (axis_size - 1))

    return out


def normalize_points(points, axes_size, reverse=False):
    """
    Normalizing a list of points [x,y] from the image space to the the [-1, 1] space (where the transformation matrix is applied)

    Input
    -----
    - points: the list of points [x,y] in the image space (numpy array of shape = [B, 2])
    - axes_size: the size of each correspoding axis [width, height] (numpy array of shape = [2])
    - reverse: if True, reversing the operation

    Returns
    -------
    - out: the normalized coordinate
    """

    if not reverse:
        out = (2 * (points / (axes_size - 1))) - 1
    else:
        out = np.round(((points + 1) / 2) * (axes_size - 1))

    return out


def get_scale(res_in, res_out):
    """
    Computing the scaling factor

    Input
    -----
    - res_in: the resolution (pixel spacing) of the input image across an axis
    - res_out: the resolution (pixel spacing) of the output image across an axis

    Returns
    -------
    - out: the scaling factor
    """

    out = res_out / res_in

    return out
