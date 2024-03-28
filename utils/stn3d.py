import tensorflow as tf
import numpy as np


def spatial_transformer_network(input_fmap, theta, out_dims=None, inter='trilinear', **kwargs):
    """
    A 3D extension of the Spatial Transformer Network layer implementation as described in [1].

    The layer is composed of 3 elements:

    - localization_net: takes the original image as input and outputs
      the parameters of the affine transformation that should be applied
      to the input image.

    - affine_grid_generator: generates a grid of (z, y, x) coordinates that
      correspond to a set of points where the input should be sampled
      to produce the transformed output.

    - sampler: takes as input the original image and the grid
      and produces the output image using trilinear/nearest interpolation.


    Input
    -----
    - input_fmap: output of the previous layer. Can be input if spatial
      transformer layer is at the beginning of architecture. Should be
      a tensor of shape (B, D, H, W, C).

    - theta: affine transform tensor of shape (B, 12). Permits scaling,
      flipping, rotation, and translation. Initialize to identity matrix.
      It is the output of the localization network.

    Returns
    -------
    - out_fmap: transformed input feature map. Tensor of size (B, D, H, W, C).

    Notes
    -----
    [1]: 'Spatial Transformer Networks', Jaderberg et. al,
         (https://arxiv.org/abs/1506.02025)

    """
    # grab input dimensions
    D = tf.shape(input_fmap)[1]
    H = tf.shape(input_fmap)[2]
    W = tf.shape(input_fmap)[3]

    # generate grids of same size or upsample/downsample if specified
    if out_dims is None:
        batch_grids = affine_grid_generator(D, H, W, theta)
    else:
        out_D = out_dims[0]
        out_H = out_dims[1]
        out_W = out_dims[2]
        batch_grids = affine_grid_generator(out_D, out_H, out_W, theta)

    z_s = batch_grids[:, 0, :, :, :]
    y_s = batch_grids[:, 1, :, :, :]
    x_s = batch_grids[:, 2, :, :, :]

    # sample input with grid to get output
    out_fmap = sampler(input_fmap, z_s, y_s, x_s, inter)

    return out_fmap


def get_pixel_value(img, z, y, x):
    """
    Utility function to get pixel value for coordinate
    vectors z, y, and x from a 5D tensor image.

    Input
    -----
    - img: tensor of shape (B, D, H, W, C)
    - z: flattened tensor of shape (B*D*H*W)
    - y: flattened tensor of shape (B*D*H*W)
    - x: flattened tensor of shape (B*D*H*W)
    
    Returns
    -------
    - output: tensor of shape (B, D, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    depth = shape[1]
    height = shape[2]
    width = shape[3]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1, 1))
    b = tf.tile(batch_idx, (1, depth, height, width))

    indices = tf.stack([b, z, y, x], 4)

    return tf.gather_nd(img, indices)


def affine_grid_generator(depth, height, width, theta):
    """
    This function returns a sampling grid, which when
    used with the trilinear sampler on the input feature
    map, will create an output feature map that is an
    affine transformation [1] of the input feature map.

    Input
    -----
    - depth: desired depth of grid/output. Used
      to downsample or upsample.

    - height: desired height of grid/output. Used
      to downsample or upsample.

    - width: desired width of grid/output. Used
      to downsample or upsample.

    - theta: affine transform matrices of shape (num_batch, 3, 4).
      For each image in the batch, we have 12 theta parameters of
      the form (3x4) that define the affine transformation T.

    Returns
    -------
    - normalized grid (-1, 1) of shape (num_batch, 2, D, H, W).
      The 2nd dimension has 2 components: (z, y, x) which are the
      sampling points of the original image for each point in the
      target image.

    Note
    ----
    [1]: the affine transformation allows cropping, translation,
         and isotropic scaling.
    """
    num_batch = tf.shape(theta)[0]

    # create normalized 2D grid
    z = tf.linspace(-1.0, 1.0, depth)
    y = tf.linspace(-1.0, 1.0, height)
    x = tf.linspace(-1.0, 1.0, width)

    y_t, z_t, x_t = tf.meshgrid(y, z, x)

    # flatten
    z_t_flat = tf.reshape(z_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])
    x_t_flat = tf.reshape(x_t, [-1])

    # reshape to [z_t, y_t, x_t, 1] - (homogeneous form)
    ones = tf.ones_like(z_t_flat)
    sampling_grid = tf.stack([z_t_flat, y_t_flat, x_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')

    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 3, D*H*W)

    # reshape to (num_batch, 3, D, H, W)
    batch_grids = tf.reshape(batch_grids, [num_batch, 3, depth, height, width])

    return batch_grids


def sampler(img, z, y, x, inter):
    """
    Performs trilinear/nearest sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.

    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.

    Input
    -----
    - img: batch of images in (B, D, H, W, C) layout.
    - grid: z, y, x which is the output of affine_grid_generator.
    - inter: interpolation used - trilinear or nearest

    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    D = tf.shape(img)[1]
    H = tf.shape(img)[2]
    W = tf.shape(img)[3]

    max_z = tf.cast(D - 1, 'int32')
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')

    zero = tf.zeros([], dtype='int32')

    # rescale z, y, and x [0, D-1/H-1/W-1]
    z = tf.cast(z, 'float32')
    y = tf.cast(y, 'float32')
    x = tf.cast(x, 'float32')

    z = 0.5 * ((z + 1.0) * tf.cast(max_z, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y, 'float32'))
    x = 0.5 * ((x + 1.0) * tf.cast(max_x, 'float32'))

    # grab 8 nearest corner points for each (z_i, y_i, x_i)
    z0_ = tf.floor(z)
    z1_ = z0_ + 1.0
    y0_ = tf.floor(y)
    y1_ = y0_ + 1.0
    x0_ = tf.floor(x)
    x1_ = x0_ + 1.0

    # calculate deltas
    wa = (x1_-x) * (y1_-y) * (z1_-z)
    wb = (x1_-x) * (y-y0_) * (z-z0_)
    wc = (x-x0_) * (y1_-y) * (z1_-z)
    wd = (x-x0_) * (y-y0_) * (z-z0_)
    we = (x1_-x) * (y1_-y) * (z-z0_)
    wf = (x1_-x) * (y-y0_) * (z1_-z)
    wg = (x-x0_) * (y1_-y) * (z-z0_)
    wh = (x-x0_) * (y-y0_) * (z1_-z)

    if inter == 'nearest':
        wa = tf.cast((wa >= wb) & (wa >= wc) & (wa >= wd) & (wa >= we) & (wa >= wf) & (wa >= wg) & (wa >= wh), 'float32')
        wb = tf.cast((wb >  wa) & (wb >= wc) & (wb >= wd) & (wb >= we) & (wb >= wf) & (wb >= wg) & (wb >= wh), 'float32')
        wc = tf.cast((wc >  wa) & (wc >  wb) & (wc >= wd) & (wc >= we) & (wc >= wf) & (wc >= wg) & (wc >= wh), 'float32')
        wd = tf.cast((wd >  wa) & (wd >  wb) & (wd >  wc) & (wd >= we) & (wd >= wf) & (wd >= wg) & (wd >= wh), 'float32')
        we = tf.cast((we >  wa) & (we >  wb) & (we >  wc) & (we >  wd) & (we >= wf) & (we >= wg) & (we >= wh), 'float32')
        wf = tf.cast((wf >  wa) & (wf >  wb) & (wf >  wc) & (wf >  wd) & (wf >  we) & (wf >= wg) & (wf >= wh), 'float32')
        wg = tf.cast((wg >  wa) & (wg >  wb) & (wg >  wc) & (wg >  wd) & (wg >  we) & (wg >  wf) & (wg >= wh), 'float32')
        wh = tf.cast((wh >  wa) & (wh >  wb) & (wh >  wc) & (wh >  wd) & (wh >  we) & (wh >  wf) & (wh >  wg), 'float32')

    z0 = tf.cast(z0_, 'int32')
    z1 = tf.cast(z1_, 'int32')
    y0 = tf.cast(y0_, 'int32')
    y1 = tf.cast(y1_, 'int32')
    x0 = tf.cast(x0_, 'int32')
    x1 = tf.cast(x1_, 'int32')

    # clip to range [0, D-1/H-1/W-1] to not violate img boundaries
    z0 = tf.clip_by_value(z0, zero, max_z)
    z1 = tf.clip_by_value(z1, zero, max_z)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, z0, y0, x0)
    Ib = get_pixel_value(img, z1, y1, x0)
    Ic = get_pixel_value(img, z0, y0, x1)
    Id = get_pixel_value(img, z1, y1, x1)
    Ie = get_pixel_value(img, z1, y0, x0)
    If = get_pixel_value(img, z0, y1, x0)
    Ig = get_pixel_value(img, z1, y0, x1)
    Ih = get_pixel_value(img, z0, y1, x1)



    max_z = tf.cast(max_z, 'float32')
    max_y = tf.cast(max_y, 'float32')
    max_x = tf.cast(max_x, 'float32')

    x0_ = tf.expand_dims(x0_, axis=4)
    y0_ = tf.expand_dims(y0_, axis=4)
    z0_ = tf.expand_dims(z0_, axis=4)
    x1_ = tf.expand_dims(x1_, axis=4)
    y1_ = tf.expand_dims(y1_, axis=4)
    z1_ = tf.expand_dims(z1_, axis=4)


    Ia = tf.where(tf.logical_or(tf.less(z0_, 0.), tf.greater(z0_, max_z)), tf.zeros_like(z0_), Ia)
    Ia = tf.where(tf.logical_or(tf.less(y0_, 0.), tf.greater(y0_, max_y)), tf.zeros_like(z0_), Ia)
    Ia = tf.where(tf.logical_or(tf.less(x0_, 0.), tf.greater(x0_, max_x)), tf.zeros_like(z0_), Ia)

    Ib = tf.where(tf.logical_or(tf.less(z1_, 0.), tf.greater(z1_, max_z)), tf.zeros_like(z0_), Ib)
    Ib = tf.where(tf.logical_or(tf.less(y1_, 0.), tf.greater(y1_, max_y)), tf.zeros_like(z0_), Ib)
    Ib = tf.where(tf.logical_or(tf.less(x0_, 0.), tf.greater(x0_, max_x)), tf.zeros_like(z0_), Ib)

    Ic = tf.where(tf.logical_or(tf.less(z0_, 0.), tf.greater(z0_, max_z)), tf.zeros_like(z0_), Ic)
    Ic = tf.where(tf.logical_or(tf.less(y0_, 0.), tf.greater(y0_, max_y)), tf.zeros_like(z0_), Ic)
    Ic = tf.where(tf.logical_or(tf.less(x1_, 0.), tf.greater(x1_, max_x)), tf.zeros_like(z0_), Ic)

    Id = tf.where(tf.logical_or(tf.less(z1_, 0.), tf.greater(z1_, max_z)), tf.zeros_like(z0_), Id)
    Id = tf.where(tf.logical_or(tf.less(y1_, 0.), tf.greater(y1_, max_y)), tf.zeros_like(z0_), Id)
    Id = tf.where(tf.logical_or(tf.less(x1_, 0.), tf.greater(x1_, max_x)), tf.zeros_like(z0_), Id)

    Ie = tf.where(tf.logical_or(tf.less(z1_, 0.), tf.greater(z1_, max_z)), tf.zeros_like(z0_), Ie)
    Ie = tf.where(tf.logical_or(tf.less(y0_, 0.), tf.greater(y0_, max_y)), tf.zeros_like(z0_), Ie)
    Ie = tf.where(tf.logical_or(tf.less(x0_, 0.), tf.greater(x0_, max_x)), tf.zeros_like(z0_), Ie)

    If = tf.where(tf.logical_or(tf.less(z0_, 0.), tf.greater(z0_, max_z)), tf.zeros_like(z0_), If)
    If = tf.where(tf.logical_or(tf.less(y1_, 0.), tf.greater(y1_, max_y)), tf.zeros_like(z0_), If)
    If = tf.where(tf.logical_or(tf.less(x0_, 0.), tf.greater(x0_, max_x)), tf.zeros_like(z0_), If)

    Ig = tf.where(tf.logical_or(tf.less(z1_, 0.), tf.greater(z1_, max_z)), tf.zeros_like(z0_), Ig)
    Ig = tf.where(tf.logical_or(tf.less(y0_, 0.), tf.greater(y0_, max_y)), tf.zeros_like(z0_), Ig)
    Ig = tf.where(tf.logical_or(tf.less(x1_, 0.), tf.greater(x1_, max_x)), tf.zeros_like(z0_), Ig)

    Ih = tf.where(tf.logical_or(tf.less(z0_, 0.), tf.greater(z0_, max_z)), tf.zeros_like(z0_), Ih)
    Ih = tf.where(tf.logical_or(tf.less(y1_, 0.), tf.greater(y1_, max_y)), tf.zeros_like(z0_), Ih)
    Ih = tf.where(tf.logical_or(tf.less(x1_, 0.), tf.greater(x1_, max_x)), tf.zeros_like(z0_), Ih)



    # add dimension for addition
    wa = tf.expand_dims(wa, axis=4)
    wb = tf.expand_dims(wb, axis=4)
    wc = tf.expand_dims(wc, axis=4)
    wd = tf.expand_dims(wd, axis=4)
    we = tf.expand_dims(we, axis=4)
    wf = tf.expand_dims(wf, axis=4)
    wg = tf.expand_dims(wg, axis=4)
    wh = tf.expand_dims(wh, axis=4)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id, we*Ie, wf*If, wg*Ig, wh*Ih])

    return out


def get_transform(in_dims, in_res=None, out_dims=None, out_res=None, scale=None, flip=None, rotate=None, translate=None):
    """
    Computing a 3D transformation matrix

    Input
    -----
    - scale: scaling the z/y/x axis by a factor of 1/scale
    - flip: boolean array, whether to flip the image along the z/y/x axis
    - rotate: rotating rotate degrees clockwise/counter-clockwise/counter-clockwise across the xy/xz/yz plane
    - translate: translating translate voxels along the z/y/x axis (in the direction of ceiling/top/left). Already in the [-1,1] space.

    Returns
    -------
    - theta: the 3D transformation matrix of shape [3, 4]
    """

    # identity transform
    theta = np.eye(3, 4, dtype=np.float32)

    if in_res is None:
        in_res = np.ones(3)
    
    if out_dims is None:
        out_dims = in_dims

    if out_res is None:
        out_res = in_res

    t_dims = np.ones(3)
    t_res = 1 / t_dims

    # moving to normalized transformation space
    n_tform = get_transform_crop(in_dims, in_res, t_dims, t_res)
    theta = compose_transforms(theta, n_tform)

    # scaling transform
    if scale is not None:
        s_tform = np.array([
            [scale[0], 0, 0, 0],
            [0, scale[1], 0, 0],
            [0, 0, scale[2], 0]],
            dtype=np.float32)
        theta = compose_transforms(theta, s_tform)

    # flipping transform
    if flip is not None:
        flip_val = [1, 1, 1]
        if flip[0]:
            flip_val[0] = -1
        if flip[1]:
            flip_val[1] = -1
        if flip[2]:
            flip_val[2] = -1

        f_tform = np.array([
            [flip_val[0], 0, 0, 0],
            [0, flip_val[1], 0, 0],
            [0, 0, flip_val[2], 0]],
            dtype=np.float32)
        theta = compose_transforms(theta, f_tform)

    if rotate is not None:
        rotate_xy = rotate[0]
        rotate_xz = rotate[1]
        rotate_yz = rotate[2]

        # rotation transform - across the xy axis
        if rotate_xy != 0:
            rxy_tform = np.array([
                [1, 0, 0, 0],
                [0, np.cos(np.deg2rad(rotate_xy)), -np.sin(np.deg2rad(rotate_xy)), 0],
                [0, np.sin(np.deg2rad(rotate_xy)), np.cos(np.deg2rad(rotate_xy)), 0]],
                dtype=np.float32)
            theta = compose_transforms(theta, rxy_tform)

        # rotation transform - across the xz axis
        if rotate_xz != 0:
            rxz_tform = np.array([
                [np.cos(np.deg2rad(rotate_xz)), 0, np.sin(np.deg2rad(rotate_xz)), 0],
                [0, 1, 0, 0],
                [-np.sin(np.deg2rad(rotate_xz)), 0, np.cos(np.deg2rad(rotate_xz)), 0]],
                dtype=np.float32)
            theta = compose_transforms(theta, rxz_tform)

        # rotation transform - across the yz axis
        if rotate_yz != 0:
            ryz_tform = np.array([
                [np.cos(np.deg2rad(rotate_yz)), -np.sin(np.deg2rad(rotate_yz)), 0, 0],
                [np.sin(np.deg2rad(rotate_yz)), np.cos(np.deg2rad(rotate_yz)), 0, 0],
                [0, 0, 1, 0]],
                dtype=np.float32)
            theta = compose_transforms(theta, ryz_tform)

    # translation transform
    if translate is not None:
        t_tform = np.array([
            [1, 0, 0, translate[0]],
            [0, 1, 0, translate[1]],
            [0, 0, 1, translate[2]]],
            dtype=np.float32)
        theta = compose_transforms(theta, t_tform)

    # moving back from normalized transformation space
    o_tform = get_transform_crop(t_dims, t_res, out_dims, out_res)
    theta = compose_transforms(theta, o_tform)

    return theta


def get_transform_crop(in_dims, in_res, out_dims, out_res):
    """
    Computes the 3D transformation from one resolution to another
    assuming that the center of the image is (0,0) position.

    Input
    -----
    - in_dims: triple containing the input image size (depth, height, width)
    - in_res: triple containing the input image pixel spacing (depth, height, width)
    - out_dims: triple containing the output image size (depth, height, width)
    - out_res: triple containing the output image pixel spacing (depth, height, width)

    Returns
    -------
    - theta: the 3D transformation matrix of shape [3, 4]
    """

    scale = out_res * out_dims / in_res / in_dims
    theta = np.eye(3, 4, dtype=np.float32)
    theta[0, 0] = scale[0]
    theta[1, 1] = scale[1]
    theta[2, 2] = scale[2]

    return theta


def compose_transforms(trans1, trans2):
    """
    Combines two transformations into one

    Input
    -----
    - trans1: first transform matrix of shape [3 ,4] or [4, 4]
    - trans2: second transform matrix of shape [3 ,4] or [4, 4]

    Returns
    -------
    - theta: the 3D transformation matrix of shape [3, 4]
    """

    if (trans1.shape[0] == 3):
        trans1 = np.vstack([trans1, np.array([0, 0, 0, 1])])
    if (trans2.shape[0] == 3):
        trans2 = np.vstack([trans2, np.array([0, 0, 0, 1])])

    output = np.matmul(trans1, trans2)

    return output[0:3, :]


def normalize_translation(num_of_voxels, axis_size):
    """
    Convert the value of num_of_voxels from the image space to the [-1, 1] space (where the transformation matrix is applied)

    Input
    -----
    - num_of_voxels: the number of voxels that will be translated across an axis
    - axis_size: the size of the axis where the translation is applied

    Returns
    -------
    - out: the converted num_of_voxels
    """

    out = num_of_voxels * (2 / (axis_size - 1))

    return out


def normalize_points(points, axes_size, reverse=False):
    """
    Normalizing a list of points [z,y,x] from the image space to the the [-1, 1] space (where the transformation matrix is applied)

    Input
    -----
    - points: the list of points [z,y,x] in the image space (numpy array of shape = [B, 3])
    - axes_size: the size of each correspoding axis [depth, height, width] (numpy array of shape = [3])
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
