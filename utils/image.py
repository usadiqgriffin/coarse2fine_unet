from scipy.ndimage.interpolation import affine_transform
import numpy as np
import cv2
import math
from numpy.linalg import inv


def get_bounding_box3d(mask, label):
    """ Finds the 3D bounding box which bounds all pixels in 'mask' with the value 'label'

    :param mask: a 3D mask
    :param label: a value in 'mask' which represents a label
    :returns box_start: a 3D coordinate representing the "start" conrner of the bounding box
    :returns box_end: a 3D coordinate representing the "end" corner of the bounding box

    """

    label_indices = np.argwhere(mask == label)
    arr0 = []
    arr1 = []
    arr2 = []
    for i in range(len(label_indices)):
        arr0.append(label_indices[i][0])
        arr1.append(label_indices[i][1])
        arr2.append(label_indices[i][2])

    box_start = [np.min(arr0), np.min(arr1), np.min(arr2)]
    box_end = [np.max(arr0), np.max(arr1), np.max(arr2)]

    return box_start, box_end


def warp_mask(mask, flow):
    """ Warped mask according to flow using nearest neighbor interpolation

    mask is assumed to be a numpy array of type bool
    flow is assumed to be of shape [1, height, width, 2]

    """

    height = flow.shape[1]
    width = flow.shape[2]
    warped_mask = np.zeros((height, width), np.bool)

    for i in range(height):

        for j in range(width):

            float_y = i - flow[0, i, j, 0]
            float_x = j - flow[0, i, j, 1]

            ceil_y = int(min(max(np.ceil(float_y), 0), height-1))
            floor_y = int(min(max(np.floor(float_y), 0), height-1))
            round_y = int(min(max(np.round(float_y), 0), height-1))

            ceil_x = int(min(max(np.ceil(float_x), 0), height-1))
            floor_x = int(min(max(np.floor(float_x), 0), height-1))
            round_x = int(min(max(np.round(float_x), 0), height-1))

            y_tie = ((float_y % 0.5) == 0) and ((float_y / 0.5) % 2 == 1)
            x_tie = ((float_x % 0.5) == 0) and ((float_x / 0.5) % 2 == 1)

            if y_tie and x_tie:
                warped_mask[i, j] = \
                    mask[floor_y, floor_x] * \
                    mask[floor_y, ceil_x] * \
                    mask[ceil_y, floor_x] * \
                    mask[ceil_y, ceil_x]
            elif y_tie:
                warped_mask[i, j] = \
                    mask[floor_y, round_x] * \
                    mask[ceil_y, round_x]
            elif x_tie:
                warped_mask[i, j] = \
                    mask[round_y, floor_x] * \
                    mask[round_y, ceil_x]
            else:
                warped_mask[i, j] = mask[round_y, round_x]

    return warped_mask


def dice(mask1, mask2):
    """ Computing the Dice between two binary masks

    mask1 and mask2 are assumed to be numpy array of type np.bool

    """
    summation = np.sum(mask1) + np.sum(mask2)
    if summation == 0:
        dice_score = 1
    else:
        dice_score = (2 * np.sum(mask1 * mask2)) / summation

    return dice_score


def bin_img(img, num_of_bins):
    """Generating a binned image using num_of_bins bins

    Assumes that img is in [0,255]

    Return an image in [0,255]
    """
    binned_img = np.copy(img)

    for i in range(binned_img.shape[0]):
        for j in range(binned_img.shape[1]):
            bin_num = int((binned_img[i, j] * num_of_bins) / 256)
            binned_img[i, j] = bin_num

    binned_img = adjust_range(binned_img)

    return binned_img


def patch_mi(img1, img2, num_of_bins, patch_size, patch_stride, patch_count):
    """Computes the mutual information of each of the patches in img1 and img2

    img1 and img2 values are in [0,255]
    img1 and img2 have shape [height, width]

    Rertrn a tensor of shape [patch_count, 1]
    """

    assert((patch_size - patch_stride) % 2 == 0)

    margin = int((patch_size - patch_stride) / 2)

    # padding img1 and img2 with the same padding algorithm that tf.extract_image_patches() is using (i.e. 'SAME')
    img1 = np.pad(img1, ((margin, margin), (margin, margin)), 'constant')
    img2 = np.pad(img2, ((margin, margin), (margin, margin)), 'constant')

    height = img1.shape[0]
    width = img1.shape[1]

    img_patch_mi = np.zeros([int(np.sqrt(patch_count)), int(np.sqrt(patch_count))])

    for i in range(0, height - patch_size + 1, patch_stride):
        for j in range(0, width - patch_size + 1, patch_stride):
            patch_val = mi(img1[i:i+patch_size, j:j+patch_size], img2[i:i+patch_size, j:j+patch_size], num_of_bins)
            img_patch_mi[int(i/patch_stride), int(j/patch_stride)] = patch_val

    return np.reshape(img_patch_mi, (patch_count, 1))


def mi(img1, img2, num_of_bins):
    """Computes the mutual information of img1 and img2

    img1 and img2 values are in [0,255]
    img1 and img2 have shape [height, width]
    """

    height = img1.shape[0]
    width = img1.shape[1]

    # joint histogram of img1 and img2
    joint_hist = np.zeros([num_of_bins, num_of_bins], np.float32)
    for i in range(height):
        for j in range(width):
            bin1 = int((img1[i,j] * num_of_bins) / 256)
            bin2 = int((img2[i,j] * num_of_bins) / 256)
            # histogram of img1
            img1_hist[bin1] = img1_hist[bin1] + 1
            # histogram of img2
            img2_hist[bin2] = img2_hist[bin2] + 1
            # joint histogram of img1 and img2
            joint_hist[bin1, bin2] = joint_hist[bin1, bin2] + 1
    joint_hist = joint_hist / float(height * width)

    # Computing the mutual information
    # TODO: Make this into mat operations
    mi = 0
    for i in range(num_of_bins):
        for j in range(num_of_bins):
            if joint_hist[i,j] == 0 or img1_hist[i] == 0 or img2_hist[j] == 0:
                continue
            mi = mi + joint_hist[i,j] * np.log(joint_hist[i,j] / (img1_hist[i] * img2_hist[j]))

    return mi


def joint_hist(img1, img2, num_of_bins):
    """Computes the joint histogram of img1 and img2

    img1 and img2 are in [0,255]
    img1 and img2 have shape [height, width]
    """

    height = img1.shape[0]
    width = img1.shape[1]

    # joint histogram of img1 and img2
    hist = np.zeros([num_of_bins, num_of_bins], np.float32)
    for i in range(height):
        for j in range(width):
            bin1 = int((img1[i, j] * num_of_bins) / 256)
            bin2 = int((img2[i, j] * num_of_bins) / 256)
            hist[bin1, bin2] = hist[bin1, bin2] + 1

    return hist


def adjust_depth(seg3d, target_depth):
    """Adjusts seg3d depth to target_depth."""

    # seg3d is of shape [depth, height, width]
    src_shape3d = seg3d.shape
    src_depth = src_shape3d[0]
    if src_depth < target_depth:
        pad_up = int((target_depth - src_depth) / 2)
        pad_down = target_depth - src_depth - pad_up
        seg3d = np.pad(seg3d, [[pad_up, pad_down],[0,0],[0,0]], mode='constant')
    elif src_depth > target_depth:
        crop_start = int((src_depth - target_depth) / 2)
        crop_end = crop_start + target_depth
        seg3d = seg3d[crop_start:crop_end, :, :]
    return seg3d


def get_transform2d(shape_in, res_in, res_out, rotate_degree, mirror, shift_y=0, shift_x=0):
    """Return the transformation matrix to be used by tf.contrib.transform



    This function must behave exactly the same as its implementation in mlc.cc!!!



    Input:
    shape_in: src image matrix size
    ** we assume that shape_in == shape_out since tf.contrib.image.transform does not currenlty support changing the output shape
    res_in: src image pixel resolution
    res_out: dst image pixel resolution
    rot_degree: rotation angle in degree (clockwise)
    mirror: boolean to decide if mirroring is required



    Output:
    trans_vec: used by tf.contrib.image.transform to rotate an image
    transform and offset: used to transform a coordinate
    canvas_shape: The shape of the image after being scaled and rotated
    ** The transform will be calculated/applied on canvas_shape and not shape_in



    Notes:
    - Test function is at ml_util_test.testGetTransform2D()
    - The image and coordinates, which the output is applied on, are assumed to be of numpy form (i.e. (y,x))



    Details:
    Each point (coordinate) p in the output image is transformed to pT + s
    where T and s are the transform matrix and offset passed to affine_transform function.
    If we want point c_out in the output to be mapped to and sampled from c_in from the input image,
    with rotation R and scaling S we need pT + s = (p-c_out)RS + c_in
    this yields to s = c_in - c_out * T (where T = R*S)
    """

    shift = [shift_y, shift_x]

    height_resized = shape_in[0] * res_in[0] / res_out[0]
    width_resized = shape_in[1] * res_in[1] / res_out[1]



    if rotate_degree != 0:
        # Need to make sure the rotated and scaled pixels do not fall outside of the canvas
        # And that no cropping is made on the original image
        hw_resized = 2 * (((height_resized / 2) ** 2 + (width_resized / 2) ** 2) ** 0.5)
        canvas_shape = np.array([max(int(hw_resized), shape_in[0]), max(int(hw_resized), shape_in[1])])
    else:
        # Just making sure that the scaled pixels do not fall outside of the canvas
        # And that no cropping is made on the original image
        canvas_shape = np.array([max(int(height_resized), shape_in[0]), max(int(width_resized), shape_in[1])])



    centre_in = 0.5 * canvas_shape
    centre_out = centre_in



    theta = rotate_degree * np.pi / 180
    rot = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])



    if (mirror):
        mirmtrx = np.array([[-1, 0],
                        [0, 1]])
        rot = np.matmul(rot, mirmtrx)



    scale_in = np.array([[res_in[0], 0],
                        [0, res_in[1]]])



    scale_out = np.array([[res_out[0], 0],
                        [0, res_out[1]]])



    transform = np.matmul(np.matmul(scale_out, rot), inv(scale_in))



    offset = centre_in - np.matmul(centre_out, transform) + shift



    trans_vec = np.array([
                    transform[1,1],
                    transform[0,1],
                    offset[1],
                    transform[1,0],
                    transform[0,0],
                    offset[0],
                    0,
                    0])



    return transform, offset, trans_vec, canvas_shape


def adjust_points(points, shape_in, shape_out):
    """Adjusting the coordinates in shape_in to the coordinates in shape_out.

    Assuming only padding/cropping is performed
    points = [[y1,x1],[y2,x2],[y3,x3],...]
    """

    y_factor = float(shape_out[0] - shape_in[0]) / 2
    x_factor = float(shape_out[1] - shape_in[1]) / 2

    points[:,0] = points[:,0] + y_factor
    points[:,1] = points[:,1] + x_factor

    return points




def crop2d(image, box_height, box_width):
    cropped = np.zeros(shape=[box_height, box_width])
    x_start = int(max((image.shape[1]/2) - (box_width/2), 0))
    y_start = int(max((image.shape[0]/2) - (box_height/2), 0))
    small_cropped = image[y_start : min(y_start+box_height, image.shape[0]),
                          x_start : min(x_start+box_width, image.shape[1])]
    x_start = int((box_width - small_cropped.shape[1])/2)
    y_start = int((box_height - small_cropped.shape[0])/2)
    cropped[y_start : y_start+small_cropped.shape[0],
            x_start : x_start+small_cropped.shape[1]] = small_cropped
    return cropped


def crop3d(image, box_height, box_width):
    cropped = np.zeros(shape=[image.shape[0], box_height, box_width])
    x_start = int(max((image.shape[2]/2) - (box_width/2), 0))
    y_start = int(max((image.shape[1]/2) - (box_height/2), 0))
    small_cropped = image[:,
                          y_start : min(y_start+box_height, image.shape[1]),
                          x_start : min(x_start+box_width, image.shape[2])]
    x_start = int((box_width - small_cropped.shape[2])/2)
    y_start = int((box_height - small_cropped.shape[1])/2)
    cropped[:,
            y_start : y_start+small_cropped.shape[1],
            x_start : x_start+small_cropped.shape[2]] = small_cropped
    return cropped


def his_equal(image):
    """Scaling and histogram equalization."""
    min_val = np.min(image)
    max_val = np.max(image)
    image = np.round(np.multiply(np.divide(np.subtract(image, min_val), (max_val-min_val)), 255))
    image = cv2.equalizeHist(image.astype('uint8'))

    return image


def his_equal3d2d(image3d):
    """Scaling and histogram equalization."""
    for i in range(image3d.shape[0]):
        min_val = np.min(image3d[i])
        max_val = np.max(image3d[i])
        image = np.round(np.multiply(np.divide(np.subtract(image3d[i], min_val), (max_val-min_val)), 255))
        image3d[i] = cv2.equalizeHist(image.astype('uint8'))
    return image3d


def his_equal3d(image3d):
    """Scaling and histogram equalization."""
    min_val = np.min(image3d)
    max_val = np.max(image3d)
    image3d = np.round(np.multiply(np.divide(np.subtract(image3d, min_val), (max_val-min_val)), 255))
    shape3d = image3d.shape
    shape2d = [shape3d[0] * shape3d[1], shape3d[2]]
    image3d = np.reshape(cv2.equalizeHist(np.reshape(image3d, shape2d).astype('uint8')), shape3d)
    return image3d


def adjust_range(image):
    min_val = np.min(image)
    max_val = np.max(image)

    if min_val == max_val:
        return image

    image = np.round(np.multiply(np.divide(np.subtract(image, min_val), (max_val-min_val)), 255))

    return image


def add_frame(img):
    out_img = img
    for i in range(img.shape[0]):
        img[i, 0] = 255
        img[i, img.shape[0]-1] = 255
    for j in range(img.shape[1]):
        img[0, j] = 255
        img[img.shape[1]-1, j] = 255
    return out_img


def adjust_range_3d2d(image3d):
    for i in range(image3d.shape[0]):
        min_val = np.min(image3d[i])
        max_val = np.max(image3d[i])
        image3d[i] = np.round(np.multiply(np.divide(np.subtract(image3d[i], min_val), (max_val-min_val)), 255))
    return image3d


def salt_pepper(cropped_image, prob):
    thres = 1 - prob
    pepper = np.min(cropped_image)
    salt = np.max(cropped_image)
    for j in range(cropped_image.shape[0]):
        for k in range(cropped_image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                cropped_image[j,k] = pepper
            elif rdn > thres:
                cropped_image[j,k] = salt

    return cropped_image


def salt_pepper3d(cropped_image3d, prob):
    thres = 1 - prob
    pepper = np.min(cropped_image3d)
    salt = np.max(cropped_image3d)
    for j in range(cropped_image3d.shape[0]):
        for k in range(cropped_image3d.shape[1]):
            for l in range(cropped_image3d.shape[2]):
                rdn = np.random.random()
                if rdn < prob:
                    cropped_image3d[j,k,l] = pepper
                elif rdn > thres:
                    cropped_image3d[j,k,l] = salt

    return cropped_image3d


def gaussian_noise(cropped_image, threshold):

    val = np.random.randint(0, 100) / 100.0

    if (val < threshold):
        sigma = np.random.randint(0, 5) * 255.0 / 100.0
        mean = 0.0

        pixel_count = cropped_image.shape[0] * cropped_image.shape[1]
        gauss = np.random.normal(mean, sigma, pixel_count)
        gauss = gauss.reshape(cropped_image.shape[0], cropped_image.shape[1])
        cropped_image = cropped_image + gauss

    cropped_image = adjust_range(cropped_image)

    return cropped_image


def gamma_corr(cropped_image, thresh):

    max_val = np.max(cropped_image)

    val = np.random.randint(0, 100) / 100.0

    if (val < thresh):
        if np.random.random() < 0.5:
            gamma = np.random.randint(15, 100) / 100.0
        else:
            gamma = np.random.randint(100, 115) / 100.0

        for j in range(cropped_image.shape[0]):
            for k in range(cropped_image.shape[1]):
                pix_value = cropped_image[j,k]
                pix_value = (pix_value/float(max_val))**gamma * float(max_val)
                cropped_image[j,k] = round(pix_value)

    cropped_image = adjust_range(cropped_image)

    return cropped_image


def speckle_noise(cropped_image, threshold):

    val = np.random.randint(0, 100) / 100.0

    if (val < threshold):
        sigma = np.random.randint(0, 25) / 1000.0
        mean = 0.0

        for j in range(cropped_image.shape[0]):
            for k in range(cropped_image.shape[1]):
                pix_value = cropped_image[j,k]
                speckle = np.random.normal(mean, sigma)
                cropped_image[j,k] = round(pix_value + float(pix_value) * speckle)

    cropped_image = adjust_range(cropped_image)

    return cropped_image


def warp_affine_with_rotation_matrix2d(src, angle, scale, mirror, src_size, dst_size, isSrcImage, inter):

    (h, w) = src_size
    (cX, cY) = (w // 2, h // 2) # center of source

    (h_dst, w_dst) = dst_size

    if (isSrcImage):

        # get rotation matrix for rotating the image around its center
        M = cv2.getRotationMatrix2D((cX,cY), angle, scale)

        # adjust the rotation matrix to take into account translation
        M[0, 2] += w_dst / 2 - cX
        M[1, 2] += h_dst / 2 - cY

        # apply updated transformation to the image and points
        if inter == 'linear':
            dst = cv2.warpAffine(np.float32(src),M,(w_dst,h_dst),flags=cv2.INTER_LINEAR) #(nW,nH)
        elif inter == 'nearest':
            dst = cv2.warpAffine(np.float32(src),M,(w_dst,h_dst),flags=cv2.INTER_NEAREST) #(nW,nH)

        if (mirror):
            dst = np.fliplr(dst)

    else:

        M_point = cv2.getRotationMatrix2D((cY,cX), -angle, scale)
        dst = np.dot((M_point), (np.append(src, np.ones((1,src.shape[1])), axis=0)))
        dst += np.repeat([[(h_dst / 2) - cY], [(w_dst / 2) - cX]],src.shape[1],axis=1)
        if mirror:
            dst[1,:] = w_dst - dst[1,:]

    return dst


def rotate_all_axes(image3d, seg3d, zx_zy_angle_range, threshold):

    # zx_zy_angle_range should be in radians

    seg3d_endo = np.zeros(shape = seg3d.shape)
    seg3d_epi = np.zeros(shape = seg3d.shape)
    seg3d_rv_endo = np.zeros(shape = seg3d.shape)

    for i in range(seg3d.shape[0]):
        for j in range(seg3d.shape[1]):
            for k in range(seg3d.shape[2]):
                if seg3d[i,j,k] == 1:
                    seg3d_endo[i,j,k] = 1
                elif seg3d[i,j,k] == 2:
                    seg3d_epi[i,j,k] = 1
                elif seg3d[i,j,k] == 3:
                    seg3d_rv_endo[i,j,k] = 1


    # Defining the scale and affine transform matrices
    theta1 = (np.random.random() * 2 * zx_zy_angle_range) - zx_zy_angle_range
    t1 = np.array([[math.cos(theta1), 0, -math.sin(theta1)],
                    [0, 1, 0],
                    [math.sin(theta1), 0, math.cos(theta1)]])

    theta2 = (np.random.random() * 2 * zx_zy_angle_range) - zx_zy_angle_range
    t2 = np.array([[math.cos(theta2), -math.sin(theta2), 0],
                    [math.sin(theta2), math.cos(theta2), 0],
                    [0, 0, 1]])

    theta3 = np.random.random() * (2 * math.pi)
    t3 = np.array([[1, 0, 0],
                    [0, math.cos(theta3), -math.sin(theta3)],
                    [0, math.sin(theta3), math.cos(theta3)]])

    T = np.matmul(np.matmul(t1, t2), t3)

    scale1 = np.array([[10, 0, 0],
                        [0, 1.855, 0],
                        [0, 0, 1.855]])

    scale2 = np.array([[1/10, 0, 0],
                        [0, 1/1.855, 0],
                        [0, 0, 1/1.855]])

    tform = np.matmul(np.matmul(scale1, T), scale2)

    c_in = 0.5 * np.array(image3d.shape)
    c_out = c_in #np.array([10, 200, 200])
    offset = c_in - c_out.dot(tform)

    image3d_r = affine_transform(image3d, tform.T, offset = offset) #, output_shape = [20, 400, 400])
    seg3d_endo_r = affine_transform(seg3d_endo, tform.T, offset = offset)
    seg3d_epi_r = affine_transform(seg3d_epi, tform.T, offset = offset)
    seg3d_rv_endo_r = affine_transform(seg3d_rv_endo, tform.T, offset = offset)

    # Computing seg3d_r using seg3d_endo_r, seg3d_epi_r, seg3d_rv_endo_r
    seg3d_r = np.zeros(shape = image3d_r.shape)
    for i in range(image3d_r.shape[0]):
        for j in range(image3d_r.shape[1]):
            for k in range(image3d_r.shape[2]):
                if seg3d_endo_r[i,j,k] > threshold:
                    seg3d_r[i,j,k] = 1
                elif seg3d_epi_r[i,j,k] > threshold:
                    seg3d_r[i,j,k] = 2
                elif seg3d_rv_endo_r[i,j,k] > threshold:
                    seg3d_r[i,j,k] = 3

    return image3d_r, seg3d_r


def rotate_zx_zy_axes(image3d, seg3d, zx_angle, zy_angle, threshold):

    # zx_zy_angle_range should be in radians

    seg3d_endo = np.zeros(shape = seg3d.shape)
    seg3d_epi = np.zeros(shape = seg3d.shape)
    seg3d_rv_endo = np.zeros(shape = seg3d.shape)

    for i in range(seg3d.shape[0]):
        for j in range(seg3d.shape[1]):
            for k in range(seg3d.shape[2]):
                if seg3d[i,j,k] == 1:
                    seg3d_endo[i,j,k] = 1
                elif seg3d[i,j,k] == 2:
                    seg3d_epi[i,j,k] = 1
                elif seg3d[i,j,k] == 3:
                    seg3d_rv_endo[i,j,k] = 1


    # Defining the scale and affine transform matrices
    theta1 = zx_angle
    t1 = np.array([[math.cos(theta1), 0, -math.sin(theta1)],
                    [0, 1, 0],
                    [math.sin(theta1), 0, math.cos(theta1)]])

    theta2 = zy_angle
    t2 = np.array([[math.cos(theta2), -math.sin(theta2), 0],
                    [math.sin(theta2), math.cos(theta2), 0],
                    [0, 0, 1]])

    T = np.matmul(t1, t2)

    scale1 = np.array([[10, 0, 0],
                        [0, 1.855, 0],
                        [0, 0, 1.855]])

    scale2 = np.array([[1/10, 0, 0],
                        [0, 1/1.855, 0],
                        [0, 0, 1/1.855]])

    tform = np.matmul(np.matmul(scale1, T), scale2)

    c_in = 0.5 * np.array(image3d.shape)
    c_out = c_in #np.array([10, 200, 200])
    offset = c_in - c_out.dot(tform)

    image3d_r = affine_transform(image3d, tform.T, offset = offset) #, output_shape = [20, 400, 400])
    seg3d_endo_r = affine_transform(seg3d_endo, tform.T, offset = offset)
    seg3d_epi_r = affine_transform(seg3d_epi, tform.T, offset = offset)
    seg3d_rv_endo_r = affine_transform(seg3d_rv_endo, tform.T, offset = offset)

    # Computing seg3d_r using seg3d_endo_r, seg3d_epi_r, seg3d_rv_endo_r
    seg3d_r = np.zeros(shape = image3d_r.shape)
    for i in range(image3d_r.shape[0]):
        for j in range(image3d_r.shape[1]):
            for k in range(image3d_r.shape[2]):
                if seg3d_endo_r[i,j,k] > threshold:
                    seg3d_r[i,j,k] = 1
                elif seg3d_epi_r[i,j,k] > threshold:
                    seg3d_r[i,j,k] = 2
                elif seg3d_rv_endo_r[i,j,k] > threshold:
                    seg3d_r[i,j,k] = 3

    return image3d_r, seg3d_r
