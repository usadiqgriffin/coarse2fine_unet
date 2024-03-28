import numpy as np
from numpy.linalg import inv
import scipy
import utils.image as im_util

import tensorflow as tf
from utils.layers import AffineTransform, HistEqual

def test_get_transform2d():

    image1 = np.load('/mnt/SSD_1/final_seg/sax_biobank/1001553/1_14_image.npy')
    header1 = np.load('/mnt/SSD_1/final_seg/sax_biobank/1001553/1_14_header.npy').tolist()
    seg1 = np.load('/mnt/SSD_1/final_seg/sax_biobank/1001553/1_14_seg.npy')
    #image1 = np.load('/mnt/SSD_1/final_seg/sax_atlas/CAMRI-AA02_027Y/4_14_image.npy')
    #header1 = np.load('/mnt/SSD_1/final_seg/sax_atlas/CAMRI-AA02_027Y/4_14_header.npy').tolist()
    #seg1 = np.load('/mnt/SSD_1/final_seg/sax_atlas/CAMRI-AA02_027Y/4_14_seg.npy')

    image2 = np.load('/mnt/SSD_1/final_seg/sax_biobank/1001553/18_14_image.npy')
    header2 = np.load('/mnt/SSD_1/final_seg/sax_biobank/1001553/18_14_header.npy').tolist()
    seg2 = np.load('/mnt/SSD_1/final_seg/sax_biobank/1001553/18_14_seg.npy')

    scipy.misc.imsave('debug/image1.jpg', image1)
    scipy.misc.imsave('debug/seg1.jpg', seg1)

    scipy.misc.imsave('debug/image2.jpg', image2)
    scipy.misc.imsave('debug/seg2.jpg', seg2)

    crop_shape = np.array([192, 192], dtype=np.int32)

    ### Defining the graph - Start ###
    x = tf.placeholder(tf.float32, shape=(None, None, None, 1))
    y = tf.placeholder(tf.int64, shape=(None, None, None))
    canvas_shape_ph = tf.placeholder(tf.int32, shape=(2))
    trans_ph = tf.placeholder(tf.float32, shape=(8))
    x_hist = HistEqual(x)
    x_cropped = AffineTransform(x_hist, canvas_shape_ph, trans_ph, 'BILINEAR', crop_shape)
    y_cropped = AffineTransform(tf.expand_dims(y, -1), canvas_shape_ph, trans_ph, 'NEAREST', crop_shape)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session = tf.Session(config = config)
    session.run(tf.global_variables_initializer())
    ### Defining the graph - End ###

    images = np.empty(shape = [2, image1.shape[0], image1.shape[1], 1])
    images[0,:,:,0] = image1
    images[1,:,:,0] = image2

    segs = np.empty(shape = [2, image1.shape[0], image1.shape[1]])
    segs[0,:,:] = seg1
    segs[1,:,:] = seg2

    x_hist = session.run(x_hist, feed_dict={x: images})

    scipy.misc.imsave('debug/image1_his.jpg', x_hist[0,:,:,0])
    scipy.misc.imsave('debug/image2_his.jpg', x_hist[1,:,:,0])

    ############################################

    scipy.misc.imsave('debug/image1_hist_opencv.jpg', im_util.his_equal(image1))

    diff = x_hist[0,:,:,0] - im_util.his_equal(image1)
    scipy.misc.imsave('debug/image1_hist_diff.jpg', diff)

    ############################################

    ### Rotation, Scaling, and Mirroring ###
    shape_in = np.array(np.shape(image1))
    res_in = np.array([float(header1['pixel_height']), float(header1['pixel_width'])])
    #res_in = np.array([3.4, 3.4])
    res_out = np.array([1.855, 1.855])
    rotate_degree = 0#np.random.random() * 360
    mirror = False
    #if np.random.random() < 0.5:
    #    mirror = True

    transform, offset, trans_vec, canvas_shape = im_util.getTransform2D(shape_in, res_in, res_out, rotate_degree, mirror)

    cropped_images, cropped_segs = session.run([x_cropped, y_cropped], feed_dict={
                        x: images,
                        y: segs,
                        canvas_shape_ph: canvas_shape,
                        trans_ph: trans_vec})


    scipy.misc.imsave('debug/cropped_image1.jpg', cropped_images[0,:,:,0])
    scipy.misc.imsave('debug/cropped_seg1.jpg', cropped_segs[0,:,:,0])

    scipy.misc.imsave('debug/cropped_image2.jpg', cropped_images[1,:,:,0])
    scipy.misc.imsave('debug/cropped_seg2.jpg', cropped_segs[1,:,:,0])

    ############################################

    # Testing transformed points
    points = np.array([[108,90], [100,90], [104, 93]], dtype=np.float)
    image_p = np.array(image1)
    image_p[int(points[0,0]),int(points[0,1])] = 0
    image_p[int(points[1,0]),int(points[1,1])] = 0
    image_p[int(points[2,0]),int(points[2,1])] = 0
    scipy.misc.imsave('debug/image1_p.jpg', image_p)

    points = im_util.adjust_points(points, image_p.shape, canvas_shape)
    points = np.matmul((points - offset), inv(transform))
    points = im_util.adjust_points(points, canvas_shape, crop_shape)

    cropped_image_p = np.array(cropped_images[0,:,:,0])
    cropped_image_p[int(points[0,0]),int(points[0,1])] = 0
    cropped_image_p[int(points[1,0]),int(points[1,1])] = 0
    cropped_image_p[int(points[2,0]),int(points[2,1])] = 0
    scipy.misc.imsave('debug/cropped_image1_p.jpg', cropped_image_p)

    points[:,0] = points[:,0] / crop_shape[0] # height (y)
    points[:,1] = points[:,1] / crop_shape[1] # width (x)
    points = np.reshape(points, -1)
    print('points =', points)


def main():
    testGetTransform2D()

if __name__ == "__main__":
    main()
