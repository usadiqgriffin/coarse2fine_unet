import numpy as np
import tensorflow as tf
import scipy.misc

import utils.stn3d as stn3d

B = 1
D = 16
H = 512
W = 256
C = 1

in_dims = np.array([D, H, W])
in_res = np.array([1, 1, 1])
out_dims = np.array([D, H, W])
out_res = np.array([1, 1, 1])

img1 = np.zeros(shape=[B, D, H, W, C], dtype=np.float32)
img1[0, 5:15, 100:200, 140:160, 0] = 1
img1[0, 5, 100, 140, 0] = 2

print()
print('img1.shape =', img1.shape)
print('np.sum(img1)', np.sum(img1))
print()

for i in range(in_dims[0]):
    scipy.misc.imsave('test_output/before/img' + str(int(i)) + '.bmp', img1[0, i, :, :, 0])

theta_arr = stn3d.get_transform(
    in_dims=in_dims,
    in_res=in_res,
    out_dims=out_dims,
    out_res=out_res,
    scale=[1, 1, 1],
    flip=[False, False, False],
    rotate=[0, 0, 0],
    translate=[5, 0, 0])

print(theta_arr)

theta_arr = np.expand_dims(theta_arr, axis=0)  # [1, 3, 4]

x = tf.placeholder(tf.float32, [B, D, H, W, C])
theta = tf.placeholder(tf.float32, [B, 3, 4])

with tf.variable_scope('spatial_transformer'):

    x_trans = stn3d.spatial_transformer_network(x, theta, out_dims=out_dims)

# run session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
img2 = sess.run(x_trans, feed_dict={x: img1, theta: theta_arr})

print()
print('img2.shape =', img2.shape)
print('np.sum(img2)', np.sum(img2))
print()

img2[:, :, 0, 0, 0] = 1
for i in range(out_dims[0]):
    scipy.misc.imsave('test_output/after/img' + str(int(i)) + '.bmp', img2[0, i, :, :, 0])


'''
print(np.sum(np.abs(img2[0, :, :, 0] - img1[0, :, :, 0])))
scipy.misc.imsave('test_output/img3.jpg', np.abs(img2[0, :, :, 0] - img1[0, :, :, 0]))


########################## Regression points ##########################
# Make sure points = [[x1, y1], [x2, y2], ...]
points = [[200, 100], [0, 0]]
B_reg = len(points)

# Inversing theta_arr
theta_arr_reg = np.squeeze(theta_arr, axis=0)  # [2, 3]
last_row = np.array([[0, 0, 1]])  # [1, 3]
theta_arr_reg = np.concatenate([theta_arr_reg, last_row])  # [3, 3]
theta_arr_reg = np.linalg.inv(theta_arr_reg)  # [3, 3]
theta_arr_reg = theta_arr_reg[0:2, :]  # [2, 3]

points = np.array(points)  # [B_reg, 2]
axes_size = np.array([H, W])  # [2]

print('points before transformation =', points)

points = stn3d.normalize_points(points, axes_size)  # [B_reg, 2]
ones = np.ones(shape=[B_reg, 1])
points = np.concatenate([points, ones], axis=1)  # [B_reg, 3]
points = np.swapaxes(points, 0, 1)  # [3, B_reg]

points = np.matmul(theta_arr_reg, points)  # [2, B_reg]

points = np.swapaxes(points, 0, 1)  # [B_reg, 2]
points = stn3d.normalize_points(points, axes_size, reverse=True)  # [B_reg, 2]

print('points after transformation =', points)
########################################################################
'''
