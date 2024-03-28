import numpy as np
import tensorflow as tf
import scipy.misc

import utils.stn as stn
import matplotlib.pyplot as plt

B = 2
H = 600
W = 600
C = 1

img1 = np.zeros(shape=[B, H, W, C], dtype=np.float32)
img1[0, 100:500, 200:400, 0] = 1
img1[0, 100, 200, 0] = 2
img1[1, 10:20, 100:150, 0] = 1
img1[1, 10, 100, 0] = 2

for i in range(B):
    plt.imshow(img1[i, :, :, 0], cmap="gray")
    plt.show()

theta_arr = np.zeros(shape=(B, 2, 3))
for i in range(B):
    theta_arr[i] = stn.get_transform(
        scale_x=2,
        flip_x=False,
        rotate_xy=45)

with tf.device('/device:GPU:2'):
    x = tf.placeholder(tf.float32, [B, H, W, C])
    theta = tf.placeholder(tf.float32, [B, 2, 3])

    with tf.variable_scope('spatial_transformer'):

        x_trans = stn.spatial_transformer_network(x, theta, 'nearest')

    # run session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    img2 = sess.run(x_trans, feed_dict={x: img1, theta: theta_arr})

for i in range(B):
    plt.imshow(img2[i, :, :, 0], cmap="gray")
    plt.show()

exit()

########################## Regression points ##########################

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

points = stn.normalize_points(points, axes_size)  # [B_reg, 2]
ones = np.ones(shape=[B_reg, 1])
points = np.concatenate([points, ones], axis=1)  # [B_reg, 3]
points = np.swapaxes(points, 0, 1)  # [3, B_reg]

points = np.matmul(theta_arr_reg, points)  # [2, B_reg]

points = np.swapaxes(points, 0, 1)  # [B_reg, 2]
points = stn.normalize_points(points, axes_size, reverse=True)  # [B_reg, 2]

print('points after transformation =', points)
########################################################################
