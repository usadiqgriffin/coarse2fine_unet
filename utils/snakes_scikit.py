import pdb

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline

'''
Ref:
    [1] https://github.com/scikit-image/scikit-image/blob/main/skimage/segmentation/active_contour_model.py
    [2] https://www.mathworks.com/matlabcentral/fileexchange/28149-snake-active-contour
    [3] Michael Kass, Andrew Witkin and Demetri TerzoPoulos "Snakes Active Contour Models", 1987
    [4] Jim Invins and John Porril, "Everything you always wanted to know about snakes (but were afraid to ask)
    [5] Chenyang Xu and Jerry L. Prince, "Gradient Vector Flow: A New external force for Snakes
'''


def active_contour_gvf_3d(
    u, v, w,
    snake, 
    alpha=0.01, beta=0.1, gamma=0.01,
    max_px_move=1.0,
    max_num_iter=2500, 
    boundary_condition='fixed',
    float_dtype=np.float32):
    """Active contour model.
    Active contours by fitting snakes to gradient vector flow.
    Snakes can be periodic (for segmentation) or
    have fixed and/or free ends.
    The output snake has the same length as the input boundary.
    ----------
    u : (N, M, L) ndarray
        3D gradient vector flow corresponding to the y-dimension (dim-0).
    v : (N, M, L) ndarray
        3D gradient vector flow corresponding to the x-dimension (dim-1).
    w : (N, M, L) ndarray
        3D gradient vector flow corresponding to the z-dimension (dim-2).
    snake : (N, 3) ndarray
        Initial snake coordinates. For periodic boundary conditions, endpoints
        must not be duplicated.
    alpha : float, optional
        Snake length shape parameter. Higher values makes snake contract
        faster.
    beta : float, optional
        Snake smoothness shape parameter. Higher values makes snake smoother.
    gamma : float, optional
        Explicit time stepping parameter.
    max_px_move : float, optional
        Maximum pixel distance to move per iteration.
    max_num_iter : int, optional
        Maximum iterations to optimize snake shape.
    boundary_condition : string, optional
        Boundary conditions for the contour. Can be one of 'periodic',
        'free', 'fixed', 'free-fixed', or 'fixed-free'. 'periodic' attaches
        the two ends of the snake, 'fixed' holds the end-points in place,
        and 'free' allows free movement of the ends. 'fixed' and 'free' can
        be combined by parsing 'fixed-free', 'free-fixed'. Parsing
        'fixed-fixed' or 'free-free' yields same behaviour as 'fixed' and
        'free', respectively.
    Returns
    -------
    snake : (N, 3) ndarray
        Optimised snake, same shape as input parameter.
    References
    ----------
    .. [1]  Kass, M.; Witkin, A.; Terzopoulos, D. "Snakes: Active contour
            models". International Journal of Computer Vision 1 (4): 321
            (1988). :DOI:`10.1007/BF00133570`
    """

    max_num_iter = int(max_num_iter)
    if max_num_iter <= 0:
        raise ValueError("max_num_iter should be >0.")
    valid_bcs = ['periodic', 'free', 'fixed', 'free-fixed',
                 'fixed-free', 'fixed-fixed', 'free-free']
    if boundary_condition not in valid_bcs:
        raise ValueError("Invalid boundary condition.\n" +
                         "Should be one of: "+", ".join(valid_bcs)+'.')

    x = snake[:,0].astype(float_dtype)
    y = snake[:,1].astype(float_dtype)
    z = snake[:,2].astype(float_dtype)
    n = len(x)

    # Build snake shape matrix for Euler equation in double precision
    eye_n = np.eye(n, dtype=float)
    a = (np.roll(eye_n, -1, axis=0)
         + np.roll(eye_n, -1, axis=1)
         - 2 * eye_n)  # second order derivative, central difference
    b = (np.roll(eye_n, -2, axis=0)
         + np.roll(eye_n, -2, axis=1)
         - 4 * np.roll(eye_n, -1, axis=0)
         - 4 * np.roll(eye_n, -1, axis=1)
         + 6 * eye_n)  # fourth order derivative, central difference
    A = -alpha * a + beta * b

    # Impose boundary conditions different from periodic:
    sfixed = False
    if boundary_condition.startswith('fixed'):
        A[0, :] = 0
        A[1, :] = 0
        A[1, :3] = [1, -2, 1]
        sfixed = True
    efixed = False
    if boundary_condition.endswith('fixed'):
        A[-1, :] = 0
        A[-2, :] = 0
        A[-2, -3:] = [1, -2, 1]
        efixed = True
    sfree = False
    if boundary_condition.startswith('free'):
        A[0, :] = 0
        A[0, :3] = [1, -2, 1]
        A[1, :] = 0
        A[1, :4] = [-1, 3, -3, 1]
        sfree = True
    efree = False
    if boundary_condition.endswith('free'):
        A[-1, :] = 0
        A[-1, -3:] = [1, -2, 1]
        A[-2, :] = 0
        A[-2, -4:] = [-1, 3, -3, 1]
        efree = True

    # Only one inversion is needed for implicit spline energy minimization:
    inv = np.linalg.inv(A + gamma * eye_n)
    # can use float_dtype once we have computed the inverse in double precision
    inv = inv.astype(float_dtype, copy=False)

    # Explicit time stepping for image energy minimization:
    for i in range(max_num_iter):

        coords = np.stack([y,x,z], axis=0)
        fx = ndimage.interpolation.map_coordinates(v, coords, order=1)
        fy = ndimage.interpolation.map_coordinates(u, coords, order=1)
        fz = ndimage.interpolation.map_coordinates(w, coords, order=1)

        if sfixed:
            fx[0] = 0
            fy[0] = 0
            fz[0] = 0
        if efixed:
            fx[-1] = 0
            fy[-1] = 0
            fz[-1] = 0
        if sfree:
            fx[0] *= 2
            fy[0] *= 2
            fz[0] *= 2
        if efree:
            fx[-1] *= 2
            fy[-1] *= 2
            fz[0] *= 2
        xn = inv @ (gamma*x + fx)
        yn = inv @ (gamma*y + fy)
        zn = inv @ (gamma*z + fz)

        # Movements are capped to max_px_move per iteration:
        dx = max_px_move * np.tanh(xn - x)
        dy = max_px_move * np.tanh(yn - y)
        dz = max_px_move * np.tanh(zn - z)
        if sfixed:
            dx[0] = 0
            dy[0] = 0
            dz[0] = 0
        if efixed:
            dx[-1] = 0
            dy[-1] = 0
            dz[-1] = 0
        x += dx
        y += dy
        z += dz

    return np.stack([x,y,z], axis=1)