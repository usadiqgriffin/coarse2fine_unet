import os
import pdb
import sys
import unittest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import snakes_scikit
from gvf import GVF3D


def imshowProjections(vol, fun=np.max, block=False):
    fig, ax = plt.subplots(1,3)
    for dim in range(3):
        im = np.squeeze(fun(vol, axis=dim))
        ax[dim].imshow(im)
        ax[dim].set_title(f"dim-{dim} projection")
        if dim==0:
            ax[dim].set_xlabel("z, dim-2")
            ax[dim].set_ylabel("x, dim-1")
        if dim==1:
            ax[dim].set_xlabel("z, dim-2")
            ax[dim].set_ylabel("y, dim-0")
        if dim==2:
            ax[dim].set_xlabel("x, dim-1")
            ax[dim].set_ylabel("y, dim-0")
    plt.show(block=block)

def imshowProjectionsPoints(vol, pts, fun=np.max, block=False):
    '''pts are [N,3], [x,y,z]'''
    fig, ax = plt.subplots(1,3)
    for dim in range(3):
        im = np.squeeze(fun(vol, axis=dim))
        ax[dim].imshow(im)
        ax[dim].set_title(f"dim-{dim} projection")
        if dim==0:
            ax[dim].set_xlabel("z, dim-2")
            ax[dim].set_ylabel("x, dim-1")
            ax[dim].scatter(pts[:,2], pts[:,0], c="r", s=0.2)
        if dim==1:
            ax[dim].set_xlabel("z, dim-2")
            ax[dim].set_ylabel("y, dim-0")
            ax[dim].scatter(pts[:,2], pts[:,1], c="r", s=0.2)
        if dim==2:
            ax[dim].set_xlabel("x, dim-1")
            ax[dim].set_ylabel("y, dim-0")
            ax[dim].scatter(pts[:,0], pts[:,1], c="r", s=0.2)
    plt.show(block=block)


class TestSnakesScikit(unittest.TestCase):

    def test_active_contour_gvf_3d(self):
                
        # create dummy image of single line
        im = np.zeros((100,105,110))
        im[30:50,50,70] = 1
        imshowProjections(im, fun=np.max)

        # skel
        skel = im.copy()

        # GVF
        '''
        V := X := dim-1 (columns)
        U := Y := dim-0 (rows)
        W := Z := dim-2 (pages) 
        '''
        U,V,W = GVF3D(
            skel.astype(np.float32), 
            mu=0.01, 
            iterations=500,
            verbose=True
            )
        imshowProjections(V, np.max) # should change along the x-axis
        imshowProjections(U, np.max) # should change along the y-axis
        imshowProjections(W, np.max) # should change along the z-axis

        # define straight line between beginning and end points of vessel
        N = 10
        y = np.linspace(30, 50, N)
        x = 50 * np.ones_like(y)
        z = 73 * np.ones_like(y)
        snake_init = np.stack([x,y,z], axis=1)
        print(snake_init)
        assert list(snake_init.shape) == [N,3]
        imshowProjectionsPoints(im, pts=snake_init, fun=np.max)

        # fit snake
        snake_final = snakes_scikit.active_contour_gvf_3d(
            u=U,
            v=V,
            w=W,
            snake=snake_init,
            boundary_condition="fixed",
            # alpha=0.1,
            # beta=0.1,
            # gamma=0.1,
            # max_px_move=1,
            max_num_iter=1000,
        )
        print(snake_init)
        print(snake_final)
        imshowProjectionsPoints(im, pts=snake_final, fun=np.max)