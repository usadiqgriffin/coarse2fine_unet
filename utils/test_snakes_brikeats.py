import os
import pdb
import sys
import unittest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import snakes_brikeats


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


class TestSnakesBrikeats(unittest.TestCase):

    def test_computeExternalEnergy(self):
        pass

    def test_computeSpacingEnergy(self):
        pass

    def test_computeCurvatureEnergy(self):
        pass

    def test_computeEndPtsEnergy(self):
        pass

    def test_computeTotalEnergy(self):
        pass

    def test_fitSnake(self):
        pass