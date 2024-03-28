#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:50:56 2020

@author: azhar
"""

import SimpleITK as sitk
import numpy as np
import nibabel as nib
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--i', dest='inFile', default='', help='input nifti volume', required=True)
parser.add_argument('--r', dest='rFile', default='', help='reference nifti volume', required=True)
parser.add_argument('--t', dest='transFile', default='', help='input transform file', required=True)
parser.add_argument('--o', dest='outFile', default='', help='output nifti volume', required=True)

args = parser.parse_args()

if __name__ == '__main__':

    print("Loading %s ..."% args.inFile)
    nb = nib.load(args.inFile)
    vox = nb.header.get_zooms()
    vol = nb.get_fdata()
    
    #clamp pixels
#    fvol[fvol <= 0] = 0
#    fvol[fvol > 250] = 250
    
    vol = np.moveaxis(vol, -1, 0)
    image = sitk.GetImageFromArray(vol)
    image.SetSpacing((float(vox[0]), float(vox[1]), float(vox[2])))
    
    print("Loading %s ..."% args.rFile)
    nb = nib.load(args.rFile)
    r_vox = nb.header.get_zooms()
    rvol = nb.get_fdata()
    
    #clamp pixels
#    fvol[fvol <= 0] = 0
#    fvol[fvol > 250] = 250
    
    rvol = np.moveaxis(rvol, -1, 0)
    ref_image = sitk.GetImageFromArray(rvol)
    ref_image.SetSpacing((float(r_vox[0]), float(r_vox[1]), float(r_vox[2])))
    
    trans = sitk.ReadTransform(args.transFile)
    
    result_image = sitk.Resample(image, ref_image, trans, sitk.sitkLinear, image.GetPixel(0,0,0), image.GetPixelID())
    
    print("Writing file: ", args.outFile)
    sitk.WriteImage(result_image, args.outFile)