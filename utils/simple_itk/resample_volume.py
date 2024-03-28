#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:22:52 2020

@author: azhar
"""

import SimpleITK as sitk
import numpy as np
import nibabel as nib
import argparse
import copy

parser = argparse.ArgumentParser(description='')
parser.add_argument('--in_file', dest='inFile', default='', help='input nifti volume', required=True)
parser.add_argument('--out_file', dest='outFile', default='', help='output nifti volume', required=True)
parser.add_argument('--out_dims', dest='outSize', nargs='+', help='output dimensions', required=True)
parser.add_argument('--out_spacing', dest='outSpacing', nargs='+', help='output voxel sizes', required=False)
parser.add_argument('--brain', dest='brainMM', default=-1, help='brain (z-direction) to be extracted (in mm)', required=False)

args = parser.parse_args()

#--in_file /mnt/HDD_5/processed_data_nvi/ncct_atlas/ncct_atlas_image.nii.gz --out_file atlas_resampled.nii.gz --out_dims 256 256 64 --out_spacing 0.8476 0.8476 2.8281


def resample_vol(inFile, outFile, outSize, outSpacing):
    
    print("Loading %s ..."% inFile)
    nb = nib.load(inFile)
    f_vox = nb.header.get_zooms()
    fvol = nb.get_fdata()
    f_size = fvol.shape
    
    if float(args.brainMM) > 0.:
        nSlices = int(float(args.brainMM)/f_vox[2])
        #print(nSlices)
        tmp = fvol[:,:,-nSlices:]
        #print(tmp.shape)
        fvol = tmp
        
        
    fvol = np.moveaxis(fvol, -1, 0)
    fixed_image = sitk.GetImageFromArray(fvol)
    fixed_image.SetSpacing((float(f_vox[0]), float(f_vox[1]), float(f_vox[2])))
    #fixed_image.SetOrigin((0, 0, float(f_size[2])))
    
    print("Input Volume Size: ", fixed_image.GetSize())
    print("Input Volume Spacing: ", fixed_image.GetSpacing())
    i_size = fixed_image.GetSize()
    
    desired_dims = outSize
    desired_spacing = outSpacing
    
    rx = f_vox[0] * float(f_size[0]/desired_dims[0])
    ry = f_vox[1] * float(f_size[1]/desired_dims[1])
    rz = f_vox[2] * float(f_size[2]/desired_dims[2])
#    
    output_origin = [0, 0, 0]
    output_spacing = [rx, ry, rz]
    output_direction = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
    
    EPSILON = 0.001
    x_offset = 0.0
    y_offset = 0.0
    z_offset = 0.0
    
    if (desired_spacing[0]) > 0 and (abs(desired_spacing[0] - output_spacing[0]) > EPSILON) :
        x_offset = desired_spacing[0] * (desired_dims[0] - int(f_vox[0] * i_size[0]/desired_spacing[0]))
        output_spacing[0] = desired_spacing[0]
    
    if (desired_spacing[1] > 0) and (abs(desired_spacing[1] - output_spacing[1]) > EPSILON) :
        y_offset = desired_spacing[1] *  (desired_dims[1] - int(f_vox[1] * i_size[1]/desired_spacing[1]))
        output_spacing[1] = desired_spacing[1]
    
    if (desired_spacing[2] > 0) and (abs(desired_spacing[2] - output_spacing[2]) > EPSILON) :
        z_offset = desired_spacing[2] * (desired_dims[2] - int(f_vox[2] * i_size[2]/desired_spacing[2]))
        output_spacing[2] = desired_spacing[2]
        
    my_trans = sitk.Euler3DTransform()
    print("offset: ", (-x_offset/2, -y_offset/2, -z_offset/2) )
    my_trans.SetTranslation( (-x_offset/2, -y_offset/2, -z_offset/2) )
    resampled = sitk.Resample(fixed_image, desired_dims, my_trans, sitk.sitkLinear, output_origin, \
                              output_spacing, output_direction, fixed_image.GetPixel(0,0,0), fixed_image.GetPixelID())
    
    print("Resampled Volume Size: ", resampled.GetSize())
    print("Resampled Volume Spacing: ", resampled.GetSpacing())
    print("Writing file: ", outFile)
    sitk.WriteImage(resampled, outFile)

if __name__ == '__main__':

    outSize = [0, 0, 0]
    outSize[0] = int(args.outSize[0])
    outSize[1] = int(args.outSize[1])
    outSize[2] = int(args.outSize[2])
    
    outSpacing = [-1, -1, -1]
    
    if args.outSpacing:
        outSpacing[0] = float(args.outSpacing[0])
        outSpacing[1] = float(args.outSpacing[1])
        outSpacing[2] = float(args.outSpacing[2])

    resample_vol(args.inFile, args.outFile, outSize, outSpacing)