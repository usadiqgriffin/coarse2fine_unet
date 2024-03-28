#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 12:27:10 2020

@author: azhar
"""

import SimpleITK as sitk
import numpy as np
import nibabel as nib
import argparse
import copy

parser = argparse.ArgumentParser(description='')
parser.add_argument('--fixed', dest='fixFile', default='', help='input nifti fixed volume', required=True)
parser.add_argument('--moving', dest='moveFile', default='', help='input nifti moving volume', required=True)
parser.add_argument('--out_file', dest='outFile', default='', help='output registered nifti volume', required=True)
parser.add_argument('--out_trans', dest='outTrans', default='', help='output transform file', required=True)

args = parser.parse_args()

#--fixed /mnt/HDD_5/processed_data_nvi/prove_it/cta/Prove-It-01-001/cta1.nii.gz --moving /mnt/HDD_5/processed_data_nvi/prove_it/ctp/Prove-It-01-001/ctp0.nii.gz --out_file result_image.nii.gz  --out_trans  result_trans.tfm


def register_vol(fixedFile, movingFile, outFile, outTransform):
    
    print("Loading %s ..."% fixedFile)
    nb = nib.load(fixedFile)
    f_vox = nb.header.get_zooms()
    fvol = nb.get_fdata()
    
    #clamp pixels
    fvol[fvol <= 0] = 0
    fvol[fvol > 250] = 250
    
    fvol = np.moveaxis(fvol, -1, 0)
    fixed_image = sitk.GetImageFromArray(fvol)
    fixed_image.SetSpacing((float(f_vox[0]), float(f_vox[1]), float(f_vox[2])))
    
    print("Loading %s ..."% movingFile)
    nb = nib.load(movingFile)
    m_vox = nb.header.get_zooms()
    mvol = nb.get_fdata()
    mvol_orig = copy.deepcopy(mvol)
    
    #clamp pixels
    mvol[mvol <= 0] = 0
    mvol[mvol > 250] = 250
    
    #--fixe if moving image has more slices than the fixed image
    if m_vox[2]*mvol.shape[2] > f_vox[2]*fvol.shape[2] :
        print("Moving volume is larger than fixed volume")
        print("Fixing the moving volume...")
        target_size = f_vox[2]*fvol.shape[2]
        nSlices = int(target_size/m_vox[2])
        tmp = mvol[:,:,-nSlices:]
        mvol = tmp
        tmp = mvol_orig[:,:,-nSlices:]
        mvol_orig = tmp

    mvol = np.moveaxis(mvol, -1, 0)
    moving_image = sitk.GetImageFromArray(mvol)
    moving_image.SetSpacing((float(m_vox[0]), float(m_vox[1]), float(m_vox[2])))
    
    mvol_orig = np.moveaxis(mvol_orig, -1, 0)
    moving_image0 = sitk.GetImageFromArray(mvol_orig)
    moving_image0.SetSpacing((float(m_vox[0]), float(m_vox[1]), float(m_vox[2])))

    print("Fixed Image Size: ",fixed_image.GetSize())
    print("Moving Image Size: ", moving_image.GetSize())
    
    interp = sitk.sitkLinear
    #interp = sitk.sitkBSpline
    
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                      moving_image, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

    moving_resampled = sitk.Resample(moving_image, fixed_image, initial_transform, interp, 0.0, moving_image.GetPixelID())
    print("Resampled Image Size: ", moving_resampled.GetSize())
    print("Resampled Image Spacing: ", moving_resampled.GetSpacing())

    
    #---- Regsitration Setup -----
    registration_method = sitk.ImageRegistrationMethod()
    
    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    
    registration_method.SetInterpolator(interp)
    
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200, \
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Setup for the multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
#    # Connect all of the observers so that we can perform plotting during registration.
#    registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
#    registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
#    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
#    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))
#    
    #----- Do the Registration ------
    print("Start Registration ...")
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), \
                                                   sitk.Cast(moving_image, sitk.sitkFloat32))

    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    
    #------ display regsiteration results -----
    result_image = sitk.Resample(moving_image0, fixed_image, final_transform, sitk.sitkLinear, moving_image0.GetPixel(0,0,0), moving_image0.GetPixelID())

    print("Writing file: ", outFile)
    sitk.WriteImage(result_image, outFile)
    print("Writing file: ", outTransform)
    sitk.WriteTransform(final_transform, outTransform)

if __name__ == '__main__':

    register_vol(args.fixFile, args.moveFile, args.outFile, args.outTrans)