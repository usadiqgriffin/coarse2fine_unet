import sys
import os
from data import Data
from model import Model
from experiment import Experiment


if __name__ == '__main__':

    project_path = sys.argv[1]
    output_path = sys.argv[2]

    # 0: Loading train and val sets
    # 1: Loading val set only
    # 2: Loading internal test set only
    # 3: Loading FDA_400 test set only
    # 6: Loading the extra set (i.e., Occlusion-No-Seg Test Set, N=1462)
    load_mode = 6

    # Model parameters
    seg_thresh = 0.5
    pred_thresh_mm = 0.15625
    dist_thresh_dilate = 22

    # Initializing model
    model = Model(output_path, clean_output=False, create_summary=False, gpu=False)
    model.define_model(seg_thresh, pred_thresh_mm, dist_thresh_dilate)
    model.initialize_weights(global_step=40401)

    # Initializing data
    data = Data(load_mode=load_mode, model=model)

    # Initializing experiment
    experiment = Experiment(data, model, output_path)
    experiment.eval(load_mode=load_mode)
    
    # A single voxel is 0.5*0.5*0.625=0.15625, and pred_thresh_mm is >= so any prediction is considered a clot
    # dist_thresh_dilate==22 means that there are 21 empty voxels between the clot and vessel; 
    # if they're on the z axis, that corresponds to a distance of 21*0.625=13.125mm;
    # if they're on the x/y axis, that corresponds to a distance of 10.5 mm
