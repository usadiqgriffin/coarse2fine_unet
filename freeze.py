import sys
import os
from data import Data
from model import Model
from experiment import Experiment
from utils.model import generate_scrambled_proto
import numpy as np
import tensorflow as tf

if __name__ == '__main__':

    output_path = sys.argv[1]
    experiment_id = sys.argv[2]

    # Model parameters
    seg_thresh = 0.5
    pred_thresh_mm = 0.15625
    dist_thresh_dilate = 22
    global_step = 40401

    # Initializing model
    model = Model(output_path, clean_output=False, create_summary=False, gpu=False)
    model.define_model(seg_thresh, pred_thresh_mm, dist_thresh_dilate)
    model.initialize_weights(global_step=global_step)

    model.freeze(
        output_path,
        experiment_id,
        ['classifier/pred_clot', 'classifier/pred_bin', 'classifier/pred_prob'],
        global_step)

    generate_scrambled_proto(
        output_path + '/frozen/' + experiment_id + '_graph_output.pb', 
        output_path + '/frozen/ml_' + experiment_id)
