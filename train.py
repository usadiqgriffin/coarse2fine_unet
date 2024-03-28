import sys
import time
import os
from data import Data
from model import Model, coarse_to_fine
from experiment import Experiment
import argparse
import logging


if __name__ == '__main__':

    start = time.time()

    '''project_path = sys.argv[1]
    output_path = sys.argv[2]'''

    '''
    # Debug training inputs
    model = Model(output_path, clean_output=False, create_summary=False, gpu=False)
    model.define_model()
    data = Data(project_path, output_path, load_mode=4, model=model)  # Loading a single training example
    experiment = Experiment(data, model, output_path)
    experiment.debug_input(augment=False)
    experiment.debug_input(augment=True)
    exit()
    '''

    csv_dir = '/mnt/HDD_8/experiment_data/usman/mcta_coarse_to_fine/nvi_occlusion_detection/sheets/'
    train_csv_filename = 'dev_set_occlusion_segmentation_from_roi_10.csv'


    # Load user-defined params/settings/hyperparams
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', default='occlusion_from_roi')
    parser.add_argument('--input_types', nargs='+' ,default=['mcta'])

    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--max_epochs', default=1000, type=int)
    parser.add_argument('--train_csv', default="train_set10.csv")
    parser.add_argument('--val_csv', default="val_set10.csv")
    parser.add_argument('--test_csv', default="test_set10.csv")

    parser.add_argument('--project_path', default="project_dir")
    parser.add_argument('--output_path', default="output_dir")

    opts = parser.parse_args()

    project_path = opts.project_path
    output_path = opts.output_path

    data_opts = type('', (), {})()
    data_opts.data_path = csv_dir
    data_opts.train_csv = opts.train_csv
    data_opts.val_csv = opts.val_csv
    data_opts.test_csv = opts.test_csv

    # 0: Loading train and val sets
    # 1: Loading val set only
    # 2: Loading test set only
    # 3: Loading all sets
    load_mode = 0
    
    # Model parameters
    seg_thresh = 0.5
    pred_thresh_mm = 0.15625
    dist_thresh_dilate = 22


    exp_options = type('', (), {})()
    exp_options.logging = "DEBUG"
    if exp_options.logging == "DEBUG":
        logging.basicConfig(level=logging.DEBUG)
    elif exp_options.logging == "INFO":
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)

    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

    # Continue training either the unet or the classifier
    restore = False
    if restore:
        model = coarse_to_fine(output_path, clean_output=False, create_summary=True, gpu=True)
        model.define_model(seg_thresh, pred_thresh_mm, dist_thresh_dilate)
        model.initialize_weights(global_step=-1)
    else:
        model = coarse_to_fine(output_path, clean_output=True, create_summary=True, gpu=True)
        model.define_model(seg_thresh, pred_thresh_mm, dist_thresh_dilate)
        model.initialize_weights(global_step=0)

    # Initializing data
    data = Data(load_mode, model, data_opts=data_opts)
    
    # Initializing experiment
    experiment = Experiment(data, model, output_path)
    experiment.train()

    end = time.time()
    print('Execution time =', (((end - start) / 60) / 60), 'hours')
