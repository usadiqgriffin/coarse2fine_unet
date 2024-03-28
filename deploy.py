import sys
from data import Data
from model import Model
from experiment import Experiment

if __name__ == '__main__':

    project_path = sys.argv[1]
    output_path = sys.argv[2]

    # 100: Loading from a seperate spreadsheet into self.test_set
    load_mode = 100

    # Initializing model
    model = Model(output_path, clean_output=False, create_summary=False)
    model.define_model()
    model.initialize_weights(global_step=-1)

    # Initializing data
    data = Data(load_mode=load_mode, model=model)

    # Initializing experiment
    experiment = Experiment(data, model, output_path)
    experiment.deploy(save_nifti=True)