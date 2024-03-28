from Report_helper_parent import Metrics, Datasets, Graphs, Packet, to_pdf
from Report_helper_tracker_noise_gt import Study_information_tracker_noise_gt
from Report_helper_tracker_noise_ete import Study_information_tracker_noise_ete
from Report_helper_tracker_seedpoints_gt import Study_information_tracker_seedpoints_gt
from Report_helper_tracker_seedpoints_ete import Study_information_tracker_seedpoints_ete
from Report_helper_seg import Study_information_seg
import numpy as np
import tensorflow as tf
import os
import glob
import pandas as pd
from combined_metrics import (seed_point_metric,
                              seed_point_metric_each_branch,
                              compute_dice,
                              compute_hausdorff_distance,
                              centerline_avg_hdd,
                              centerline_avg_hdd_noise,
                              unannotated_metrics,
                              unannotated_metrics_noise)
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.pyplot as plt
import timeit
import matplotlib
from tqdm import tqdm


def generate_metrics(packet_array, df_save_path='./saved_metrics'):
    #metric_names = []
    graph_names = []
    dataset_names = []
    dataframe = pd.DataFrame()

    count = 0
    for packet in packet_array:
        graphs = packet.graphs.graph_dict
        study_info = packet.study_information()
        dataset_dict = packet.datasets.datasets
        metric_list = packet.metrics.metrics
        for graph_name in graphs:
            graph_names += [graph_name]
            print('Currently computing graph:', graph_name)
            graph_paths = graphs[graph_name]['graph_paths']
            compute_dicts = graphs[graph_name]['compute_dicts']
            prefixs = graphs[graph_name]['prefixs']
            study_info.prep_net(graph_paths, compute_dicts, prefixs)

            for dataset_name in dataset_dict:
                dataset_names += [dataset_name]
                dataset_path = dataset_dict[dataset_name]
                studies = glob.glob(os.path.join(dataset_path,'*/'))

                for study in studies:
                    study_name = study.split('/')[-2]
                    print('study_name:', study_name)
                    if not study_info.add_to_feed_dict(study):
                        continue
                    np.random.seed(42)
                    study_info.run_network()
                    to_dataframe = {}
                    for metric in metric_list:
                        start = timeit.default_timer()
                        computed_metrics = metric(study_info)
                        to_dataframe.update(computed_metrics)
                    met_len = -1
                    for lab in to_dataframe:
                        if not isinstance(to_dataframe[lab],list):
                            to_dataframe[lab] = [to_dataframe[lab]]
                        met_len = max(met_len,len(to_dataframe[lab]))
                    to_dataframe['dataset'] = [dataset_name]*met_len if 'dataset' not in to_dataframe else to_dataframe['dataset'] + [dataset_name]*met_len
                    to_dataframe['study'] = [study_name]*met_len if 'study' not in to_dataframe else to_dataframe['study'] + [study_name]*met_len
                    to_dataframe['graph'] = [graph_name]*met_len if 'graph' not in to_dataframe else to_dataframe['graph'] + [graph_name]*met_len
                    to_dataframe = pd.DataFrame(to_dataframe, index=[v+count for v in range(met_len)])
                    count += met_len
                    dataframe = pd.concat([dataframe, to_dataframe])
        study_info.reset_sess_and_graph()
    dataframe.to_csv(df_save_path+'.csv')
    dataframe.to_pickle(df_save_path+'.pkl')
    return dataframe

