import numpy as np
import pandas as pd
import tensorflow as tf
import os
import glob
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

class Study_information_parent:
    def __init__(self, compute_dict):
        self.input_x = None
        self.gt_radii = None

        self.sess = None
        self.compute_dict = {}
        for c in compute_dict:
            self.compute_dict[c] = None

        self.feed_dict = None

        self.seg_gt = None

    def read_data(self, study_file_path):
        img_in_path = glob.glob(study_file_path+'/*_img.npy')[0]

        img_in = np.load(img_in)
        gt_seg = np.load(img_in_path.replace('_img.npy','_seg.npy'))
        hdr = np.load(img_in_path.replace('_img.npy','_hdr.npy')).tolist()

        self.input_x = img_in
        self.gt_seg = gt_seg
        self.input_shape = np.array([128, 128, 128])
        self.input_resln = hdr['PixelSpacing']
        self.output_resln = np.array([1.5, 1.5, 1.5])

    def add_to_feed_dict(self, input_filepath):
        if not self.read_data(input_filepath):
            return False

        feed_dict ={
                "Input/node_x:0": self.img_in,
                "Input/train:0": False,
                "Input/preprocess:0": True,
                "Input/in_shape:0": self.input_shape,
                "Input/trans:0": [1,0,0,0,0,1,0,0],
                "Input/post_trans:0": [1,0,0,0,0,1,0,0]
                }
        
        self.feed_dict = feed_dict
        return True

    def reset_sess_and_graph(self):
        if self.sess is not None:
            self.sess.close()
            tf.reset_default_graph()


    def prep_net(self, graph_path, config=None, prefix=''):

        if config == None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

        self.reset_sess_and_graph()
        self.sess = tf.Session(config=config)

        with tf.gfile.GFile(graph_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, name=prefix)


    def run_network(self, gpu_num=0):
        comp = []
        for l in self.compute_dict:
            comp += [l]

        with tf.device('/device:GPU:' + str(gpu_num)):
            output = self.sess.run(comp, feed_dict=self.feed_dict)
        
        for c,o in zip(self.compute_dict,output):
            self.compute_dict[c] = o

        self.net_output = output



class Metrics():
    def __init__(self):
        self.metrics = []
    def append(self, metric):
        self.metrics += [metric]

class Graphs:
    def __init__(self,task_nickname):
        self.task_nickname = task_nickname
        self.graph_dict = {}
    def append(self,nickname,graph_path_array,comp_dicts_array,prefix_array = ['']):
        gpa_len = len(graph_path_array)
        if gpa_len != len(comp_dicts_array) or gpa_len != len(prefix_array):
            print('Error! Input arrays not the same lenght!')
            exit()

        name = nickname
        if self.task_nickname != '':
            name += '_'+self.task_nickname

        if name in self.graph_dict:
            print('Error! \"'+nickname+'\" already used!')
            exit()
        self.graph_dict[name]={
            'graph_paths':graph_path_array,
            'compute_dicts':comp_dicts_array,
            'prefixs':prefix_array
            }

class Datasets:
    def __init__(self):
        self.datasets = {}
    def append(self, nickname, dataset):
        if nickname in self.datasets:
            print('Error! \"'+nickname+'\" already used!')
            exit()
        self.datasets[nickname] = dataset

class Packet:
    def __init__(self, graphs, datasets, metrics, study_information):
        self.graphs = graphs
        self.datasets = datasets
        self.metrics = metrics
        self.study_information = study_information
    
    
def to_pdf(pkl_load_path = './saved_metrics', pdf_save_path = './saved_metrics', plots_per_page=4):
    dataframe = pd.read_pickle(pkl_load_path+'.pkl')
    columns = list(dataframe)
    not_metrics = ['dataset','graph','study']
    metric_names = [c for c in columns if c not in not_metrics]
    print()
    for m in metric_names:
        print('A metric:',m)
    print()
    fig_size = (8, 11)
    metric_names = sorted(metric_names, key=lambda m_name:m_name, reverse=True)
    #print(metric_names)
    #print('len of metric names {}'.format(len(metric_names)))
    with PdfPages(pdf_save_path+'.pdf') as pdf:
        for m_idx in range(0, len(metric_names), plots_per_page):
            upper = m_idx + plots_per_page if m_idx + plots_per_page <= len(metric_names) else len(metric_names)
            m_names_page = metric_names[m_idx:upper]

            fig, ax = plt.subplots(plots_per_page, 1, figsize=fig_size)
            fig.suptitle('Generic Metrics '+str(m_idx))
            save_to_pdf = True
            for i, m_name in enumerate(m_names_page):
                report = dataframe[[m_name, 'dataset', 'graph']]
                report = report.loc[report[m_name] != 'None']
                try:
                    # this except is for empty dataframes
                    sns.boxplot(x='dataset', y=m_name, hue='graph', data=report, ax=ax[i])
                except:
                    save_to_pdf = False
            if save_to_pdf:
                # plt.show()
                # input('Enter')
                pdf.savefig(fig)
            plt.close()
    print(dataframe)

