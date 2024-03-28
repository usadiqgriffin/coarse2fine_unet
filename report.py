from re import X
import sys
import logging
import numpy as np
import os
from hypertester.hypertest import Hypertest
from hypertester.statistic import Statistic
from hypertester.reporter import Report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


def run_hypertest(mode, test, y, t, eval_dir, output_df):

    test_class = Hypertest(
        y = y,
        t = t,
        name = test + '_' + mode,
        task = 'classification',
        results_folder = eval_dir,
        experiment_list = [test]).run()
    test_class.report()
    test_class.save()

    if test == 'roc_analysis':

        auc = test_class.get_result('auc')
        sample_size = test_class.get_result('sample_size')
        output_df = output_df.append({
            'metric': 'auc_' + mode,
            'point_estimate': round(auc, 4),
            'lower_bound': None,
            'upper_bound': None,
            'sample_size': sample_size}, ignore_index=True)

    elif test == 'classification_report':

        sensitivity = test_class.get_result('sensitivity')
        sensitivity_pe = sensitivity.point_estimate
        sensitivity_ci = sensitivity.confidence_interval
        sensitivity_sample_size = sensitivity.sample_size

        specificity = test_class.get_result('specificity')
        specificity_pe = specificity.point_estimate
        specificity_ci = specificity.confidence_interval
        specificity_sample_size = specificity.sample_size

        output_df = output_df.append({
            'metric': 'sensitivity_' + mode,
            'point_estimate': round(sensitivity_pe, 3),
            'lower_bound': round(sensitivity_ci[0], 3),
            'upper_bound': round(sensitivity_ci[1], 3),
            'sample_size': sensitivity_sample_size}, ignore_index=True)

        output_df = output_df.append({
            'metric': 'specificity_' + mode,
            'point_estimate': round(specificity_pe, 3),
            'lower_bound': round(specificity_ci[0], 3),
            'upper_bound': round(specificity_ci[1], 3),
            'sample_size': specificity_sample_size}, ignore_index=True)

    return output_df


if __name__ == '__main__':

    output_df = pd.DataFrame(columns=['metric', 'point_estimate', 'lower_bound', 'upper_bound', 'sample_size'])

    # Load data sheet
    # 0: lvo_r083
    # 1: lvo_r128 - development environment
    # 2: lvo_r128 - stanalone library execution
    # 3: lvo_r128 - StrokeSens execution
    eval_mode = 3

    if eval_mode == 0:
        input_path ='/mnt/HDD_5/experiment_data/rotem/lvo_detection/lvo_r128/output/eval_lvo_r083/lvo_r083_output.csv'
        raw_df = pd.read_csv(input_path)
        raw_df = raw_df[raw_df['usage'] == 'fda3_test']
        raw_df = raw_df[raw_df['usage'] == 'fda3_test']
        raw_df['consensus_site_of_occlusion'] = raw_df['site_of_occlusion_final']
        raw_df['pred_prob'] = 1 - raw_df['entropy']
        raw_df = raw_df.reset_index()

    elif eval_mode == 1:
        input_path = '/mnt/HDD_5/experiment_data/rotem/lvo_detection/lvo_r128/output/eval/eval_step_40401_0.5_0.15625_22_3/eval_output.csv'
        raw_df = pd.read_csv(input_path)

    elif eval_mode in [2, 3]:
        input_path = '/mnt/HDD_5/experiment_data/rotem/lvo_detection/lvo_r128/output/eval/eval_step_40401_0.5_0.15625_22_3/eval_output.csv'
        raw_df = pd.read_csv(input_path)
        print('Getting the standalone library output...')
        pred_bin_lib_arr = []
        pred_prob_lib_arr = []
        #cta_depth_arr = []
        #cta_rows_arr = []
        #cta_cols_arr = []
        for i in range(len(raw_df)):
            row = raw_df.iloc[i, :]
            patient_folder_name = row['patient_folder_name']
            if eval_mode == 2:
                base_path = '/mnt/HDD_9/test_set_files/lvo/temp_lvo_v1.4_execution/'             
                pred_bin_path = base_path + patient_folder_name + '/output/predBin.txt'
                pred_prob_path = base_path + patient_folder_name + '/output/predProb.txt'
            elif eval_mode == 3:
                base_path = '/mnt/HDD_9/test_set_files/lvo/StrokeSENS_1v4v269_temp_lvo_v1.4_execution_outputs_2022-08-12/data/'
                pred_bin_path = base_path + patient_folder_name + '/predBin.txt'
                pred_prob_path = base_path + patient_folder_name + '/predProb.txt'
            else:
                exit()

            with open(pred_bin_path) as f:
                pred_bin_lib = bool(int(f.readline()[0]))
            pred_bin_lib_arr.append(pred_bin_lib)

            with open(pred_prob_path) as f:
                pred_prob_lib = float(f.readline()[:-1])
            pred_prob_lib_arr.append(pred_prob_lib)

            '''
            cta_size_path = base_path + patient_folder_name + '/output/ctaSize.txt'
            with open(cta_size_path) as f:
                cta_depth = int(f.readline()[:-1])
                cta_rows = int(f.readline()[:-1])
                cta_cols = int(f.readline()[:-1])
            cta_depth_arr.append(cta_depth)
            cta_rows_arr.append(cta_rows)
            cta_cols_arr.append(cta_cols)
            '''

        raw_df['pred_bin_lib'] = pred_bin_lib_arr
        raw_df['pred_prob_lib'] = pred_prob_lib_arr
        #raw_df['cta_depth'] = cta_depth_arr
        #raw_df['cta_rows'] = cta_rows_arr
        #raw_df['cta_cols'] = cta_cols_arr
        if eval_mode == 2:
            input_path = '/mnt/HDD_5/experiment_data/rotem/lvo_detection/lvo_r128/output/eval/eval_step_40401_0.5_0.15625_22_3/eval_output_lib.csv'
        elif eval_mode == 3:
            input_path = '/mnt/HDD_5/experiment_data/rotem/lvo_detection/lvo_r128/output/eval/eval_step_40401_0.5_0.15625_22_3/eval_output_ss.csv'

        raw_df.to_csv(input_path)
        raw_df['pred_bin'] = raw_df['pred_bin_lib']
        raw_df['pred_prob'] = raw_df['pred_prob_lib']

    eval_dir = os.path.dirname(input_path) + '/hypertester_results_' + str(eval_mode) + '/'

    #### Full Cohort ####   
    output_df = run_hypertest(
        mode = 'full_cohort', 
        test = 'roc_analysis',
        y = raw_df['pred_prob'],
        t = raw_df['label_bin'],
        eval_dir = eval_dir,
        output_df = output_df)

    output_df = run_hypertest(
        mode = 'full_cohort', 
        test = 'classification_report',
        y = raw_df['pred_bin'],
        t = raw_df['label_bin'],
        eval_dir = eval_dir,
        output_df = output_df)

    output_df = run_hypertest(
        mode = 'full_cohort', 
        test = 'confusion_matrix',
        y = raw_df['pred_bin'],
        t = raw_df['label_bin'],
        eval_dir = eval_dir,
        output_df = output_df)

    #### ICA/M1 and ICH/Other Set ####
    positive_set = raw_df[raw_df['label_bin'] == True]
    negative_set = raw_df[raw_df['label_bin'] == False]
    ica_set = positive_set[positive_set['consensus_site_of_occlusion'] == 'ICA']
    ica_set = pd.concat([ica_set, negative_set], ignore_index=True)
    m1_set = pd.concat([positive_set, ica_set]).drop_duplicates('patient_folder_name', False)
    m1_set = pd.concat([m1_set, negative_set], ignore_index=True)
    ich_set = negative_set[negative_set['consensus_site_of_occlusion'] == 'ICH']
    ich_set = pd.concat([ich_set, positive_set], ignore_index=True)
    other_set = pd.concat([negative_set, ich_set]).drop_duplicates('patient_folder_name', False)
    other_set = pd.concat([other_set, positive_set], ignore_index=True)

    output_df = run_hypertest(
        mode = 'ica_set', 
        test = 'classification_report',
        y = ica_set['pred_bin'],
        t = ica_set['label_bin'],
        eval_dir = eval_dir,
        output_df = output_df)

    output_df = run_hypertest(
        mode = 'm1_set', 
        test = 'classification_report',
        y = m1_set['pred_bin'],
        t = m1_set['label_bin'],
        eval_dir = eval_dir,
        output_df = output_df)

    output_df = run_hypertest(
        mode = 'ich_set', 
        test = 'classification_report',
        y = ich_set['pred_bin'],
        t = ich_set['label_bin'],
        eval_dir = eval_dir,
        output_df = output_df)

    output_df = run_hypertest(
        mode = 'other_set', 
        test = 'classification_report',
        y = other_set['pred_bin'],
        t = other_set['label_bin'],
        eval_dir = eval_dir,
        output_df = output_df)

    #### Age ####
    age_70_plus_set = raw_df[raw_df['age'] >= 70].reset_index()
    age_below_70_set = raw_df[raw_df['age'] < 70].reset_index()

    output_df = run_hypertest(
        mode = 'age_70_plus_set', 
        test = 'classification_report',
        y = age_70_plus_set['pred_bin'],
        t = age_70_plus_set['label_bin'],
        eval_dir = eval_dir,
        output_df = output_df)

    output_df = run_hypertest(
        mode = 'age_below_70_set', 
        test = 'classification_report',
        y = age_below_70_set['pred_bin'],
        t = age_below_70_set['label_bin'],
        eval_dir = eval_dir,
        output_df = output_df)

    #### Sex ####
    male_set = raw_df[raw_df['sex'].str.upper().isin(['M', 'MALE'])].reset_index()
    female_set = raw_df[raw_df['sex'].str.upper().isin(['F', 'FEMALE'])].reset_index()

    output_df = run_hypertest(
        mode = 'male_set', 
        test = 'classification_report',
        y = male_set['pred_bin'],
        t = male_set['label_bin'],
        eval_dir = eval_dir,
        output_df = output_df)

    output_df = run_hypertest(
        mode = 'female_set', 
        test = 'classification_report',
        y = female_set['pred_bin'],
        t = female_set['label_bin'],
        eval_dir = eval_dir,
        output_df = output_df)

    #### Slice Thickness ####
    slice_thick_05_08 = raw_df[raw_df['slice_thickness'] >= 0.5]
    slice_thick_05_08 = slice_thick_05_08[slice_thick_05_08['slice_thickness'] <= 0.8].reset_index()
    slice_thick__08_25 = raw_df[raw_df['slice_thickness'] > 0.8]
    slice_thick__08_25 = slice_thick__08_25[slice_thick__08_25['slice_thickness'] <= 2.5].reset_index()

    output_df = run_hypertest(
        mode = 'slice_thick_05_08', 
        test = 'classification_report',
        y = slice_thick_05_08['pred_bin'],
        t = slice_thick_05_08['label_bin'],
        eval_dir = eval_dir,
        output_df = output_df)

    output_df = run_hypertest(
        mode = 'slice_thick__08_25', 
        test = 'classification_report',
        y = slice_thick__08_25['pred_bin'],
        t = slice_thick__08_25['label_bin'],
        eval_dir = eval_dir,
        output_df = output_df)

    #### Manufacturer ####
    ge_set = raw_df[raw_df['manufacturer'].str.upper() == 'GE MEDICAL SYSTEMS'].reset_index()
    siemens_set = raw_df[raw_df['manufacturer'].str.upper() == 'SIEMENS'].reset_index()
    philips_set = raw_df[raw_df['manufacturer'].str.upper() == 'PHILIPS'].reset_index()
    toshiba_set = raw_df[raw_df['manufacturer'].str.upper() == 'TOSHIBA'].reset_index()
    other_set = raw_df[raw_df['manufacturer'].str.upper().isin(['PHILIPS', 'TOSHIBA'])].reset_index()

    output_df = run_hypertest(
        mode = 'ge_set', 
        test = 'classification_report',
        y = ge_set['pred_bin'],
        t = ge_set['label_bin'],
        eval_dir = eval_dir,
        output_df = output_df)

    output_df = run_hypertest(
        mode = 'siemens_set', 
        test = 'classification_report',
        y = siemens_set['pred_bin'],
        t = siemens_set['label_bin'],
        eval_dir = eval_dir,
        output_df = output_df)

    output_df = run_hypertest(
        mode = 'philips_set', 
        test = 'classification_report',
        y = philips_set['pred_bin'],
        t = philips_set['label_bin'],
        eval_dir = eval_dir,
        output_df = output_df)

    output_df = run_hypertest(
        mode = 'toshiba_set', 
        test = 'classification_report',
        y = toshiba_set['pred_bin'],
        t = toshiba_set['label_bin'],
        eval_dir = eval_dir,
        output_df = output_df)

    output_df = run_hypertest(
        mode = 'other_set', 
        test = 'classification_report',
        y = other_set['pred_bin'],
        t = other_set['label_bin'],
        eval_dir = eval_dir,
        output_df = output_df)

    #### PPV Analysis ####
    total_count = len(raw_df)
    positive_count = len(positive_set)
    negative_count = len(negative_set)

    prevalence = 0.5425
    assert(prevalence == positive_count / total_count)
    assert(positive_count == (prevalence * negative_count) / (1 - prevalence))

    prevalence = 0.40
    pos_count_40 = int((prevalence * negative_count) / (1 - prevalence))
    prevalence = 0.20
    pos_count_20 = int((prevalence * negative_count) / (1 - prevalence))
    prevalence = 0.10
    pos_count_10 = int((prevalence * negative_count) / (1 - prevalence))
    prevalence = 0.05
    pos_count_5 = int((prevalence * negative_count) / (1 - prevalence))

    modes = ['pos_54_25_set', 'pos_40_set', 'pos_20_set', 'pos_10_set', 'pos_5_set']
    pos_counts = [positive_count, pos_count_40, pos_count_20, pos_count_10, pos_count_5]
    test = 'classification_report'
    np.random.seed(7)
    seeds = np.random.randint(low=0, high=65536, size=10)
    
    for i in range(len(modes)):

        mode = modes[i]
        pos_count = pos_counts[i]
        ppv_pe_arr = []
        ppv_sample_size_arr = []

        for seed in seeds:

            pos_set = positive_set.sample(n=pos_count, random_state=seed)
            pos_set = pd.concat([pos_set, negative_set], ignore_index=True)

            test_class = Hypertest(
                y = pos_set['pred_bin'],
                t = pos_set['label_bin'],
                name = test + '_' + mode + '_10_fold',
                task = 'classification',
                results_folder = eval_dir,
                experiment_list = [test]).run()

            ppv = test_class.get_result('ppv')
            ppv_pe_arr.append(ppv.point_estimate)
            ppv_sample_size_arr.append(ppv.sample_size)

        ppv_mean = np.mean(ppv_pe_arr)
        ppv_std = np.std(ppv_pe_arr)
        ppv_sample_size = np.mean(ppv_sample_size_arr)
        output_df = output_df.append({
            'metric': 'ppv_' + mode,
            'point_estimate': round(ppv_mean, 3),
            'lower_bound': round(ppv_mean-ppv_std, 3),
            'upper_bound': round(ppv_mean+ppv_std, 3),
            'sample_size': ppv_sample_size}, ignore_index=True)

    output_df.to_csv(input_path.replace('.csv', '_report.csv'))


