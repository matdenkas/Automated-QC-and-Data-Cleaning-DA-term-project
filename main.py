from cust_data_types import Data_Set, Job, Batch_Info
from data_set_generation import Data_Generator
from model_handler import run_jobs
from visualization import plot_before_after_df, hist, pprint, prep_result_for_excel
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np

from imblearn.over_sampling import SMOTE
from collections import Counter


def get_all_unique_data_generators():
    # Generate every unique data set utilizing different tests
    # https://stackoverflow.com/questions/60237583/python-iterate-over-all-possible-combinations-on-boolean-variables
    data_generators = []
    encodings = []
    for p in itertools.product([True, False], repeat= 5):
        data_gen = Data_Generator()
        data_gen.manual_selected_removal = p[0]
        data_gen.spike_test = p[1]
        data_gen.roc_test = p[2]
        data_gen.flat_line_test = p[3]
        data_gen.attenuated_signal_test = p[4]
        data_generators.append(data_gen)
        encodings.append(''.join([str(int(val)) for val in p]))
    return zip(data_generators, encodings)


def log_result_to_file(line):
    file1 = open("result.log", "a")  # append mode
    file1.write(line + '\n')
    file1.close()


def train_models_on_datasets():
    ### EXPERIMENT 1
    ARCHITECTURE= [8, 1]
    NAME= ''.join([str(val) + '_' for val in ARCHITECTURE])
    BATCH_SIZE = 5
    EPOCHS= 1000
    PATIENCE= 20 

    print('Building Jobs...')
    jobs = []
    for data_gen, encoding in get_all_unique_data_generators():
        data_set = data_gen.generate_data_set()

        batch_info = Batch_Info(
            architecture= ARCHITECTURE,
            batch_size= BATCH_SIZE,
            input_shape= data_set.input_shape,
            name= NAME + encoding
        )

        jobs.append(Job(
            batch_info= batch_info,
            data_set= data_set,
            epochs= EPOCHS,
            patience= PATIENCE
        ))
        print(f'\t{batch_info.name} ready...')

    print(f'Executing {len(jobs)} jobs...')
    results = run_jobs(jobs)

    print('Dumping results...')
    for idx, result in enumerate(results):
        log_result_to_file(prep_result_for_excel(result, jobs[idx].batch_info.name))

    print('fin!')


### EXPERIMENT 2
def train_ml_classifiers():
    ARCHITECTURE= [[60,]]
    NAMES= [''.join([str(val) + '_' for val in arch]) for arch in ARCHITECTURE]
    BATCH_SIZE = 5
    EPOCHS= 1000
    PATIENCE= 20 

    data_gen = Data_Generator()
    # 01100
    data_gen.manual_selected_removal =  False
    data_gen.spike_test =               True
    data_gen.roc_test =                 True
    data_gen.flat_line_test =           False
    data_gen.attenuated_signal_test =   False
    data_set = data_gen.generate_binarized_set()


    # https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(data_set.data_training, data_set.labels_training)
    data_set.data_training = X_res
    data_set.labels_training = y_res.reshape((y_res.shape[0], 1))
    jobs = []
    for idx, arch in enumerate(ARCHITECTURE):
        batch_info = Batch_Info(
            architecture= arch,
            batch_size= BATCH_SIZE,
            input_shape= data_set.input_shape,
            name= NAMES[idx]
        )

        job = Job(
            batch_info= batch_info,
            data_set= data_set,
            epochs= EPOCHS,
            patience= PATIENCE,
            binary= True
        )
        jobs.append(job)

    results = run_jobs(jobs)
    for idx, result in enumerate(results):
            log_result_to_file(prep_result_for_excel(result, jobs[idx].batch_info.name))
