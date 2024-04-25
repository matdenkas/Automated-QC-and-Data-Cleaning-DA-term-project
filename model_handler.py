import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error

from cust_data_types import Batch_Info, Data_Set, Job


def run_jobs(jobs: list[Job]):

    job_evaluations = []
    for j_idx, job in enumerate(jobs):
        print(f'\tStarting job: #{j_idx} -- {job.batch_info.name}...')
        models = build_batch(job.batch_info)
        fitted_models = fit_batch(models, job.data_set, job.patience, job.epochs)
        for idx, model in enumerate(fitted_models):
            model.save(f'./models/{job.batch_info.name}_{idx}', save_format='h5')

        if(job.binary):
            batch_evaluation = [binary_eval(model, job.data_set) for model in fitted_models]
        else:
            batch_evaluation = [evaluate_model(model, job.data_set) for model in fitted_models]
        min, max, mean = meta_eval(batch_evaluation)
        job_evaluations.append({'min': min, 'max': max, 'mean': mean})

    return job_evaluations

def build_batch(batch_info: Batch_Info):

    batch = []
    for _ in range(batch_info.batch_size):

        model = Sequential()
        model.add(InputLayer(batch_info.input_shape))
        for complexity in batch_info.architecture:
            model.add(Dense(complexity, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', 
                      optimizer='adam'
                    )

        batch.append(model)
    return batch


def fit_batch(batch_models, data_set: Data_Set, patience=20, epochs=1000):

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    for model in batch_models:
        # Train the model
        model.fit(
                data_set.data_training, data_set.labels_training,
                epochs=epochs,
                validation_data=(data_set.data_validation, data_set.labels_validation),
                callbacks=[callback]
        )

    return batch_models

def evaluate_model(model, data_set: Data_Set):

    predictions = model.predict(data_set.data_testing)
    less_than_1_counter = 0
    for index, prediction in enumerate(predictions):
        val = data_set.labels_testing[index]
        if abs(val - prediction) <= 1:
            less_than_1_counter += 1

    less_than_1_counter = (less_than_1_counter / len(predictions)) * 100

    average_error = average_error = np.mean(data_set.labels_testing - predictions)
    mse = mean_squared_error(data_set.labels_testing, predictions)
    rmse = root_mean_squared_error(data_set.labels_testing, predictions)
    mae = mean_absolute_error(data_set.labels_testing, predictions)
    medae = median_absolute_error(data_set.labels_testing, predictions)
    r2 = r2_score(data_set.labels_testing, predictions)
    test_loss = model.evaluate(data_set.data_testing, data_set.labels_testing)
    return {
            'cf_15cm_percentage': less_than_1_counter, 
            'average_error': average_error, 
            'mse': mse, 
            'rmse': rmse, 
            'mae': mae, 
            'medae': medae, 
            'r2': r2, 
            'test_loss': test_loss
        }


def binary_eval(model, data_set: Data_Set):

    predictions = model.predict(data_set.data_testing)
    truth = data_set.labels_testing

    TP = 0 # Was Bad pred bad
    FP = 0 # Was Bad pred good
    FN = 0 # Was good pred bad
    TN = 0 # Was good pred good
    for idx, prediction in enumerate(predictions):
        
        was_bad = truth[idx] == 0
        pred_bad = prediction > 0.95

        if was_bad and pred_bad: TP += 1
        if was_bad and not pred_bad: FP += 1
        if not was_bad and pred_bad: FN += 1
        if not was_bad and not pred_bad: TN += 1

    return {
            'recall': (TP / ((TP + FN) if (TP + FN) != 0 else -1)), 
            'precision': (TP / ((TP + FP) if (TP + FP) != 0 else -1)), 
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN,
        }

def meta_eval(evals):
    min = {key: 99999 for key in evals[0].keys()}
    max = {key: -99999 for key in evals[0].keys()}
    avg = {key: [] for key in evals[0].keys()}
    for eval in evals:
        for k, v in eval.items():
            if v < min[k]:
                min[k] = v

            if v > max[k]:
                max[k] = v

            avg[k].append(v)

    for k, v in avg.items():
        sum = 0
        for value in v:
            sum = sum + value
        avg[k] = sum / len(v)

    return min, max, avg