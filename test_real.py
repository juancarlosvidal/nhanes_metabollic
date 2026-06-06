# Entry point for training and evaluating the binary classification model on real NHANES data.
# Handles data loading, class balancing via upsampling, feature scaling, and K-fold CV.
import os
import argparse
import pandas as pd
import random
import numpy as np

import torch
# torch.use_deterministic_algorithms(True)

from train import train_conformance_inference
from data_real import load_data
from utils import get_device

import logging
# logging.basicConfig(format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s', level=logging.INFO)
logging.basicConfig(format='%(message)s', level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-i', '--input_dir', default="./conformance_real/real", help='Input_directory')
    parser.add_argument('-o', '--output_dir', default="./output", help='Output_directory')
    parser.add_argument('-c', '--combination', default=0, help='Selection of variables')
    parser.add_argument('-f', '--n_folds', default=5, help='Number of cv folders')
    parser.add_argument('-e', '--n_epochs', default=1000, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', default=64, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.001, help='Learning rate')
    parser.add_argument('-wd', '--weight_decay', default=0.01, help='Weight decay')
    args = parser.parse_args()

    # Fix random seeds for reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    n_folds = args.n_folds
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    # alpha = 0.1
    dropout = 0.2
    patience=4
    min_delta=0.0
    hidden_sizes=[8, 4, 2]
    # hidden_sizes=[10, 5, 2]


    # Feature combinations to experiment with (NHANES variable codes)
    # Bioch: A1c, FPG, cholesterol, triglycerides
    # Metadata: age and gender
    # Clinical measures:  body mass index, waist circunference, diastolic and diastolic pressure, pulse, cholesterol
    comb_0 = ['RIDAGEYR.x', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS', 'LBDSCHSI_43',
              'LBXSTR_43', 'LBXSGL_43', 'RIAGENDR', 'LBXGH_39']
    # Without A1c
    comb_1 = ['RIDAGEYR.x', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS',
              'BPXDI1', 'LBDSCHSI_43', 'LBXSTR_43', 'LBXSGL_43', 'RIAGENDR']
    # Without FPG
    comb_2 = ['RIDAGEYR.x', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS', 'LBDSCHSI_43',
              'LBXSTR_43', 'RIAGENDR', 'LBXGH_39']
    # Same without A1c and FPG
    comb_3 = ['RIDAGEYR.x', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS',
              'BPXDI1', 'LBDSCHSI_43', 'LBXSTR_43', 'RIAGENDR']
    # Without any biochemical measures
    comb_4 = ['RIDAGEYR.x', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS',
              'BPXDI1', 'RIAGENDR']
    # Only metadata
    comb_5 = ['RIDAGEYR.x', 'RIAGENDR']
    # Only clinical measures
    comb_6 = ['BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS',
              'BPXDI1']

    # Select the feature combination to use based on the CLI argument
    all_combinations = [comb_0, comb_1, comb_2, comb_3, comb_4, comb_5, comb_6]
    selected_combination = all_combinations[int(args.combination)]
    output_var = 'diabetes'

    alphas = np.linspace(0.1, 1.0, num=10)
    dfi = pd.DataFrame()
    dfo = pd.DataFrame()
    onlyfiles = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    for f in onlyfiles:
        file = os.path.join(args.input_dir, f)
        logging.info('Running conformal inference for file {}'.format(file))

        data = load_data(file, selected_combination, output_var)

        # Random 75/25 train/test split
        n = len(data['y'])
        indices = random.sample(np.arange(n).tolist(), n)
        train_split = indices[:int((3*n) / 4)]
        test_split = indices[int((3*n) / 4):]

        train_data = {
            'seqn': data['seqn'][train_split],
            'x': data['x'][train_split],
            'y': data['y'][train_split],
            'w': data['w'][train_split],
            'z': data['z'][train_split]
        }

        # Scale training data using StandardScaler (fit on train, apply to test)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        train_data['x'] = scaler.fit_transform(train_data['x'])

        # Upsample the minority class (diabetic) to match non-diabetic count
        from sklearn.utils import resample
        y = train_data['y'].reshape(-1)
        diabetic = np.argwhere(y == 1.0).reshape(-1)
        nondiabetic = np.argwhere(y == 0.0).reshape(-1)
        print('Diabetic: {}'.format(diabetic.shape))
        print('Non-diabetic: {}'.format(nondiabetic.shape))

        diabetic_upsample = resample(diabetic, replace=True, n_samples=len(nondiabetic), random_state=42)
        upsampled = np.concatenate([diabetic_upsample, nondiabetic]).reshape(-1)
        print('Upsampled dataset: {}'.format(upsampled.shape))

        # Rebuild train_data from original (unscaled) features after upsampling
        train_data = {
            'seqn': data['seqn'][upsampled],
            'x': data['x'][upsampled],
            'y': data['y'][upsampled],
            'w': data['w'][upsampled],
            'z': data['z'][upsampled]
        }

        test_data = {
            'seqn': data['seqn'][test_split],
            'x': data['x'][test_split],
            'y': data['y'][test_split],
            'w': data['w'][test_split],
            'z': data['z'][test_split]
        }

        # Scale test data using the scaler fitted on the training set
        test_data['x'] = scaler.transform(test_data['x'])

        from train_bc import cv_loop_bc
        from sklearn.model_selection import KFold
        from metrics import compute_classification_metrics

        # Set up K-Fold cross-validation on the training data
        k_fold = KFold(n_splits=n_folds, random_state=42, shuffle=True)
        indexes = sorted(range(len(train_data['y']) - 1))
        splits = k_fold.split(indexes)

        # Train the binary classification model via cross-validation
        model_ce, metrics_ce = cv_loop_bc(
            data=train_data,
            splits=splits,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            min_delta=min_delta,
            dropout=dropout,
            hidden_sizes=hidden_sizes
            # hidden_sizes=[10, 5, 2]
        )

        result = {
            'filename': os.path.basename(f),
            'combination': selected_combination,
            'auc': metrics_ce['auc'],
            'accuracy': metrics_ce['accuracy'],
            'recall': metrics_ce['recall'],
            'precision': metrics_ce['precision'],
            'f1': metrics_ce['f1'],
            'cm': metrics_ce['cm'],
            'ce': metrics_ce['ce']
        }
        logging.info(result)

        # Evaluate the best model on the held-out test set
        model_ce.eval()
        device = get_device()
        with torch.no_grad():
            yp = model_ce(torch.from_numpy(test_data['x']).to(device)).cpu().detach().numpy()
        results_test = compute_classification_metrics(test_data['w'], yp, test_data['y'])
        logging.info(results_test)




    #     # alphas = [0.2]
    #     for alpha in alphas:
    #         predictor = train_conformance_inference(
    #             data=train_data,
    #             n_folds=n_folds,
    #             n_epochs=n_epochs,
    #             batch_size=batch_size,
    #             learning_rate=learning_rate,
    #             weight_decay=weight_decay,
    #             alpha=alpha,
    #             patience_classification=4,
    #             min_delta_classification=0.0,
    #             patience_regression=6,
    #             min_delta_regression=0.0
    #         )
    #
    #         ci_result, correct = predictor.classify(test_data)
    #
    #         result = {
    #             'filename': os.path.basename(f),
    #             'alpha': alpha,
    #             'correct': correct
    #         }
    #
    #         output_df = pd.DataFrame([result])
    #         dfo = pd.concat([dfo, output_df], ignore_index=True)
    #
    # print("Saving file: {}".format(os.path.join(args.output_dir, "out_ci.csv")))
    # output_file = os.path.join(args.output_dir, os.path.basename(args.input_dir) + '_' + 'out_ci.csv')
    # dfo.to_csv(output_file)
