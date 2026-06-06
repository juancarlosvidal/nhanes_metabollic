# Conformal inference training pipeline: trains a classifier (CE), a quantile regressor (RR),
# and calibrates conformity scores on a held-out calibration set.
import random
import logging

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import KFold, StratifiedKFold

from train_bc import cv_loop_bc
from train_rr import cv_loop_rr
from utils import weighted_quantile, get_device

device = get_device()


###############################################################################
class Predictor(object):
    """
    Conformal predictor that combines a classification model (CE) and a quantile
    regression model (RR) to produce prediction sets at a given coverage level alpha.
    """

    def __init__(self, alpha, conformity_scores, model_ce, model_rr, weights=None):
        self._alpha = alpha  # Desired miscoverage rate
        self._conformity_scores = conformity_scores  # Calibration conformity scores
        self._model_ce = model_ce  # Classifier (outputs P(y=1|x))
        self._model_rr = model_rr  # Quantile regressor
        self._weights = weights  # Calibration sample weights

    def classify(self, data):
        """
        Assign each test sample a class label using conformal prediction sets.

        Class 2 = both classes included in the prediction set (uncertain).
        Class 1 = only positive class.
        Class 0 = only negative class.
        Class -1 = empty prediction set (no class satisfies the condition).

        Returns:
        --------
        ci_result : dict with conditions, quantiles, scores, predicted classes, and correctness.
        correct_2 : weighted fraction of correctly classified samples (class 2 counted as correct).
        """
        # Generate predictions from both models
        data4 = {
            "x": data["x"],
            "y": self._model_rr(torch.from_numpy(data["x"]).to(device))
            .cpu()
            .detach()
            .numpy(),
            "p": self._model_ce(torch.from_numpy(data["x"]).to(device))
            .cpu()
            .detach()
            .numpy(),
            "w": data["w"],
            # 't': data['t'],
            "z": data["z"],
        }

        if "y" in data.keys():
            # Flip quantile output for negative-class samples
            indexes = np.where(data["y"] == 0)
            data4["y"][indexes] = 1 - data4["y"][indexes]
            y = data["y"].ravel()

        patient_class = []
        condition_list = []
        correct_list = []
        correct_1, correct_2, correct_3, correct_4 = 0, 0, 0, 0

        # Normalize test weights and compute classifier scores
        tw_4 = (data4["w"] / np.sum(data4["w"])).ravel()
        empirical_quantile_4 = (1 - self._alpha) * (1 + 1 / data4["x"].shape[0])
        score_output_4 = (
            self._model_ce(torch.from_numpy(data4["x"]).to(device))
            .cpu()
            .detach()
            .numpy()
        )
        quant_output_4 = data4["y"]

        if self._weights is None:
            self._weights = np.ones(len(data4["w"]), dtype=np.float32)

        for i in range(len(data4["x"])):
            # Append a sentinel infinity score for the test point, then compute weighted quantile
            tw_aux = np.append(self._weights, data4["w"][i])
            tw_aux = tw_aux / np.sum(tw_aux)
            conformity_scores_aux = np.append(self._conformity_scores, 1e20)
            empirical_quantile_aux = weighted_quantile(
                conformity_scores_aux, empirical_quantile_4, tw_aux
            )
            # empirical_quantile_aux = np.quantile(conformity_scores_aux, empirical_quantile_4)

            # Condition: quantile output minus the conformal threshold
            condition = quant_output_4[i] - empirical_quantile_aux
            condition_list.append(condition.item())

            score_pos = score_output_4[i]  # P(y=1|x)
            score_neg = 1 - score_output_4[i]  # P(y=0|x)

            # Determine which classes are included in the prediction set
            if condition <= score_pos and condition <= score_neg:
                patient_class.append(2)  # Both classes: uncertain
            elif condition <= score_pos:
                patient_class.append(1)  # Only positive
            elif condition <= score_neg:
                patient_class.append(0)  # Only negative
            else:
                patient_class.append(-1)  # Empty set

            if "y" in data.keys():
                if patient_class[i] == 2:
                    correct_2 += 1 * float(tw_4[i])
                    correct_list.append(1 * float(tw_4[i]))
                else:
                    correct_2 += (patient_class[i] == int(y[i])) * float(tw_4[i])
                    correct_list.append((patient_class[i] == int(y[i])))

                correct_4 += (patient_class[i] == int(y[i])) * float(tw_4[i])

                logging.info(
                    "Alpha: {:.3f} - Condition: {:.3f} - P1: {:.3f} - P0: {:.3f} - PC/YC: {:2d}/{:2d} ({:5})".format(
                        self._alpha,
                        condition.item(),
                        score_pos.item(),
                        score_neg.item(),
                        patient_class[i],
                        int(y[i].item()),
                        str(patient_class[i] == int(y[i])),
                    )
                )

        if "y" in data.keys():
            # logging.info("Prediction_class vs probability_class: {:.3f}".format(correct_1))
            logging.info("Prediction_class vs y_class: {:.3f}".format(correct_2))
            # logging.info("Strict prediction_class vs probability_class: {:.3f}".format(correct_3))
            logging.info("Strict prediction_class vs y_class: {:.3f}".format(correct_4))

        ci_result = {
            "conditions": condition_list,
            "quantiles": quant_output_4,
            "score_pos": score_output_4,
            "score_neg": 1 - score_output_4,
            "ci": patient_class,
            "p": data4["p"],
            "y": np.where(data4["p"] > 0.5, 1, 0).reshape(-1),
        }

        if "y" in data.keys():
            ci_result["correct"] = correct_list

        return ci_result, correct_2


def train_conformance_inference(
    data,
    n_folds,
    n_epochs,
    batch_size,
    learning_rate,
    weight_decay,
    alpha,
    patience_classification=5,
    min_delta_classification=0,
    patience_regression=5,
    min_delta_regression=0,
    dropout=0.20,
    hidden_sizes_bc=None,
    hidden_sizes_rr=None,
):
    """
    Train the full conformal inference pipeline using three non-overlapping data splits:
      - set1: train the binary classifier (CE model).
      - set2: train the quantile regressor (RR model) on CE outputs.
      - set3: compute conformity scores for calibration.

    Returns:
    --------
    Predictor instance ready to classify test data.
    """
    n = len(data["y"])
    indices = random.sample(np.arange(n).tolist(), n)

    # Partition data into three sets (approximately 1/3 each; set3 overlaps with set2)
    set1 = indices[: int(n / 3)]
    set2 = indices[int(n / 3) : int(2 * n / 3)]
    set3 = indices[int(2 * n / 3) :]

    # --- Step 1: Train the binary classifier on set1 ---
    data1 = {
        "x": data["x"][set1],
        "y": data["y"][set1],
        "w": data["w"][set1],
        # 't': data['t'][set1],
        "z": data["z"][set1],
    }

    k_fold = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
    indexes = sorted(range(len(data1["w"])))
    splits = k_fold.split(indexes, data1["y"].ravel())

    model_ce, metrics_ce = cv_loop_bc(
        data=data1,
        splits=splits,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience_classification,
        min_delta=min_delta_classification,
        dropout=dropout,
        hidden_sizes=hidden_sizes_bc,
    )

    model_ce.eval()

    # --- Step 2: Train the quantile regressor on CE outputs for set2 ---
    data2 = {
        "x": data["x"][set2],
        # Use CE model predictions as regression targets
        "y": model_ce(torch.from_numpy(data["x"][set2]).to(device))
        .cpu()
        .detach()
        .numpy(),
        # 'y': torch.log(model_ce(torch.from_numpy(data['x'][set2]))).cpu().detach().numpy(),
        "w": data["w"][set2],
        # 't': data['t'][set2],
        "z": data["z"][set2],
    }

    # Flip predictions for negative-class samples
    indexes = np.where(data["y"][set2] == 0)
    data2["y"][indexes] = 1 - data2["y"][indexes]

    k_fold = KFold(n_splits=n_folds, random_state=42, shuffle=True)
    indexes = sorted(range(len(data2["w"])))
    splits = k_fold.split(indexes)

    model_rr, metrics_rr = cv_loop_rr(
        data=data2,
        splits=splits,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        alpha=alpha,
        patience=patience_regression,
        min_delta=min_delta_regression,
        dropout=dropout,
        hidden_sizes=hidden_sizes_rr,
    )

    model_rr.eval()

    # --- Step 3: Compute conformity scores on the calibration set (set3) ---
    data3 = {
        "x": data["x"][set3],
        "y": model_rr(torch.from_numpy(data["x"][set3]).to(device))
        .cpu()
        .detach()
        .numpy(),
        "w": data["w"][set3],
        # 't': data['t'][set3],
        "z": data["z"][set3],
    }

    # Flip quantile outputs for negative-class calibration samples
    indexes = np.where(data["y"][set3] == 0)
    data3["y"][indexes] = 1 - data3["y"][indexes]

    # Compute conformity scores on data3 set
    empirical_quantile = (1 - alpha) * (1 + 1 / data3["x"].shape[0])
    logging.debug("Empirical quantile 1: {}".format(empirical_quantile))

    # score_output = model_ce(torch.from_numpy(data3['x']).to(device)).cpu().detach().numpy() + data3['z'] * sigma
    score_output = (
        model_ce(torch.from_numpy(data3["x"]).to(device)).cpu().detach().numpy()
    )
    quant_output = data3["y"]
    logging.debug("Score output shape {}".format(score_output.shape))
    logging.debug("Quantile output shape {}".format(quant_output.shape))

    # Flip CE scores for negative-class samples before computing conformity scores
    negative_index = np.where(data["y"][set3] == 0)
    score_output[negative_index] = 1 - score_output[negative_index]
    conformity_scores_3 = quant_output - score_output

    return Predictor(
        alpha=alpha,
        conformity_scores=conformity_scores_3,
        model_ce=model_ce,
        model_rr=model_rr,
        weights=data3["w"],
    )
