# Cross-validation training loop for the quantile regression (pinball loss) model.
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from data import CustomDataset
from utils import run_epoch
from metrics import regression_mean_metrics
from models import RegressionModel
from losses import PinballLoss
import logging


def cv_loop_rr(data, splits, n_epochs, batch_size, learning_rate, weight_decay,
               alpha=0.1, patience=5, min_delta=0, dropout=0.20, hidden_sizes=None):
    """
    Run K-fold cross-validation for the quantile regression model.

    Tracks the best model (lowest validation loss) across all folds.

    Returns:
    --------
    tuple: (best_model, mean_regression_metrics_dict)
    """
    model = None
    min_loss = np.inf
    metrics_list = []
    for train_split, test_split in splits:
        # Select the 20% of the train size as the validation set
        train_size = round(len(train_split) * 0.8)
        train_index, val_index, test_index = train_split[:train_size], train_split[train_size:], test_split

        fold_model, fold_min_loss, fold_test_metrics = run_fold(
            data=data,
            train_split=train_index,
            val_split=val_index,
            test_split=test_index,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            alpha=alpha,
            patience=patience,
            min_delta=min_delta,
            dropout=dropout,
            hidden_sizes=hidden_sizes
        )

        metrics_list.append(fold_test_metrics)
        # Keep the model with the lowest validation loss across folds
        if fold_min_loss < min_loss:
            model = fold_model
            min_loss = fold_min_loss

    return model, regression_mean_metrics(metrics_list)


def run_fold(data, train_split, val_split, test_split, n_epochs, batch_size, learning_rate, weight_decay,
             alpha=0.1, patience=5, min_delta=0, dropout=0.20, hidden_sizes=None):
    """
    Train and evaluate the quantile regression model on a single fold using PinballLoss.

    Uses early stopping based on validation loss. Learning rate is reduced at
    50% and 75% of total epochs via MultiStepLR.

    Returns:
    --------
    tuple: (best_model, min_val_loss, test_metrics_dict)
    """
    min_loss = np.inf
    best_model = None

    # input, target, weight, t = data['x'], data['y'], data['w'], data['t']
    input, target, weight = data['x'], data['y'], data['w']
    train_input, train_target, train_weight = \
        input[train_split], target[train_split], weight[train_split]
    val_input, val_target, val_weight = \
        input[val_split], target[val_split], weight[val_split]
    test_input, test_target, test_weight = \
        input[test_split], target[test_split], weight[test_split]

    # Create DataLoaders for each split
    train_loader, val_loader, test_loader = \
        DataLoader(CustomDataset(train_input, train_target, train_weight), batch_size=batch_size), \
        DataLoader(CustomDataset(val_input, val_target, val_weight), batch_size=batch_size), \
        DataLoader(CustomDataset(test_input, test_target, test_weight), batch_size=batch_size)

    # Declaring the model
    input_size = train_input.shape[1]
    output_size = 1
    if hidden_sizes is None:
        hidden_sizes = [10, 5, 2]
    model = RegressionModel(input_size=input_size, output_size=output_size, dropout=dropout, hidden_sizes=hidden_sizes)

    # Declaring the criterion (pinball loss at the given alpha quantile)
    criterion = PinballLoss(alpha)

    # Declaring the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Declaring the scheduler: reduce LR at 50% and 75% of training
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)

    counter = 0  # Early stopping patience counter
    for epoch in range(1, n_epochs + 1):
        # ####  TRAIN LOOP  #### #
        _, train_loss, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            metric_type=1,
            train=True
        )

        scheduler.step()

        # ####  VALIDATION LOOP  #### #
        _, val_loss, _ = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            metric_type=1,
            train=False
        )

        logging.debug('Epoch train loss {:.3f} val loss {:.3f}'.format(train_loss, val_loss))

        # Determine if model is the best
        if val_loss < min_loss:
            logging.debug('New min loss {:.3f}'.format(val_loss))
            min_loss = val_loss
            best_model = model
            counter = 0
        elif val_loss > (min_loss + min_delta):
            counter += 1
            logging.debug('Delta count {} and val_loss {:.3f}'.format(counter, val_loss))
            if counter >= patience:
                break  # Early stopping triggered

    model_t = best_model

    # ####  TEST LOOP  #### #
    _, _, test_metrics = run_epoch(
        model=model_t,
        loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        epoch=0,
        n_epochs=n_epochs,
        metric_type=1,
        train=False
    )

    logging.debug('Fold Test MAE: {:.3f} MSE: {:.3f} R2: {:.3f}'.format(
        test_metrics['mae'], test_metrics['mse'], test_metrics['r2']
    ))

    return model_t, min_loss, test_metrics
