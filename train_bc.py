# Cross-validation training loop for the binary classification (BCE) model.
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from data import CustomDataset
from utils import run_epoch
from metrics import classification_mean_metrics
from models import BinaryClassificationModel
from losses import CustomBCELoss
import logging


def cv_loop_bc(data, splits, n_epochs, batch_size, learning_rate, weight_decay,
               patience=5, min_delta=0, dropout=0.20, hidden_sizes=None, refit=False, pos_weight=1.0):
    """
    Run K-fold cross-validation for the binary classification model.

    Tracks the best model (lowest validation loss) across all folds.
    If refit=True, after CV a final model is retrained on the full training data
    for the average number of epochs at which early stopping fired across folds.

    Returns:
    --------
    tuple: (best_model, mean_classification_metrics_dict)
    """
    model = None
    min_loss = np.inf
    metrics_list = []
    best_epochs = []
    for train_split, val_split in splits:
        fold_model, fold_min_loss, fold_test_metrics, fold_best_epoch = run_fold(
            data=data,
            train_split=train_split,
            val_split=val_split,
            test_split=val_split,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            min_delta=min_delta,
            dropout=dropout,
            hidden_sizes=hidden_sizes,
            pos_weight=pos_weight
        )

        metrics_list.append(fold_test_metrics)
        best_epochs.append(fold_best_epoch)
        # Keep the model with the lowest validation loss across folds
        if fold_min_loss < min_loss:
            model = fold_model
            min_loss = fold_min_loss

    if refit:
        avg_best_epoch = max(1, int(np.mean(best_epochs)))
        logging.debug('Refitting final model on full training data for {} epochs'.format(avg_best_epoch))
        model = _train_final(
            data=data,
            n_epochs=avg_best_epoch,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            hidden_sizes=hidden_sizes,
            pos_weight=pos_weight
        )

    return model, classification_mean_metrics(metrics_list)


def _train_final(data, n_epochs, batch_size, learning_rate, weight_decay, dropout, hidden_sizes, pos_weight=1.0):
    """
    Train a model on the full training dataset for a fixed number of epochs.
    No validation split, no early stopping — used after CV to produce the final model.
    """
    from data import CustomDataset
    from torch.utils.data import DataLoader

    input_data, target, weight = data['x'], data['y'], data['w']
    loader = DataLoader(
        CustomDataset(input_data, target, weight),
        batch_size=batch_size,
        shuffle=True
    )

    input_size = input_data.shape[1]
    if hidden_sizes is None:
        hidden_sizes = [10, 5, 2]
    model = BinaryClassificationModel(
        input_size=input_size, output_size=1,
        dropout=dropout, hidden_sizes=hidden_sizes
    )
    criterion = CustomBCELoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.5 * n_epochs), int(0.75 * n_epochs)],
        gamma=0.5
    )

    for epoch in range(1, n_epochs + 1):
        model, _, _ = run_epoch(
            model=model, loader=loader, criterion=criterion,
            optimizer=optimizer, epoch=epoch, n_epochs=n_epochs,
            metric_type=0, train=True
        )
        scheduler.step()

    return model


# def run_fold(train_dl, val_dl, test_dl, n_epochs, learning_rate, weight_decay, best_error):
def run_fold(data, train_split, val_split, test_split, n_epochs, batch_size, learning_rate, weight_decay,
             patience=5, min_delta=0, dropout=0.20, hidden_sizes=None, pos_weight=1.0):
    """
    Train and evaluate the binary classification model on a single fold.

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
        DataLoader(CustomDataset(train_input, train_target, train_weight), batch_size=batch_size, shuffle=True), \
        DataLoader(CustomDataset(val_input, val_target, val_weight), batch_size=batch_size), \
        DataLoader(CustomDataset(test_input, test_target, test_weight), batch_size=batch_size)

    # Declaring the model
    input_size = train_input.shape[1]
    output_size = 1
    if hidden_sizes is None:
        hidden_sizes = [10, 5, 2]
    model = BinaryClassificationModel(input_size=input_size, output_size=output_size, dropout=dropout, hidden_sizes=hidden_sizes)

    # Declaring the criterion
    criterion = CustomBCELoss(pos_weight=pos_weight)

    # Declaring the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.5 * n_epochs), int(0.75 * n_epochs)],
        gamma=0.5
    )

    counter = 0  # Early stopping patience counter
    best_epoch = 1
    for epoch in range(1, n_epochs + 1):
        # ####  TRAIN LOOP  #### #
        model, train_loss, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            metric_type=0,
            train=True
        )

        # ####  VALIDATION LOOP  #### #
        _, val_loss, _ = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            metric_type=0,
            train=False
        )

        scheduler.step()
        logging.debug('Epoch train loss {:.3f} val loss {:.3f}'.format(train_loss, val_loss))

        # Determine if model is the best
        if val_loss < min_loss - min_delta:
            logging.debug('New min loss {:.5f}'.format(val_loss))
            min_loss = val_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            counter = 0
        else:
            counter += 1
            logging.debug('Delta count {} and val_loss {:.3f}'.format(counter, val_loss))
            if counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break  # Early stopping triggered

    # # Now we're going to wrap the model with a decorator that adds temperature scaling
    # model_t = ModelWithTemperature(best_model)
    # # Tune the model temperature, and save the results
    # model_t.set_temperature(best_val_loader)

    model_t = best_model

    # ####  TEST LOOP  #### #
    _, _, test_metrics = run_epoch(
        model=model_t,
        loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        epoch=0,
        n_epochs=n_epochs,
        metric_type=0,
        train=False
    )

    logging.debug('Fold Test Acc: {:.3f} Prec: {:.3f} Rec: {:.3f} F1: {:.3f} CE: {:.3f} CM: {}'.format(
            test_metrics['accuracy'], test_metrics['precision'], test_metrics['recall'],
            test_metrics['f1'], test_metrics['ce'], test_metrics['cm']
        ))

    return model_t, min_loss, test_metrics, best_epoch
