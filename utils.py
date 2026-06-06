# Utility functions: device detection, training epoch runner, and weighted quantile.
import torch
import numpy as np
import logging

from metrics import compute_metrics, compute_mean_metrics


def get_device():
    """Detect the best available compute device: MPS (Apple Silicon), CUDA (NVIDIA GPU), or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, epoch=0, n_epochs=0,
              metric_type=0, train=True):
    """
    Execute one epoch of training or evaluation.

    Parameters:
    -----------
    model : nn.Module
        The model to train or evaluate.
    loader : DataLoader
        Batch iterator over the dataset.
    criterion : callable
        Loss function (expects output, target, weight).
    optimizer : torch.optim.Optimizer
        Optimizer used during training.
    epoch : int
        Current epoch number (for logging).
    n_epochs : int
        Total number of epochs (for logging).
    metric_type : int
        0 for classification metrics, 1 for regression metrics.
    train : bool
        If True, perform a forward + backward pass; otherwise evaluate without gradients.

    Returns:
    --------
    tuple: (model, avg_loss_per_sample, mean_metrics_dict)
    """
    device = get_device()

    metrics = None

    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0
    metrics_list = []
    for batch, (input, target, weight) in enumerate(loader):
        # Transfer Data to GPU if available
        input, target, weight = input.to(device), target.to(device), weight.to(device)
        if train:
            # Setting our stored gradients equal to zero
            optimizer.zero_grad()

            # Forward Pass
            output = model(input)
            # Get the Loss
            # loss = criterion(output, target)  # nn.BCELoss()
            loss = criterion(output, target, weight)  # CustomLoss()
            # loss = criterion(pred, torch.squeeze(y))  # nn.CrossEntropyLoss()

            # Computes the gradient of the given tensor w.r.t. the weights/bias
            loss.backward()
            # Update weight
            optimizer.step()
            # Update train_loss

        else:
            with torch.no_grad():
                output = model(input)
                # loss = criterion(output, target)  # nn.BCELoss()
                loss = criterion(output, target, weight)  # CustomLoss()

        epoch_loss += loss.item()

        # Accounting
        # _, predictions = torch.topk(output, 1)
        # error = 1 - torch.eq(predictions, target).float().mean()
        # accuracy = 1 - weighted_binary_accuracy(output, target, weight).item()
        # accuracy = 1 - binary_accuracy(output, target).item()

        # Only compute metrics when batch size > 1 (avoids issues with BatchNorm)
        if len(target) > 1:
            metrics = compute_metrics(weight.detach().cpu().numpy(),
                                      output.detach().cpu().numpy(),
                                      target.detach().cpu().numpy(),
                                      metric_type)
            metrics_list.append(metrics)

        if train:
            logging.debug('Train: [Epoch {:04d}/{:04d}] [Batch {:04d}/{:04d}] Loss: {:.4f}'.format(
                epoch, n_epochs, batch + 1, len(loader), loss
            ))
        else:
            logging.debug('Eval : [Epoch ----/----] [Batch {:04d}/{:04d}] Loss: {:.4f}'.format(
                batch + 1, len(loader), loss
            ))

    return model, epoch_loss / len(loader), \
        compute_mean_metrics(metrics_list, metric_type=metric_type)


def weighted_quantile(values, quantile, sample_weight):
    """
    Compute the weighted quantile of an array.

    Parameters:
    -----------
    values : array-like
        Values to compute the quantile over.
    quantile : float
        Target quantile level in [0, 1].
    sample_weight : array-like
        Non-negative weights associated with each value.

    Returns:
    --------
    float : the cutoff value at the requested weighted quantile.
    """
    values = np.array(values).flatten()
    sample_weight = sample_weight.flatten()

    # Sort values and weights together
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    # Cumulative sum of weights; find the index where it first exceeds the target quantile
    weighted_quantiles = np.cumsum(sample_weight)
    index = np.argmax(weighted_quantiles >= min(quantile, 1))
    cutoff = values[index]

    return cutoff
