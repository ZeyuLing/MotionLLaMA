import numpy as np
from numpy import count_nonzero
from sklearn.metrics._classification import _check_targets, _weighted_sum
from sklearn.utils import check_consistent_length


def accuracy_score(y_true, y_pred, *, shift=True, normalize=True, sample_weight=None, ignore_idx=None):
    """ modified from scipy
        Accuracy classification score.
        In multilabel classification, this function computes subset accuracy:
        the set of labels predicted for a sample must *exactly* match the
        corresponding set of labels in y_true.

    :param shift: The output of Transformer is the prediction of next pos.
                    Ground truth should be shifted left,
                    prediction should be shifted right in this case
    :param y_true:  1d array-like, or label indicator array / sparse matrix
                    Ground truth (correct) labels.
    :param y_pred: 1d array-like, or label indicator array / sparse matrix
                    Predicted labels, as returned by a classifier.
    :param normalize: bool, default=True
                    If ``False``, return the number of correctly classified samples.
                    Otherwise, return the fraction of correctly classified samples
    :param sample_weight: array-like of shape (n_samples,), default=None
    :param ignore_idx:
    :return: score : float
        If ``normalize == True``, return the fraction of correctly
        classified samples (float), else returns the number of correctly
        classified samples (int).

        The best performance is 1 with ``normalize == True`` and the number
        of samples with ``normalize == False``.

    """

    # Compute accuracy for each possible representation
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    if shift:
        y_pred = y_pred[..., :-1, ]
        y_true = y_true[..., 1:]

    check_consistent_length(y_true, y_pred, sample_weight)
    assert len(y_true) > 0, f'{y_true.shape}, {y_pred.shape}'
    if ignore_idx is not None:
        mask = y_true != ignore_idx
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if sample_weight is not None:
            sample_weight = sample_weight[mask]
    if y_type.startswith("multilabel"):
        differing_labels = count_nonzero(y_true - y_pred, axis=1)
        score = differing_labels == 0
    else:
        score = y_true == y_pred

    return _weighted_sum(score, sample_weight, normalize)


if __name__ == '__main__':
    predictions = np.asarray([8, 3, 5, 4, 2, 0, 0, 0, 0, 0, 0, 0])
    labels = np.asarray([1, 8, 3, 3, 5, 2, 2, 0, 0, 0, 0, 0])
    # 3
    print(accuracy_score(labels, predictions, ))
