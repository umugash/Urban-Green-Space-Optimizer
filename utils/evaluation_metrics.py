import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    y_true: Ground truth mask (binary 0/1)
    y_pred: Predicted mask (binary 0/1)
    """

    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # True Positives
    TP = np.sum((y_true == 1) & (y_pred == 1))

    # False Positives
    FP = np.sum((y_true == 0) & (y_pred == 1))

    # False Negatives
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # IoU
    iou = TP / (TP + FP + FN + 1e-6)

    # Dice Score
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)

    # Precision
    precision = TP / (TP + FP + 1e-6)

    # Recall
    recall = TP / (TP + FN + 1e-6)

    return {
        "IoU": iou,
        "Dice": dice,
        "Precision": precision,
        "Recall": recall
    }