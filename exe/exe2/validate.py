import numpy as np

def validate(y_pred_batch, y_true_batch):
    # Check if y_pred_batch has values greater than 1
    if np.any(y_pred_batch > 1):
        print("Warning: y_pred_batch contains values greater than 1")
        # y_pred_batch = np.clip(y_pred_batch, 0, 1)  # Clip values to [0, 1] range
    print(y_pred_batch.shape)
    size = len(y_pred_batch)
    dice, iou, sensitivity, specificity, precision = 0, 0, 0, 0, 0
    for y_pred, y_true in zip(y_pred_batch, y_true_batch):
        y_pred = (y_pred > 0.5).astype(bool).flatten()
        y_true = (y_true > 0.5).astype(bool).flatten()
        dice += Dice(y_pred, y_true)
        iou += IoU(y_pred, y_true)
        sensitivity += Sensitivity(y_pred, y_true)
        specificity += Specificity(y_pred, y_true)
        precision += Precision(y_pred, y_true)
    return dice/size, iou/size, sensitivity/size, specificity/size, precision/size

def Dice(y_pred, y_true, smooth=1e-6):
    intersection = np.sum(y_pred & y_true)
    sum_pred = np.sum(y_pred)
    sum_true = np.sum(y_true)
    return (2. * intersection) / (sum_pred + sum_true + smooth)

def IoU(y_pred, y_true, smooth=1e-6):
    intersection = np.sum(y_pred & y_true)
    union = np.sum(y_pred) + np.sum(y_true) - intersection
    return (intersection) / (union + smooth)

def Sensitivity(y_pred, y_true, smooth=1e-6):
    TP = (y_pred & y_true).sum()
    FN = ((~y_pred) & y_true).sum()
    return (TP) / (TP + FN + smooth)

def Specificity(y_pred, y_true, smooth=1e-6):
    TN = ((~y_pred) & (~y_true)).sum()
    FP = (y_pred & (~y_true)).sum()
    return (TN) / (TN + FP + smooth)

def Precision(y_pred, y_true, smooth=1e-6):
    intersection = np.sum(y_pred & y_true)
    sum_true = np.sum(y_true)
    return intersection / (sum_true + smooth)