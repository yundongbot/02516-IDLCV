import numpy as np

def validate(y_pred, y_true):
    y_pred = (y_pred > 0.5).astype(np.uint8).flatten()
    y_true = (y_true > 0.5).astype(np.uint8).flatten()

    dice = Dice(y_pred, y_true)
    iou = IoU(y_pred, y_true)
    sensitivity = Sensitivity(y_pred, y_true)
    specificity = Specificity(y_pred, y_true)
    precision = Precision(y_pred, y_true)

    print(f"Dice: {dice:.4f},\n IoU: {iou:.4f},\n Sensitivity: {sensitivity:.4f},\n Specificity: {specificity:.4f},\n Precision: {precision:.4f}")
    return dice, iou, sensitivity, specificity, precision

def Dice(y_pred, y_true, smooth=1e-6):
    intersection = np.sum(y_pred_flat * y_true_flat)
    sum_pred = np.sum(y_pred_flat)
    sum_true = np.sum(y_true_flat)
    return (2. * intersection + smooth) / (sum_pred + sum_true + smooth)

def IoU(y_pred, y_true, smooth=1e-6):
    intersection = np.sum(y_pred_flat * y_true_flat)
    union = np.sum(y_pred_flat) + np.sum(y_true_flat) - intersection
    return (intersection + smooth) / (union + smooth)

def Sensitivity(y_pred, y_true, smooth=1e-6):
    TP = np.sum(y_pred_flat * y_true_flat)
    FN = np.sum(y_true_flat) - TP
    return (TP + smooth) / (TP + FN + smooth)

def Specificity(y_pred, y_true, smooth=1e-6):
    TP = np.sum(y_pred_flat * y_true_flat)
    FP = np.sum(y_pred_flat) - TP
    TN = np.sum((1 - y_pred_flat) * (1 - y_true_flat))
    return (TN + smooth) / (TN + FP + smooth)

def Precision(y_pred, y_true, smooth=1e-6):
    TP = np.sum(y_pred_flat * y_true_flat)
    FP = np.sum(y_pred_flat) - TP
    TN = np.sum((1 - y_pred_flat) * (1 - y_true_flat))
    FN = np.sum(y_true_flat) - TP
    return (TP + smooth) / (TP + FP + smooth)
