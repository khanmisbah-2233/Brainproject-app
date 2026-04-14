import numpy as np


def _to_class_map(y):
    if y.ndim == 4:
        return np.argmax(y, axis=-1)
    if y.ndim == 3:
        return np.argmax(y, axis=-1)
    raise ValueError(f"Unsupported shape for class map conversion: {y.shape}")


def dice_score(y_true, y_pred, smooth=1e-6):
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)

    return (2.0 * intersection + smooth) / (union + smooth)


def iou_score(y_true, y_pred, smooth=1e-6):
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection

    return (intersection + smooth) / (union + smooth)


def pixel_accuracy_all(y_true, y_pred):
    true_cls = _to_class_map(y_true)
    pred_cls = _to_class_map(y_pred)

    correct = np.sum(true_cls == pred_cls)
    total = true_cls.size

    return correct / total if total > 0 else 0.0


def tumor_only_accuracy(y_true, y_pred):
    """
    Accuracy only on tumor pixels.
    Background class (0) is ignored.
    """
    true_cls = _to_class_map(y_true)
    pred_cls = _to_class_map(y_pred)

    tumor_mask = true_cls != 0
    total = np.sum(tumor_mask)

    if total == 0:
        return 0.0

    correct = np.sum(true_cls[tumor_mask] == pred_cls[tumor_mask])
    return correct / total


def classwise_dice(y_true, y_pred, num_classes=4):
    scores = {}

    for c in range(num_classes):
        yt = y_true[:, :, c]
        yp = y_pred[:, :, c]
        scores[f"class_{c}_dice"] = dice_score(yt, yp)

    return scores


def classwise_iou(y_true, y_pred, num_classes=4):
    scores = {}

    for c in range(num_classes):
        yt = y_true[:, :, c]
        yp = y_pred[:, :, c]
        scores[f"class_{c}_iou"] = iou_score(yt, yp)

    return scores


def evaluate_sample(y_true, y_pred, num_classes=4):
    """
    y_true: one-hot mask (H, W, C)
    y_pred: softmax prediction or one-hot mask (H, W, C)
    """
    pred_bin = np.zeros_like(y_pred, dtype=np.float32)
    pred_class = np.argmax(y_pred, axis=-1)

    for c in range(num_classes):
        pred_bin[:, :, c] = (pred_class == c).astype(np.float32)

    dice_scores = classwise_dice(y_true, pred_bin, num_classes)
    iou_scores = classwise_iou(y_true, pred_bin, num_classes)

    tumor_dice_values = [dice_scores[f"class_{c}_dice"] for c in range(1, num_classes)]
    tumor_iou_values = [iou_scores[f"class_{c}_iou"] for c in range(1, num_classes)]

    results = {
        "pixel_accuracy_all": pixel_accuracy_all(y_true, pred_bin),
        "tumor_only_accuracy": tumor_only_accuracy(y_true, pred_bin),
        "mean_tumor_dice": float(np.mean(tumor_dice_values)),
        "mean_tumor_iou": float(np.mean(tumor_iou_values)),
    }

    results.update(dice_scores)
    results.update(iou_scores)

    return results


def evaluate_dataset(y_true_all, y_pred_all, num_classes=4):
    metrics_list = []

    for i in range(len(y_true_all)):
        metrics = evaluate_sample(y_true_all[i], y_pred_all[i], num_classes)
        metrics_list.append(metrics)

    mean_metrics = {}
    for key in metrics_list[0].keys():
        mean_metrics[key] = float(np.mean([m[key] for m in metrics_list]))

    return mean_metrics