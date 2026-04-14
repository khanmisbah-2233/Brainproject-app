import cv2
import numpy as np
from tensorflow.keras.models import load_model

from src.train import dice_coef, iou_metric, combined_loss


def load_trained_model(model_path):
    return load_model(
        model_path,
        custom_objects={
            "dice_coef": dice_coef,
            "iou_metric": iou_metric,
            "combined_loss": combined_loss
        }
    )


def predict_mask(model, image):
    """
    Predict segmentation mask for one image.
    image shape: (H, W, 4) or (1, H, W, 4)
    returns: prediction shape (H, W, 4) for 4-class segmentation
    """
    image = np.array(image, dtype=np.float32)

    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    if image.ndim != 4:
        raise ValueError(f"Expected 4D input, got shape {image.shape}")

    pred = model.predict(image, verbose=0)
    return pred[0]


def remove_small_regions(onehot_mask, min_area=30):
    cleaned = np.zeros_like(onehot_mask, dtype=np.float32)

    for c in range(1, onehot_mask.shape[-1]):  # skip background
        channel = onehot_mask[:, :, c].astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(channel, connectivity=8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[:, :, c][labels == i] = 1.0

    background = 1.0 - np.clip(np.sum(cleaned[:, :, 1:], axis=-1), 0, 1)
    cleaned[:, :, 0] = background
    return cleaned


def prediction_to_onehot(prediction):
    pred_class = np.argmax(prediction, axis=-1)
    onehot = np.zeros_like(prediction, dtype=np.float32)

    for c in range(prediction.shape[-1]):
        onehot[:, :, c] = (pred_class == c).astype(np.float32)

    onehot = remove_small_regions(onehot, min_area=30)
    return onehot


def prediction_to_classmap(prediction):
    return np.argmax(prediction, axis=-1)