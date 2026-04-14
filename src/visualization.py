import os
import matplotlib.pyplot as plt
import numpy as np

from src.config import PREDICTIONS_DIR, PLOTS_DIR


def plot_sample_prediction(image, mask, prediction, save_path=None):
    """
    image: (H, W, 4)
    mask: (H, W, 4)
    prediction: (H, W, 4)
    """
    gt_class = np.argmax(mask, axis=-1)
    pred_class = np.argmax(prediction, axis=-1)

    gt_display = np.ma.masked_where(gt_class == 0, gt_class)
    pred_display = np.ma.masked_where(pred_class == 0, pred_class)

    plt.figure(figsize=(16, 4), facecolor="white")

    plt.subplot(1, 4, 1)
    plt.imshow(image[:, :, 0], cmap="gray")
    plt.title("MRI (FLAIR)")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(image[:, :, 0], cmap="gray")
    plt.imshow(gt_display, cmap="jet", alpha=0.5, vmin=1, vmax=3)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(image[:, :, 0], cmap="gray")
    plt.imshow(pred_display, cmap="jet", alpha=0.5, vmin=1, vmax=3)
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(image[:, :, 0], cmap="gray")
    plt.imshow(pred_display, cmap="jet", alpha=0.5, vmin=1, vmax=3)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def save_prediction_figure(image, mask, prediction, file_name="sample_prediction.png"):
    save_path = os.path.join(PREDICTIONS_DIR, file_name)
    plot_sample_prediction(image, mask, prediction, save_path=save_path)
    return save_path


def plot_training_history(history, save_path=None):
    history_dict = history.history

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.plot(history_dict.get("loss", []), label="Train Loss")
    plt.plot(history_dict.get("val_loss", []), label="Val Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(history_dict.get("dice_coef", []), label="Train Dice")
    plt.plot(history_dict.get("val_dice_coef", []), label="Val Dice")
    plt.title("Dice Coefficient")
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(history_dict.get("iou_metric", []), label="Train IoU")
    plt.plot(history_dict.get("val_iou_metric", []), label="Val IoU")
    plt.title("IoU")
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(history_dict.get("accuracy", []), label="Train Accuracy")
    plt.plot(history_dict.get("val_accuracy", []), label="Val Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def save_training_plot(history, file_name="training_history.png"):
    save_path = os.path.join(PLOTS_DIR, file_name)
    plot_training_history(history, save_path=save_path)
    return save_path