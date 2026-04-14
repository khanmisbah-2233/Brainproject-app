import os
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_PATH,
    BATCH_SIZE,
    EPOCHS,
    VALIDATION_SPLIT,
    TEST_SPLIT,
    RANDOM_STATE,
    MODEL_PATH,
    LIMIT_PATIENTS,
    ARRAY_DTYPE,
    USE_TUMOR_SLICES_ONLY
)
from src.utils import create_directories, set_seed
from src.data_loader import get_patient_dirs, load_patient
from src.preprocess import extract_patient_slices
from src.model import build_unet
from src.train import train_model
from src.predict import load_trained_model, predict_mask, prediction_to_onehot
from src.evaluate import evaluate_dataset, evaluate_sample
from src.visualization import save_prediction_figure, save_training_plot
from src.report import save_text_report, save_metrics_csv


def load_split_dataset(patient_dirs, split_name="Split"):
    X = []
    Y = []

    print(f"\nLoading {split_name} data from {len(patient_dirs)} patients...")

    for idx, patient_dir in enumerate(patient_dirs):
        try:
            flair, t1, t1ce, t2, seg = load_patient(patient_dir)

            x_patient, y_patient = extract_patient_slices(
                flair,
                t1,
                t1ce,
                t2,
                seg,
                tumor_only=USE_TUMOR_SLICES_ONLY
            )

            X.extend(x_patient)
            Y.extend(y_patient)

            print(
                f"[{split_name} {idx + 1}/{len(patient_dirs)}] "
                f"{os.path.basename(patient_dir)} -> {len(x_patient)} slices"
            )

        except Exception as e:
            print(f"Skipping {patient_dir}: {e}")

    if len(X) == 0 or len(Y) == 0:
        return np.array([]), np.array([])

    X = np.array(X, dtype=ARRAY_DTYPE)
    Y = np.array(Y, dtype=ARRAY_DTYPE)

    print(f"{split_name} shape: X={X.shape}, Y={Y.shape}")
    return X, Y


def split_patients(dataset_path):
    patient_dirs = get_patient_dirs(dataset_path)

    if LIMIT_PATIENTS is not None:
        patient_dirs = patient_dirs[:LIMIT_PATIENTS]

    print(f"Total patients considered: {len(patient_dirs)}")

    train_val_patients, test_patients = train_test_split(
        patient_dirs,
        test_size=TEST_SPLIT,
        random_state=RANDOM_STATE
    )

    val_ratio = VALIDATION_SPLIT / (1 - TEST_SPLIT)

    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=val_ratio,
        random_state=RANDOM_STATE
    )

    print(f"Train patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")
    print(f"Test patients: {len(test_patients)}")

    return train_patients, val_patients, test_patients


def select_best_visual_sample(Y_test, y_pred):
    """
    Pick a sample where:
    - GT has tumor
    - Prediction has tumor
    - overlap is best (highest mean_tumor_dice)

    Fallback:
    - first sample with GT tumor
    - else index 0
    """
    best_index = None
    best_score = -1.0
    fallback_gt_index = None

    for i in range(len(Y_test)):
        gt = Y_test[i].astype(np.float32)
        pred = prediction_to_onehot(y_pred[i].astype(np.float32))

        gt_has_tumor = np.sum(gt[:, :, 1:]) > 0
        pred_has_tumor = np.sum(pred[:, :, 1:]) > 0

        if gt_has_tumor and fallback_gt_index is None:
            fallback_gt_index = i

        if gt_has_tumor and pred_has_tumor:
            sample_metrics = evaluate_sample(gt, pred, num_classes=4)
            score = sample_metrics["mean_tumor_dice"]

            if score > best_score:
                best_score = score
                best_index = i

    if best_index is not None:
        return best_index

    if fallback_gt_index is not None:
        return fallback_gt_index

    return 0


def main():
    create_directories()
    set_seed()

    print("Splitting dataset at patient level...")
    train_patients, val_patients, test_patients = split_patients(DATASET_PATH)

    X_train, Y_train = load_split_dataset(train_patients, split_name="Train")
    X_val, Y_val = load_split_dataset(val_patients, split_name="Validation")
    X_test, Y_test = load_split_dataset(test_patients, split_name="Test")

    if len(X_train) == 0 or len(Y_train) == 0:
        raise ValueError("Training dataset is empty after preprocessing.")
    if len(X_val) == 0 or len(Y_val) == 0:
        raise ValueError("Validation dataset is empty after preprocessing.")
    if len(X_test) == 0 or len(Y_test) == 0:
        raise ValueError("Test dataset is empty after preprocessing.")

    print(f"\nFinal split summary:")
    print(f"Train: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Validation: X={X_val.shape}, Y={Y_val.shape}")
    print(f"Test: X={X_test.shape}, Y={Y_test.shape}")

    print("\nBuilding model...")
    model = build_unet()
    model.summary()

    print("\nTraining model...")
    model, history = train_model(
        model,
        X_train,
        Y_train,
        X_val,
        Y_val,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    print(f"\nBest model saved at: {MODEL_PATH}")

    print("Saving training plot...")
    training_plot_path = save_training_plot(history)
    print(f"Training plot saved: {training_plot_path}")

    print("\nLoading best saved model for evaluation...")
    best_model = load_trained_model(MODEL_PATH)

    print("Running predictions on test set...")
    y_pred = np.array(
        [predict_mask(best_model, x) for x in X_test],
        dtype=np.float32
    )

    print("Evaluating model...")
    metrics = evaluate_dataset(Y_test.astype(np.float32), y_pred, num_classes=4)

    print("\nKey Metrics:")
    print(f"pixel_accuracy_all: {metrics['pixel_accuracy_all']:.6f}")
    print(f"tumor_only_accuracy: {metrics['tumor_only_accuracy']:.6f}")
    print(f"mean_tumor_dice: {metrics['mean_tumor_dice']:.6f}")
    print(f"mean_tumor_iou: {metrics['mean_tumor_iou']:.6f}")

    print("\nClass-wise Dice / IoU:")
    for c in range(4):
        print(f"class_{c}_dice: {metrics[f'class_{c}_dice']:.6f}")
        print(f"class_{c}_iou: {metrics[f'class_{c}_iou']:.6f}")

    print("\nSaving sample prediction figure...")
    sample_index = select_best_visual_sample(Y_test, y_pred)

    sample_pred_onehot = prediction_to_onehot(y_pred[sample_index].astype(np.float32))

    sample_path = save_prediction_figure(
        X_test[sample_index].astype(np.float32),
        Y_test[sample_index].astype(np.float32),
        sample_pred_onehot,
        file_name="test_sample_prediction.png"
    )
    print(f"Prediction figure saved: {sample_path}")

    extra_info = {
        "Dataset Path": DATASET_PATH,
        "Limited Patients": LIMIT_PATIENTS,
        "Train Patients": len(train_patients),
        "Validation Patients": len(val_patients),
        "Test Patients": len(test_patients),
        "Train Samples": len(X_train),
        "Validation Samples": len(X_val),
        "Test Samples": len(X_test),
        "Epochs": EPOCHS,
        "Batch Size": BATCH_SIZE,
        "Array DType": ARRAY_DTYPE,
        "Split Method": "Patient-level split"
    }

    print("Saving reports...")
    save_text_report(metrics, extra_info=extra_info)
    save_metrics_csv(metrics)

    print("\nProject pipeline completed successfully.")


if __name__ == "__main__":
    main()