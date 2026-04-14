import numpy as np

from src.config import DATASET_PATH
from src.data_loader import get_patient_dirs, load_patient
from src.preprocess import extract_patient_slices
from src.predict import load_trained_model, predict_mask, prediction_to_onehot
from src.visualization import save_prediction_figure
from src.evaluate import evaluate_sample


MODEL_PATH = r"outputs\models\best_model.keras"


def select_best_visual_sample(X_patient, Y_patient, model):
    """
    Pick a sample where:
    - GT has tumor
    - Prediction also has tumor
    - overlap is best (highest mean_tumor_dice)

    Fallback:
    - first sample with GT tumor
    - else index 0
    """
    best_index = None
    best_score = -1.0
    fallback_gt_index = None

    for i in range(len(Y_patient)):
        gt = Y_patient[i].astype(np.float32)
        image = X_patient[i].astype(np.float32)

        gt_has_tumor = np.sum(gt[:, :, 1:]) > 0

        if gt_has_tumor and fallback_gt_index is None:
            fallback_gt_index = i

        prediction = predict_mask(model, image)
        pred_onehot = prediction_to_onehot(prediction)
        pred_has_tumor = np.sum(pred_onehot[:, :, 1:]) > 0

        if gt_has_tumor and pred_has_tumor:
            sample_metrics = evaluate_sample(gt, pred_onehot, num_classes=4)
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
    patient_dirs = get_patient_dirs(DATASET_PATH)
    model = load_trained_model(MODEL_PATH)

    for patient_dir in patient_dirs:
        flair, t1, t1ce, t2, seg = load_patient(patient_dir)

        X_patient, Y_patient = extract_patient_slices(
            flair, t1, t1ce, t2, seg, tumor_only=True
        )

        if len(X_patient) == 0:
            continue

        sample_index = select_best_visual_sample(X_patient, Y_patient, model)

        image = X_patient[sample_index].astype(np.float32)
        mask = Y_patient[sample_index].astype(np.float32)
        prediction = predict_mask(model, image)
        pred_onehot = prediction_to_onehot(prediction)

        save_path = save_prediction_figure(
            image,
            mask,
            pred_onehot,
            file_name="test_sample_prediction.png"
        )

        print(f"Saved new prediction image: {save_path}")
        print(f"Selected sample index: {sample_index}")
        break


if __name__ == "__main__":
    main()