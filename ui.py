import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.config import DATASET_PATH, MODEL_PATH, NUM_CLASSES
from src.data_loader import (
    get_patient_dirs,
    load_patient,
    load_uploaded_modalities
)
from src.preprocess import (
    extract_patient_slices,
    extract_uploaded_slices
)
from src.predict import (
    load_trained_model,
    predict_mask,
    predict_volume,
    prediction_to_onehot
)
from src.evaluate import evaluate_sample


st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")


@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_trained_model(MODEL_PATH)


@st.cache_data
def get_all_patients(dataset_path):
    return get_patient_dirs(dataset_path)


@st.cache_data
def load_patient_slices(patient_path):
    flair, t1, t1ce, t2, seg = load_patient(patient_path)
    X_patient, Y_patient = extract_patient_slices(
        flair, t1, t1ce, t2, seg, tumor_only=True
    )
    return X_patient, Y_patient


def create_overlay_figure(base_image, class_map, title="Overlay"):
    display_map = np.ma.masked_where(class_map == 0, class_map)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(base_image, cmap="gray")
    ax.imshow(display_map, cmap="jet", alpha=0.5, vmin=1, vmax=3)
    ax.set_title(title)
    ax.axis("off")
    return fig


def create_side_by_side_overlay(base_image, gt_class, pred_class):
    gt_display = np.ma.masked_where(gt_class == 0, gt_class)
    pred_display = np.ma.masked_where(pred_class == 0, pred_class)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(base_image, cmap="gray")
    axes[0].imshow(gt_display, cmap="jet", alpha=0.5, vmin=1, vmax=3)
    axes[0].set_title("Ground Truth Overlay")
    axes[0].axis("off")

    axes[1].imshow(base_image, cmap="gray")
    axes[1].imshow(pred_display, cmap="jet", alpha=0.5, vmin=1, vmax=3)
    axes[1].set_title("Prediction Overlay")
    axes[1].axis("off")

    plt.tight_layout()
    return fig


def show_metrics(metrics):
    st.subheader("Main Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean Tumor Dice", f"{metrics['mean_tumor_dice']:.4f}")
    m2.metric("Mean Tumor IoU", f"{metrics['mean_tumor_iou']:.4f}")
    m3.metric("Tumor Accuracy", f"{metrics['tumor_only_accuracy']:.4f}")
    m4.metric("Pixel Accuracy", f"{metrics['pixel_accuracy_all']:.4f}")

    st.subheader("Class-wise Metrics")
    class_cols = st.columns(NUM_CLASSES)

    class_names = {
        0: "Background",
        1: "Necrotic Core",
        2: "Edema",
        3: "Enhancing Tumor"
    }

    for c in range(NUM_CLASSES):
        with class_cols[c]:
            st.write(class_names.get(c, f"Class {c}"))
            st.write(f"Dice: {metrics.get(f'class_{c}_dice', 0):.4f}")
            st.write(f"IoU: {metrics.get(f'class_{c}_iou', 0):.4f}")


def show_prediction_section(image, mask, pred_onehot, metrics=None):
    gt_class = None
    if mask is not None:
        gt_class = np.argmax(mask, axis=-1)

    pred_class = np.argmax(pred_onehot, axis=-1)

    st.subheader("Images")
    if mask is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**MRI Slice (FLAIR)**")
            st.image(image[:, :, 0], clamp=True)

        with col2:
            st.markdown("**Ground Truth Overlay**")
            gt_fig = create_overlay_figure(image[:, :, 0], gt_class, title="Ground Truth")
            st.pyplot(gt_fig)

        with col3:
            st.markdown("**Prediction Overlay**")
            pred_fig = create_overlay_figure(image[:, :, 0], pred_class, title="Prediction")
            st.pyplot(pred_fig)

        st.subheader("Overlay Comparison")
        overlay_compare_fig = create_side_by_side_overlay(image[:, :, 0], gt_class, pred_class)
        st.pyplot(overlay_compare_fig)

    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**MRI Slice (FLAIR)**")
            st.image(image[:, :, 0], clamp=True)

        with col2:
            st.markdown("**Prediction Overlay**")
            pred_fig = create_overlay_figure(image[:, :, 0], pred_class, title="Prediction")
            st.pyplot(pred_fig)

    if metrics is not None:
        show_metrics(metrics)

        st.subheader("Quick Interpretation")
        if metrics["mean_tumor_dice"] >= 0.80:
            st.success("Tumor segmentation quality is strong on this slice.")
        elif metrics["mean_tumor_dice"] >= 0.70:
            st.info("Tumor segmentation quality is reasonable on this slice.")
        else:
            st.warning("Tumor segmentation quality is weak on this slice.")


st.title("Brain Tumor Segmentation from MRI Scans")
st.write("Multi-modal MRI segmentation using a trained U-Net model.")

model = get_model()

if model is None:
    st.error("Trained model not found. Pehle `python main.py` chalao.")
    st.stop()

mode = st.sidebar.radio(
    "Select Mode",
    ["Demo Mode", "Upload MRI Mode"]
)

if mode == "Demo Mode":
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset path not found: {DATASET_PATH}")
        st.stop()

    try:
        patient_dirs = get_all_patients(DATASET_PATH)
    except Exception as e:
        st.error(f"Failed to read dataset: {e}")
        st.stop()

    patient_names = [os.path.basename(p) for p in patient_dirs]

    if not patient_names:
        st.error("No patient folders found in dataset path.")
        st.stop()

    selected_patient = st.sidebar.selectbox("Select Patient", patient_names)
    patient_path = os.path.join(DATASET_PATH, selected_patient)

    try:
        X_patient, Y_patient = load_patient_slices(patient_path)
    except Exception as e:
        st.error(f"Failed to load patient data: {e}")
        st.stop()

    if len(X_patient) == 0:
        st.warning("No tumor-containing slices found for this patient.")
        st.stop()

    slice_idx = st.sidebar.slider("Select Slice", 0, len(X_patient) - 1, 0)

    image = X_patient[slice_idx].astype(np.float32)
    mask = Y_patient[slice_idx].astype(np.float32)

    prediction = predict_mask(model, image)
    pred_onehot = prediction_to_onehot(prediction)
    metrics = evaluate_sample(mask, pred_onehot, num_classes=NUM_CLASSES)

    show_prediction_section(image, mask, pred_onehot, metrics=metrics)

    with st.expander("Prediction Details"):
        st.write(f"Patient: {selected_patient}")
        st.write(f"Slice Index: {slice_idx}")
        st.write(f"Input Shape: {image.shape}")
        st.write(f"Mask Shape: {mask.shape}")
        st.write(f"Prediction Shape: {prediction.shape}")
        st.write(f"Model Path: {MODEL_PATH}")

else:
    st.sidebar.subheader("Upload MRI Files")

    flair_file = st.sidebar.file_uploader("Upload FLAIR (.nii/.nii.gz)", type=["nii", "gz"])
    t1_file = st.sidebar.file_uploader("Upload T1 (.nii/.nii.gz)", type=["nii", "gz"])
    t1ce_file = st.sidebar.file_uploader("Upload T1ce (.nii/.nii.gz)", type=["nii", "gz"])
    t2_file = st.sidebar.file_uploader("Upload T2 (.nii/.nii.gz)", type=["nii", "gz"])
    seg_file = st.sidebar.file_uploader("Upload SEG (optional)", type=["nii", "gz"])

    if flair_file and t1_file and t1ce_file and t2_file:
        try:
            flair, t1, t1ce, t2, seg = load_uploaded_modalities(
                flair_file,
                t1_file,
                t1ce_file,
                t2_file,
                seg_file
            )

            X_uploaded, Y_uploaded = extract_uploaded_slices(
                flair,
                t1,
                t1ce,
                t2,
                seg
            )

            st.success("MRI files loaded successfully.")

            slice_idx = st.sidebar.slider("Select Uploaded Slice", 0, len(X_uploaded) - 1, len(X_uploaded) // 2)

            image = X_uploaded[slice_idx].astype(np.float32)
            prediction = predict_mask(model, image)
            pred_onehot = prediction_to_onehot(prediction)

            if Y_uploaded is not None:
                mask = Y_uploaded[slice_idx].astype(np.float32)
                metrics = evaluate_sample(mask, pred_onehot, num_classes=NUM_CLASSES)
                show_prediction_section(image, mask, pred_onehot, metrics=metrics)
            else:
                show_prediction_section(image, None, pred_onehot, metrics=None)
                st.info("No ground truth segmentation uploaded, so metrics are not available.")

            with st.expander("Uploaded Case Details"):
                st.write(f"Uploaded volume slices: {len(X_uploaded)}")
                st.write(f"Selected slice index: {slice_idx}")
                st.write(f"Input shape: {image.shape}")
                st.write(f"Model path: {MODEL_PATH}")

        except Exception as e:
            st.error(f"Failed to process uploaded MRI files: {e}")

    else:
        st.info("Upload FLAIR, T1, T1ce, and T2 MRI files to run prediction.")