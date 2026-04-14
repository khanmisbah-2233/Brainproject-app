import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.config import DATASET_PATH, MODEL_PATH, NUM_CLASSES
from src.data_loader import get_patient_dirs, load_patient
from src.preprocess import extract_patient_slices
from src.predict import load_trained_model, predict_mask, prediction_to_onehot
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


st.title("Brain Tumor Segmentation from MRI Scans")
st.write("Multi-modal MRI segmentation using a trained U-Net model.")

model = get_model()

if model is None:
    st.error("Trained model not found. Pehle `python main.py` chalao.")
    st.stop()

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

st.sidebar.header("Controls")
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

gt_class = np.argmax(mask, axis=-1)
pred_class = np.argmax(pred_onehot, axis=-1)

metrics = evaluate_sample(mask, pred_onehot, num_classes=NUM_CLASSES)

st.subheader("Main Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Mean Tumor Dice", f"{metrics['mean_tumor_dice']:.4f}")
m2.metric("Mean Tumor IoU", f"{metrics['mean_tumor_iou']:.4f}")
m3.metric("Tumor Accuracy", f"{metrics['tumor_only_accuracy']:.4f}")
m4.metric("Pixel Accuracy", f"{metrics['pixel_accuracy_all']:.4f}")

st.subheader("Images")
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

st.subheader("Quick Interpretation")
if metrics["mean_tumor_dice"] >= 0.80:
    st.success("Tumor segmentation quality is strong on this slice.")
elif metrics["mean_tumor_dice"] >= 0.70:
    st.info("Tumor segmentation quality is reasonable on this slice.")
else:
    st.warning("Tumor segmentation quality is weak on this slice.")

with st.expander("Prediction Details"):
    st.write(f"Patient: {selected_patient}")
    st.write(f"Slice Index: {slice_idx}")
    st.write(f"Input Shape: {image.shape}")
    st.write(f"Mask Shape: {mask.shape}")
    st.write(f"Prediction Shape: {prediction.shape}")
    st.write(f"Model Path: {MODEL_PATH}")