import os
import tempfile
import nibabel as nib
import numpy as np


def get_patient_dirs(dataset_path):
    if not os.path.exists(dataset_path):
        return []

    patient_dirs = []
    for item in os.listdir(dataset_path):
        full_path = os.path.join(dataset_path, item)
        if os.path.isdir(full_path):
            patient_dirs.append(full_path)

    patient_dirs.sort()
    return patient_dirs


def _find_modality_file(patient_dir, keyword):
    files = os.listdir(patient_dir)
    for f in files:
        lower_f = f.lower()
        if keyword in lower_f and (lower_f.endswith(".nii") or lower_f.endswith(".nii.gz")):
            return os.path.join(patient_dir, f)
    raise FileNotFoundError(f"Could not find file containing '{keyword}' in {patient_dir}")


def load_nifti_file(file_path):
    nii = nib.load(file_path)
    return nii.get_fdata().astype(np.float32)


def load_patient(patient_dir):
    flair_path = _find_modality_file(patient_dir, "flair")
    t1_path = _find_modality_file(patient_dir, "_t1.")
    t1ce_path = _find_modality_file(patient_dir, "t1ce")
    t2_path = _find_modality_file(patient_dir, "_t2.")
    seg_path = _find_modality_file(patient_dir, "seg")

    flair = load_nifti_file(flair_path)
    t1 = load_nifti_file(t1_path)
    t1ce = load_nifti_file(t1ce_path)
    t2 = load_nifti_file(t2_path)
    seg = load_nifti_file(seg_path)

    return flair, t1, t1ce, t2, seg


def load_uploaded_nifti(uploaded_file):
    suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    try:
        volume = nib.load(temp_path).get_fdata().astype(np.float32)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return volume


def load_uploaded_modalities(flair_file, t1_file, t1ce_file, t2_file, seg_file=None):
    flair = load_uploaded_nifti(flair_file)
    t1 = load_uploaded_nifti(t1_file)
    t1ce = load_uploaded_nifti(t1ce_file)
    t2 = load_uploaded_nifti(t2_file)

    seg = None
    if seg_file is not None:
        seg = load_uploaded_nifti(seg_file)

    return flair, t1, t1ce, t2, seg