import os
import nibabel as nib

VALID_EXTENSIONS = (".nii", ".nii.gz")


def _find_modality_file(files, keywords):
    """
    Return the first file that matches all keywords.
    """
    for file_name in files:
        lower_name = file_name.lower()
        if lower_name.endswith(VALID_EXTENSIONS) and all(k in lower_name for k in keywords):
            return file_name
    return None


def get_patient_dirs(dataset_path):
    """
    Return all valid patient directories inside the dataset path.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    patient_dirs = []
    for item in sorted(os.listdir(dataset_path)):
        full_path = os.path.join(dataset_path, item)
        if os.path.isdir(full_path):
            patient_dirs.append(full_path)

    if not patient_dirs:
        raise ValueError(f"No patient folders found in: {dataset_path}")

    return patient_dirs


def load_nifti(file_path):
    """
    Load a NIfTI file and return the image data as float32.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"NIfTI file not found: {file_path}")

    return nib.load(file_path).get_fdata().astype("float32")


def load_patient(patient_path):
    """
    Load all required BraTS modalities for one patient:
    FLAIR, T1, T1CE, T2, SEG
    """
    if not os.path.exists(patient_path):
        raise FileNotFoundError(f"Patient path not found: {patient_path}")

    files = os.listdir(patient_path)

    flair_file = _find_modality_file(files, ["flair"])
    t1_file = _find_modality_file(files, ["t1"]) and not _find_modality_file(files, ["t1ce"])
    t1ce_file = _find_modality_file(files, ["t1ce"])
    t2_file = _find_modality_file(files, ["t2"])
    seg_file = _find_modality_file(files, ["seg"])

    # Fix T1 detection properly
    t1_file = None
    for file_name in files:
        lower_name = file_name.lower()
        if lower_name.endswith(VALID_EXTENSIONS) and "t1" in lower_name and "t1ce" not in lower_name:
            t1_file = file_name
            break

    required = {
        "FLAIR": flair_file,
        "T1": t1_file,
        "T1CE": t1ce_file,
        "T2": t2_file,
        "SEG": seg_file,
    }

    missing = [name for name, file_name in required.items() if file_name is None]
    if missing:
        raise ValueError(
            f"Missing required modality files in '{patient_path}'. Missing: {', '.join(missing)}"
        )

    flair = load_nifti(os.path.join(patient_path, flair_file))
    t1 = load_nifti(os.path.join(patient_path, t1_file))
    t1ce = load_nifti(os.path.join(patient_path, t1ce_file))
    t2 = load_nifti(os.path.join(patient_path, t2_file))
    seg = load_nifti(os.path.join(patient_path, seg_file))

    return flair, t1, t1ce, t2, seg