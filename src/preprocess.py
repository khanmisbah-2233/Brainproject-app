import cv2
import numpy as np

from src.config import IMAGE_SIZE, CLASS_LABELS, MIN_TUMOR_PIXELS


def normalize_image(img):
    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val - min_val < 1e-8:
        return np.zeros_like(img, dtype=np.float32)

    return (img - min_val) / (max_val - min_val)


def resize_slice(slice_img, size=IMAGE_SIZE, is_mask=False):
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(slice_img, (size, size), interpolation=interpolation)


def create_input(flair_slice, t1_slice, t1ce_slice, t2_slice):
    flair_slice = resize_slice(flair_slice)
    t1_slice = resize_slice(t1_slice)
    t1ce_slice = resize_slice(t1ce_slice)
    t2_slice = resize_slice(t2_slice)

    flair_slice = normalize_image(flair_slice)
    t1_slice = normalize_image(t1_slice)
    t1ce_slice = normalize_image(t1ce_slice)
    t2_slice = normalize_image(t2_slice)

    image = np.stack([flair_slice, t1_slice, t1ce_slice, t2_slice], axis=-1)
    return image.astype(np.float32)


def create_mask(seg_slice):
    seg_slice = resize_slice(seg_slice, is_mask=True)

    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 4), dtype=np.float32)
    mask[:, :, 0] = (seg_slice == CLASS_LABELS["background"]).astype(np.float32)
    mask[:, :, 1] = (seg_slice == CLASS_LABELS["necrotic_core"]).astype(np.float32)
    mask[:, :, 2] = (seg_slice == CLASS_LABELS["edema"]).astype(np.float32)
    mask[:, :, 3] = (seg_slice == CLASS_LABELS["enhancing_tumor"]).astype(np.float32)

    return mask


def has_enough_tumor(mask, min_tumor_pixels=MIN_TUMOR_PIXELS):
    tumor_pixels = np.sum(mask[:, :, 1:])
    return tumor_pixels >= min_tumor_pixels


def augment_sample(image, mask):
    if np.random.rand() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)

    if np.random.rand() < 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)

    return image.copy(), mask.copy()


def extract_patient_slices(flair, t1, t1ce, t2, seg, tumor_only=True):
    if not (flair.shape == t1.shape == t1ce.shape == t2.shape == seg.shape):
        raise ValueError("All modality volumes and segmentation mask must have the same shape.")

    num_slices = flair.shape[2]

    X_patient = []
    Y_patient = []

    for i in range(num_slices):
        flair_slice = flair[:, :, i]
        t1_slice = t1[:, :, i]
        t1ce_slice = t1ce[:, :, i]
        t2_slice = t2[:, :, i]
        seg_slice = seg[:, :, i]

        image = create_input(flair_slice, t1_slice, t1ce_slice, t2_slice)
        mask = create_mask(seg_slice)

        if tumor_only:
            if has_enough_tumor(mask):
                image, mask = augment_sample(image, mask)
                X_patient.append(image)
                Y_patient.append(mask)
        else:
            image, mask = augment_sample(image, mask)
            X_patient.append(image)
            Y_patient.append(mask)

    return X_patient, Y_patient


def extract_uploaded_slices(flair, t1, t1ce, t2, seg=None):
    if seg is not None:
        if not (flair.shape == t1.shape == t1ce.shape == t2.shape == seg.shape):
            raise ValueError("All uploaded modalities and segmentation must have the same shape.")
    else:
        if not (flair.shape == t1.shape == t1ce.shape == t2.shape):
            raise ValueError("All uploaded modalities must have the same shape.")

    num_slices = flair.shape[2]

    X_slices = []
    Y_slices = []

    for i in range(num_slices):
        image = create_input(
            flair[:, :, i],
            t1[:, :, i],
            t1ce[:, :, i],
            t2[:, :, i]
        )
        X_slices.append(image)

        if seg is not None:
            mask = create_mask(seg[:, :, i])
            Y_slices.append(mask)

    X_slices = np.array(X_slices, dtype=np.float32)

    if seg is not None:
        Y_slices = np.array(Y_slices, dtype=np.float32)
        return X_slices, Y_slices

    return X_slices, None