import csv
import os
from datetime import datetime

from src.config import FINAL_REPORT_PATH, METRICS_CSV_PATH


def save_text_report(metrics, extra_info=None, file_path=FINAL_REPORT_PATH):
    lines = []
    lines.append("Brain Tumor Segmentation Report")
    lines.append("=" * 50)
    lines.append(f"Generated on: {datetime.now()}")
    lines.append("")

    if extra_info:
        lines.append("Project Information")
        lines.append("-" * 30)
        for key, value in extra_info.items():
            lines.append(f"{key}: {value}")
        lines.append("")

    lines.append("Key Metrics (Trustworthy)")
    lines.append("-" * 30)
    for key in ["pixel_accuracy_all", "tumor_only_accuracy", "mean_tumor_dice", "mean_tumor_iou"]:
        if key in metrics:
            lines.append(f"{key}: {metrics[key]:.6f}")

    lines.append("")
    lines.append("Class-wise Metrics")
    lines.append("-" * 30)
    for key, value in metrics.items():
        if key.startswith("class_"):
            lines.append(f"{key}: {value:.6f}")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_metrics_csv(metrics, file_path=METRICS_CSV_PATH):
    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(metrics.keys())

        writer.writerow(metrics.values())