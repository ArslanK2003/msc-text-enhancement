"""Evaluate TSRN outputs against the 300 TextZoom HR references."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


HR_DIR = Path("datasets/textzoom_subset_300")
SR_DIR = Path("experiments/outputs/textzoom_tsrn_300")
OUTPUT_CSV = Path("experiments/metrics/psnr_ssim_tsrn_300_results.csv")
SUMMARY_CSV = Path("experiments/metrics/psnr_ssim_tsrn_300_summary.csv")

DIFFICULTIES = ("easy", "medium", "hard")
GAUSSIAN_KERNEL_SIZE = 11
GAUSSIAN_SIGMA = 1.5


def calculate_psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = np.mean((reference - candidate) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    coordinates = np.arange(size, dtype=np.float64) - size // 2
    kernel = np.exp(-(coordinates**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_filter(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a separable Gaussian filter using only NumPy."""

    padding = len(kernel) // 2
    horizontally_padded = np.pad(channel, ((0, 0), (padding, padding)), mode="reflect")
    horizontal_windows = np.lib.stride_tricks.sliding_window_view(
        horizontally_padded, len(kernel), axis=1
    )
    horizontal = np.tensordot(horizontal_windows, kernel, axes=([-1], [0]))
    vertically_padded = np.pad(horizontal, ((padding, padding), (0, 0)), mode="reflect")
    vertical_windows = np.lib.stride_tricks.sliding_window_view(
        vertically_padded, len(kernel), axis=0
    )
    return np.tensordot(vertical_windows, kernel, axes=([-1], [0]))


def calculate_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Calculate channel-averaged local SSIM with an 11x11 Gaussian window."""

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = gaussian_kernel(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
    channel_scores = []

    for channel_index in range(reference.shape[2]):
        reference_channel = reference[:, :, channel_index]
        candidate_channel = candidate[:, :, channel_index]

        reference_mean = gaussian_filter(reference_channel, kernel)
        candidate_mean = gaussian_filter(candidate_channel, kernel)

        reference_variance = gaussian_filter(reference_channel**2, kernel) - reference_mean**2
        candidate_variance = gaussian_filter(candidate_channel**2, kernel) - candidate_mean**2
        covariance = (
            gaussian_filter(reference_channel * candidate_channel, kernel)
            - reference_mean * candidate_mean
        )

        reference_variance = np.maximum(reference_variance, 0)
        candidate_variance = np.maximum(candidate_variance, 0)

        numerator = (2 * reference_mean * candidate_mean + c1) * (2 * covariance + c2)
        denominator = (
            (reference_mean**2 + candidate_mean**2 + c1)
            * (reference_variance + candidate_variance + c2)
        )
        channel_scores.append(float(np.mean(numerator / denominator)))

    return float(np.mean(channel_scores))


def load_aligned_images(sr_path: Path, hr_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with Image.open(hr_path) as image:
        hr_image = image.convert("RGB")
    with Image.open(sr_path) as image:
        sr_image = image.convert("RGB")

    resize_applied = sr_image.size != hr_image.size
    if resize_applied:
        sr_image = sr_image.resize(hr_image.size, Image.Resampling.BICUBIC)

    hr_array = np.asarray(hr_image, dtype=np.float64)
    sr_array = np.asarray(sr_image, dtype=np.float64)
    return hr_array, sr_array, "bicubic_to_hr" if resize_applied else "none"


def evaluate_images() -> list[dict[str, object]]:
    rows = []
    for difficulty in DIFFICULTIES:
        hr_folder = HR_DIR / difficulty / "hr"
        sr_folder = SR_DIR / difficulty
        hr_paths = sorted(hr_folder.glob("*_hr.png"))

        if len(hr_paths) != 100:
            raise ValueError(f"Expected 100 HR images in {hr_folder}; found {len(hr_paths)}")

        for hr_path in hr_paths:
            index = hr_path.stem.split("_")[1]
            sr_name = f"{difficulty}_{index}_lr_tsrn.png"
            sr_path = sr_folder / sr_name
            if not sr_path.exists():
                raise FileNotFoundError(f"Missing SR image: {sr_path}")

            hr_array, sr_array, alignment = load_aligned_images(sr_path, hr_path)
            psnr = calculate_psnr(hr_array, sr_array)
            ssim = calculate_ssim(hr_array, sr_array)

            rows.append(
                {
                    "difficulty": difficulty,
                    "image_name": sr_name,
                    "psnr": round(psnr, 4),
                    "ssim": round(ssim, 4),
                    "alignment": alignment,
                }
            )

    if len(rows) != 300:
        raise ValueError(f"Expected 300 evaluated pairs; found {len(rows)}")
    return rows


def write_results(rows: list[dict[str, object]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["difficulty", "image_name", "psnr", "ssim", "alignment"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["difficulty"]].append(row)
        grouped["overall"].append(row)

    summary_rows = []
    for difficulty in (*DIFFICULTIES, "overall"):
        values = grouped[difficulty]
        psnr_values = np.asarray([row["psnr"] for row in values], dtype=np.float64)
        ssim_values = np.asarray([row["ssim"] for row in values], dtype=np.float64)
        summary_rows.append(
            {
                "difficulty": difficulty,
                "num_images": len(values),
                "mean_psnr": round(float(psnr_values.mean()), 4),
                "std_psnr": round(float(psnr_values.std()), 4),
                "mean_ssim": round(float(ssim_values.mean()), 4),
                "std_ssim": round(float(ssim_values.std()), 4),
            }
        )

    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def main() -> None:
    rows = evaluate_images()
    write_results(rows)
    summary_rows = write_summary(rows)

    print("\nTSRN PSNR/SSIM summary")
    print("Difficulty | Images | Mean PSNR | Mean SSIM")
    for row in summary_rows:
        print(
            f"{row['difficulty'].title():10} | {row['num_images']:6} | "
            f"{row['mean_psnr']:9.4f} | {row['mean_ssim']:.4f}"
        )
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {SUMMARY_CSV}")
    print("Note: TSRN outputs are expected to match the HR dimensions directly.")


if __name__ == "__main__":
    main()

