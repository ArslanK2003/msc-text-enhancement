"""Evaluate Bicubic, Real-ESRGAN and TSRN at a common 128 x 32 size."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_psnr_ssim_realesrgan_300 import calculate_psnr, calculate_ssim


DATASET_ROOT = Path("datasets/textzoom_subset_300_canonical")
REALESRGAN_ROOT = Path(
    "experiments/outputs/textzoom_realesrgan_canonical_300"
)
TSRN_ROOT = Path("experiments/outputs/textzoom_tsrn_300")

RESULTS_CSV = Path(
    "experiments/metrics/canonical_psnr_ssim_all_methods_300_results.csv"
)
SUMMARY_CSV = Path(
    "experiments/metrics/canonical_psnr_ssim_all_methods_300_summary.csv"
)

DIFFICULTIES = ("easy", "medium", "hard")
METHODS = ("bicubic", "realesrgan", "tsrn")

METHOD_LABELS = {
    "bicubic": "Bicubic",
    "realesrgan": "Real-ESRGAN",
    "tsrn": "TSRN",
}

LR_SIZE = (64, 16)
TARGET_SIZE = (128, 32)


def load_fixed_image(
    path: Path,
    expected_size: tuple[int, int],
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    with Image.open(path) as image:
        rgb = image.convert("RGB")

        if rgb.size != expected_size:
            raise ValueError(
                f"Unexpected size for {path}: "
                f"{rgb.size}; expected {expected_size}"
            )

        return np.asarray(
            rgb,
            dtype=np.float64,
        )


def load_bicubic_baseline(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    with Image.open(path) as image:
        rgb = image.convert("RGB")

        if rgb.size != LR_SIZE:
            raise ValueError(
                f"Unexpected LR size for {path}: "
                f"{rgb.size}; expected {LR_SIZE}"
            )

        upscaled = rgb.resize(
            TARGET_SIZE,
            Image.Resampling.BICUBIC,
        )

        return np.asarray(
            upscaled,
            dtype=np.float64,
        )


def evaluate() -> list[dict[str, object]]:
    rows = []

    for difficulty in DIFFICULTIES:
        for index in range(1, 101):
            prefix = f"{difficulty}_{index:03d}"

            lr_path = (
                DATASET_ROOT
                / difficulty
                / "lr"
                / f"{prefix}_lr.png"
            )

            hr_path = (
                DATASET_ROOT
                / difficulty
                / "hr"
                / f"{prefix}_hr.png"
            )

            realesrgan_path = (
                REALESRGAN_ROOT
                / difficulty
                / f"{prefix}_lr_realesrgan.png"
            )

            tsrn_path = (
                TSRN_ROOT
                / difficulty
                / f"{prefix}_lr_tsrn.png"
            )

            reference = load_fixed_image(
                hr_path,
                TARGET_SIZE,
            )

            candidates = {
                "bicubic": (
                    load_bicubic_baseline(lr_path),
                    "64x16_bicubic_to_128x32",
                ),
                "realesrgan": (
                    load_fixed_image(
                        realesrgan_path,
                        TARGET_SIZE,
                    ),
                    "canonical_128x32",
                ),
                "tsrn": (
                    load_fixed_image(
                        tsrn_path,
                        TARGET_SIZE,
                    ),
                    "canonical_128x32",
                ),
            }

            for method, (
                candidate,
                preparation,
            ) in candidates.items():
                rows.append(
                    {
                        "difficulty": difficulty,
                        "index": index,
                        "method": method,
                        "psnr": round(
                            calculate_psnr(
                                reference,
                                candidate,
                            ),
                            4,
                        ),
                        "ssim": round(
                            calculate_ssim(
                                reference,
                                candidate,
                            ),
                            4,
                        ),
                        "preparation": preparation,
                        "reference_size": "128x32",
                        "candidate_size": "128x32",
                    }
                )

    expected = 300 * len(METHODS)

    if len(rows) != expected:
        raise ValueError(
            f"Expected {expected} rows; "
            f"found {len(rows)}"
        )

    return rows


def write_results(
    rows: list[dict[str, object]],
) -> None:
    RESULTS_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def create_summary(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped = defaultdict(list)

    for row in rows:
        grouped[
            (
                row["difficulty"],
                row["method"],
            )
        ].append(row)

        grouped[
            (
                "overall",
                row["method"],
            )
        ].append(row)

    summary_rows = []

    for difficulty in (*DIFFICULTIES, "overall"):
        for method in METHODS:
            values = grouped[
                (
                    difficulty,
                    method,
                )
            ]

            psnr_values = np.asarray(
                [
                    row["psnr"]
                    for row in values
                ]
            )

            ssim_values = np.asarray(
                [
                    row["ssim"]
                    for row in values
                ]
            )

            summary_rows.append(
                {
                    "difficulty": difficulty,
                    "method": method,
                    "num_images": len(values),
                    "mean_psnr": round(
                        float(
                            psnr_values.mean()
                        ),
                        4,
                    ),
                    "std_psnr": round(
                        float(
                            psnr_values.std()
                        ),
                        4,
                    ),
                    "mean_ssim": round(
                        float(
                            ssim_values.mean()
                        ),
                        4,
                    ),
                    "std_ssim": round(
                        float(
                            ssim_values.std()
                        ),
                        4,
                    ),
                }
            )

    with SUMMARY_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_rows


def print_summary(
    summary_rows: list[dict[str, object]],
) -> None:
    lookup = {
        (
            row["difficulty"],
            row["method"],
        ): row
        for row in summary_rows
    }

    print(
        "\nCanonical 128 x 32 "
        "PSNR/SSIM summary"
    )

    print(
        "Difficulty | Method       | "
        "Images | Mean PSNR | Mean SSIM"
    )

    for difficulty in (
        *DIFFICULTIES,
        "overall",
    ):
        for method in METHODS:
            row = lookup[
                (
                    difficulty,
                    method,
                )
            ]

            print(
                f"{difficulty.title():10} | "
                f"{METHOD_LABELS[method]:12} | "
                f"{row['num_images']:6} | "
                f"{row['mean_psnr']:9.4f} | "
                f"{row['mean_ssim']:.4f}"
            )


def main() -> None:
    rows = evaluate()
    write_results(rows)

    summary_rows = create_summary(rows)
    print_summary(summary_rows)

    print(f"\nSaved: {RESULTS_CSV}")
    print(f"Saved: {SUMMARY_CSV}")

    print(
        "Protocol: RGB PSNR and "
        "channel-averaged 11x11 "
        "Gaussian-window SSIM at 128 x 32."
    )


if __name__ == "__main__":
    main()
