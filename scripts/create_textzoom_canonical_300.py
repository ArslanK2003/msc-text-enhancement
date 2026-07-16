"""Create fixed-size TextZoom LR and HR images for fair model evaluation."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image


INPUT_ROOT = Path("datasets/textzoom_subset_300")
OUTPUT_ROOT = Path("datasets/textzoom_subset_300_canonical")
LABELS_CSV = INPUT_ROOT / "textzoom_subset_300_labels.csv"
OUTPUT_METADATA_CSV = OUTPUT_ROOT / "canonical_subset_metadata.csv"

DIFFICULTIES = ("easy", "medium", "hard")

LR_SIZE = (64, 16)
HR_SIZE = (128, 32)


def read_labels() -> dict[tuple[str, int], str]:
    labels = {}

    with LABELS_CSV.open("r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            key = (
                row["difficulty"].lower(),
                int(row["index"]),
            )
            labels[key] = row["label"]

    return labels


def main() -> None:
    labels = read_labels()
    metadata_rows = []

    for difficulty in DIFFICULTIES:
        lr_output_folder = OUTPUT_ROOT / difficulty / "lr"
        hr_output_folder = OUTPUT_ROOT / difficulty / "hr"

        lr_output_folder.mkdir(parents=True, exist_ok=True)
        hr_output_folder.mkdir(parents=True, exist_ok=True)

        for index in range(1, 101):
            lr_name = f"{difficulty}_{index:03d}_lr.png"
            hr_name = f"{difficulty}_{index:03d}_hr.png"

            lr_source = (
                INPUT_ROOT
                / difficulty
                / "lr"
                / lr_name
            )

            hr_source = (
                INPUT_ROOT
                / difficulty
                / "hr"
                / hr_name
            )

            if not lr_source.exists():
                raise FileNotFoundError(lr_source)

            if not hr_source.exists():
                raise FileNotFoundError(hr_source)

            with Image.open(lr_source) as image:
                lr_original = image.convert("RGB")
                original_lr_size = lr_original.size

                lr_canonical = lr_original.resize(
                    LR_SIZE,
                    Image.Resampling.BICUBIC,
                )

            with Image.open(hr_source) as image:
                hr_original = image.convert("RGB")
                original_hr_size = hr_original.size

                hr_canonical = hr_original.resize(
                    HR_SIZE,
                    Image.Resampling.BICUBIC,
                )

            lr_destination = lr_output_folder / lr_name
            hr_destination = hr_output_folder / hr_name

            lr_canonical.save(lr_destination)
            hr_canonical.save(hr_destination)

            metadata_rows.append(
                {
                    "difficulty": difficulty,
                    "index": index,
                    "label": labels[(difficulty, index)],
                    "original_lr_width": original_lr_size[0],
                    "original_lr_height": original_lr_size[1],
                    "original_hr_width": original_hr_size[0],
                    "original_hr_height": original_hr_size[1],
                    "canonical_lr_width": LR_SIZE[0],
                    "canonical_lr_height": LR_SIZE[1],
                    "canonical_hr_width": HR_SIZE[0],
                    "canonical_hr_height": HR_SIZE[1],
                    "lr_image": str(lr_destination),
                    "hr_image": str(hr_destination),
                }
            )

        print(
            f"{difficulty}: created "
            f"100 LR and 100 HR canonical images"
        )

    with OUTPUT_METADATA_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        fieldnames = list(metadata_rows[0].keys())

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(metadata_rows)

    print()
    print("Total paired samples:", len(metadata_rows))
    print("Canonical LR size:", LR_SIZE)
    print("Canonical HR size:", HR_SIZE)
    print("Saved metadata:", OUTPUT_METADATA_CSV)


if __name__ == "__main__":
    main()
