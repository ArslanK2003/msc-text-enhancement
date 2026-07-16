"""Run EasyOCR on canonical Bicubic, Real-ESRGAN, TSRN and HR images."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import easyocr
from PIL import Image


DATASET_ROOT = Path("datasets/textzoom_subset_300_canonical")

BICUBIC_ROOT = Path(
    "experiments/outputs/textzoom_bicubic_canonical_300"
)

REALESRGAN_ROOT = Path(
    "experiments/outputs/textzoom_realesrgan_canonical_300"
)

TSRN_ROOT = Path(
    "experiments/outputs/textzoom_tsrn_300"
)

OUTPUT_CSV = Path(
    "experiments/metrics/easyocr_canonical_all_methods_300_results.csv"
)

DIFFICULTIES = ("easy", "medium", "hard")
METHODS = ("bicubic", "sr_realesrgan", "sr_tsrn", "hr")

LR_SIZE = (64, 16)
TARGET_SIZE = (128, 32)


def create_bicubic_images() -> None:
    total_created = 0

    for difficulty in DIFFICULTIES:
        output_folder = BICUBIC_ROOT / difficulty
        output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index in range(1, 101):
            prefix = f"{difficulty}_{index:03d}"

            input_path = (
                DATASET_ROOT
                / difficulty
                / "lr"
                / f"{prefix}_lr.png"
            )

            output_path = (
                output_folder
                / f"{prefix}_lr_bicubic.png"
            )

            if not input_path.exists():
                raise FileNotFoundError(input_path)

            with Image.open(input_path) as image:
                rgb = image.convert("RGB")

                if rgb.size != LR_SIZE:
                    raise ValueError(
                        f"Unexpected LR size for {input_path}: "
                        f"{rgb.size}"
                    )

                upscaled = rgb.resize(
                    TARGET_SIZE,
                    Image.Resampling.BICUBIC,
                )

                upscaled.save(output_path)

            total_created += 1

    if total_created != 300:
        raise ValueError(
            f"Expected 300 bicubic images; created {total_created}"
        )

    print("Created 300 bicubic baseline images.")


def get_image_path(
    difficulty: str,
    index: int,
    method: str,
) -> Path:
    prefix = f"{difficulty}_{index:03d}"

    if method == "bicubic":
        return (
            BICUBIC_ROOT
            / difficulty
            / f"{prefix}_lr_bicubic.png"
        )

    if method == "sr_realesrgan":
        return (
            REALESRGAN_ROOT
            / difficulty
            / f"{prefix}_lr_realesrgan.png"
        )

    if method == "sr_tsrn":
        return (
            TSRN_ROOT
            / difficulty
            / f"{prefix}_lr_tsrn.png"
        )

    if method == "hr":
        return (
            DATASET_ROOT
            / difficulty
            / "hr"
            / f"{prefix}_hr.png"
        )

    raise ValueError(f"Unknown method: {method}")


def validate_images() -> None:
    counts = Counter()

    for difficulty in DIFFICULTIES:
        for method in METHODS:
            for index in range(1, 101):
                image_path = get_image_path(
                    difficulty,
                    index,
                    method,
                )

                if not image_path.exists():
                    raise FileNotFoundError(image_path)

                with Image.open(image_path) as image:
                    if image.size != TARGET_SIZE:
                        raise ValueError(
                            f"Unexpected size for {image_path}: "
                            f"{image.size}; expected {TARGET_SIZE}"
                        )

                counts[(difficulty, method)] += 1

    for difficulty in DIFFICULTIES:
        for method in METHODS:
            count = counts[(difficulty, method)]

            if count != 100:
                raise ValueError(
                    f"Expected 100 {difficulty}/{method} images; "
                    f"found {count}"
                )

    print("Validated 1,200 canonical OCR input images.")


def run_ocr() -> list[dict[str, object]]:
    reader = easyocr.Reader(
        ["en"],
        gpu=False,
    )

    rows = []
    processed = 0

    for difficulty in DIFFICULTIES:
        for method in METHODS:
            for index in range(1, 101):
                image_path = get_image_path(
                    difficulty,
                    index,
                    method,
                )

                results = reader.readtext(
                    str(image_path)
                )

                if not results:
                    ocr_text = ""
                    confidence = 0.0
                    notes = "No text detected"
                else:
                    detected_texts = [
                        text
                        for _, text, _ in results
                    ]

                    confidences = [
                        float(confidence)
                        for _, _, confidence in results
                    ]

                    ocr_text = " ".join(
                        detected_texts
                    )

                    confidence = (
                        sum(confidences)
                        / len(confidences)
                    )

                    notes = ""

                rows.append(
                    {
                        "difficulty": difficulty,
                        "image_type": method,
                        "image_name": image_path.name,
                        "ocr_text": ocr_text,
                        "confidence": round(
                            confidence,
                            4,
                        ),
                        "notes": notes,
                    }
                )

                processed += 1

                if processed % 25 == 0:
                    print(
                        f"Processed {processed}/1200"
                    )

    if len(rows) != 1200:
        raise ValueError(
            f"Expected 1,200 OCR rows; found {len(rows)}"
        )

    return rows


def write_results(
    rows: list[dict[str, object]],
) -> None:
    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "difficulty",
        "image_type",
        "image_name",
        "ocr_text",
        "confidence",
        "notes",
    ]

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    create_bicubic_images()
    validate_images()

    rows = run_ocr()
    write_results(rows)

    print()
    print("OCR evaluation completed.")
    print("Rows:", len(rows))
    print("Saved:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
