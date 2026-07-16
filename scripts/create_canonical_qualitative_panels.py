"""Create representative qualitative comparison panels for the controlled study."""

from __future__ import annotations

import csv
import math
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from summarise_canonical_ocr_300 import (
    character_similarity,
    normalise_text,
    read_labels,
)


OCR_CSV = Path(
    "experiments/metrics/easyocr_canonical_all_methods_300_results.csv"
)

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

OUTPUT_ROOT = Path(
    "experiments/visual_comparisons/canonical_selected_300"
)

SELECTION_CSV = Path(
    "experiments/metrics/canonical_qualitative_selection_300.csv"
)

OVERVIEW_IMAGE = OUTPUT_ROOT / "canonical_selected_overview.png"

DIFFICULTIES = ("easy", "medium", "hard")
OCR_METHODS = ("bicubic", "sr_realesrgan", "sr_tsrn", "hr")

METHOD_LABELS = {
    "bicubic": "Bicubic",
    "sr_realesrgan": "Real-ESRGAN",
    "sr_tsrn": "TSRN",
    "hr": "HR",
}

COLUMN_ORDER = (
    "lr",
    "bicubic",
    "sr_realesrgan",
    "sr_tsrn",
    "hr",
)

COLUMN_LABELS = {
    "lr": "Canonical LR",
    "bicubic": "Bicubic",
    "sr_realesrgan": "Real-ESRGAN",
    "sr_tsrn": "TSRN",
    "hr": "HR",
}


def sample_index_from_name(image_name: str) -> int:
    match = re.search(r"_(\d+)_", image_name)
    if not match:
        raise ValueError(f"Could not parse sample index from: {image_name}")
    return int(match.group(1))


def read_ocr_results() -> dict[tuple[str, int, str], dict]:
    data = {}

    with OCR_CSV.open("r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            difficulty = row["difficulty"].lower()
            method = row["image_type"].lower()
            index = sample_index_from_name(row["image_name"])

            key = (difficulty, index, method)

            if key in data:
                raise ValueError(f"Duplicate OCR row: {key}")

            data[key] = {
                "difficulty": difficulty,
                "index": index,
                "method": method,
                "image_name": row["image_name"],
                "ocr_text": row["ocr_text"],
                "confidence": float(row["confidence"] or 0),
                "notes": row["notes"].strip(),
            }

    if len(data) != 1200:
        raise ValueError(f"Expected 1200 OCR rows, found {len(data)}")

    return data


def image_path(difficulty: str, index: int, column: str) -> Path:
    prefix = f"{difficulty}_{index:03d}"

    if column == "lr":
        return DATASET_ROOT / difficulty / "lr" / f"{prefix}_lr.png"

    if column == "bicubic":
        return BICUBIC_ROOT / difficulty / f"{prefix}_lr_bicubic.png"

    if column == "sr_realesrgan":
        return REALESRGAN_ROOT / difficulty / f"{prefix}_lr_realesrgan.png"

    if column == "sr_tsrn":
        return TSRN_ROOT / difficulty / f"{prefix}_lr_tsrn.png"

    if column == "hr":
        return DATASET_ROOT / difficulty / "hr" / f"{prefix}_hr.png"

    raise ValueError(f"Unknown column: {column}")


def build_sample_records() -> list[dict]:
    labels = read_labels()
    ocr_map = read_ocr_results()

    records = []

    for difficulty in DIFFICULTIES:
        for index in range(1, 101):
            gt_text = labels[(difficulty, index)]
            gt_norm = normalise_text(gt_text)

            methods = {}

            for method in OCR_METHODS:
                row = ocr_map[(difficulty, index, method)]
                prediction = row["ocr_text"]
                prediction_norm = normalise_text(prediction)

                methods[method] = {
                    "prediction": prediction,
                    "confidence": row["confidence"],
                    "notes": row["notes"],
                    "exact_match": prediction_norm == gt_norm,
                    "character_similarity": character_similarity(
                        prediction_norm,
                        gt_norm,
                    ),
                }

            bicubic_sim = methods["bicubic"]["character_similarity"]
            realesrgan_sim = methods["sr_realesrgan"]["character_similarity"]
            tsrn_sim = methods["sr_tsrn"]["character_similarity"]

            best_baseline_sim = max(bicubic_sim, realesrgan_sim)

            records.append(
                {
                    "difficulty": difficulty,
                    "index": index,
                    "ground_truth": gt_text,
                    "methods": methods,
                    "tsrn_gain_vs_best_baseline": tsrn_sim - best_baseline_sim,
                    "tsrn_gap_vs_bicubic": tsrn_sim - bicubic_sim,
                    "tsrn_gap_vs_realesrgan": tsrn_sim - realesrgan_sim,
                    "best_baseline_similarity": best_baseline_sim,
                }
            )

    if len(records) != 300:
        raise ValueError(f"Expected 300 sample records, found {len(records)}")

    return records


def select_representative_samples(records: list[dict]) -> list[dict]:
    """Select meaningful word-based improvement and limitation examples."""

    selected = []

    def is_meaningful(row: dict) -> bool:
        label = normalise_text(row["ground_truth"])
        letter_count = sum(character.isalpha() for character in label)
        hr_similarity = row["methods"]["hr"]["character_similarity"]

        return (
            4 <= len(label) <= 14
            and letter_count >= 4
            and hr_similarity >= 0.75
        )

    for difficulty in DIFFICULTIES:
        group = [
            row
            for row in records
            if row["difficulty"] == difficulty
        ]

        meaningful = [
            row
            for row in group
            if is_meaningful(row)
        ]

        if not meaningful:
            meaningful = group

        strong_improvements = [
            row
            for row in meaningful
            if (
                row["tsrn_gain_vs_best_baseline"] >= 0.25
                and row["methods"]["sr_tsrn"]["character_similarity"] >= 0.60
            )
        ]

        positive_improvements = [
            row
            for row in meaningful
            if row["tsrn_gain_vs_best_baseline"] > 0
        ]

        improvement_pool = (
            strong_improvements
            or positive_improvements
            or meaningful
        )

        improvement = max(
            improvement_pool,
            key=lambda row: (
                int(row["methods"]["sr_tsrn"]["exact_match"]),
                row["tsrn_gain_vs_best_baseline"],
                row["methods"]["sr_tsrn"]["character_similarity"],
                row["methods"]["sr_tsrn"]["confidence"],
            ),
        )

        remaining = [
            row
            for row in meaningful
            if row["index"] != improvement["index"]
        ]

        strong_limitations = [
            row
            for row in remaining
            if (
                row["best_baseline_similarity"] >= 0.50
                and row["tsrn_gain_vs_best_baseline"] <= -0.10
            )
        ]

        underperformers = [
            row
            for row in remaining
            if row["tsrn_gain_vs_best_baseline"] < 0
        ]

        limitation_pool = (
            strong_limitations
            or underperformers
            or remaining
        )

        limitation = min(
            limitation_pool,
            key=lambda row: (
                row["tsrn_gain_vs_best_baseline"],
                -row["best_baseline_similarity"],
                row["methods"]["sr_tsrn"]["character_similarity"],
            ),
        )

        improvement = dict(improvement)
        limitation = dict(limitation)

        improvement["selection_reason"] = (
            "Representative TSRN improvement over baselines"
        )

        limitation["selection_reason"] = (
            "Representative TSRN limitation or failure"
        )

        selected.extend(
            (
                improvement,
                limitation,
            )
        )

    return selected


def write_selection_csv(selected: list[dict]) -> None:
    rows = []

    for row in selected:
        rows.append(
            {
                "difficulty": row["difficulty"],
                "index": row["index"],
                "selection_reason": row["selection_reason"],
                "ground_truth": row["ground_truth"],
                "bicubic_prediction": row["methods"]["bicubic"]["prediction"],
                "realesrgan_prediction": row["methods"]["sr_realesrgan"]["prediction"],
                "tsrn_prediction": row["methods"]["sr_tsrn"]["prediction"],
                "hr_prediction": row["methods"]["hr"]["prediction"],
                "bicubic_similarity": round(
                    row["methods"]["bicubic"]["character_similarity"], 4
                ),
                "realesrgan_similarity": round(
                    row["methods"]["sr_realesrgan"]["character_similarity"], 4
                ),
                "tsrn_similarity": round(
                    row["methods"]["sr_tsrn"]["character_similarity"], 4
                ),
                "hr_similarity": round(
                    row["methods"]["hr"]["character_similarity"], 4
                ),
                "tsrn_gain_vs_best_baseline": round(
                    row["tsrn_gain_vs_best_baseline"], 4
                ),
            }
        )

    SELECTION_CSV.parent.mkdir(parents=True, exist_ok=True)

    with SELECTION_CSV.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def wrap_text(value: str, width: int = 24) -> str:
    value = value.strip()
    if not value:
        return "None"
    return "\n".join(textwrap.wrap(value, width=width))


def render_panel(record: dict) -> Path:
    difficulty = record["difficulty"]
    index = record["index"]
    gt_text = record["ground_truth"]
    selection_reason = record["selection_reason"]

    output_folder = OUTPUT_ROOT / difficulty
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / f"{difficulty}_{index:03d}_panel.png"

    font = ImageFont.load_default()

    margin = 20
    title_height = 70
    image_width = 320
    image_height = 80
    text_height = 135
    cell_width = image_width
    cell_height = image_height + text_height
    footer_height = 10

    canvas_width = margin * 6 + cell_width * 5
    canvas_height = title_height + cell_height + margin * 2 + footer_height

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    title = (
        f"{difficulty.title()} | Sample {index:03d} | "
        f'GT: "{gt_text}" | {selection_reason}'
    )

    draw.text((margin, 15), title, fill="black", font=font)

    for column_index, column in enumerate(COLUMN_ORDER):
        x = margin + column_index * (cell_width + margin)
        y = title_height

        column_title = COLUMN_LABELS[column]
        draw.rectangle(
            [x, y, x + cell_width, y + cell_height],
            outline="black",
            width=2,
        )

        draw.text((x + 8, y + 8), column_title, fill="black", font=font)

        image_y = y + 28

        with Image.open(image_path(difficulty, index, column)) as image:
            resized = image.convert("RGB").resize(
                (image_width, image_height),
                Image.Resampling.NEAREST,
            )

        canvas.paste(resized, (x, image_y))

        draw.rectangle(
            [x, image_y, x + image_width, image_y + image_height],
            outline="black",
            width=1,
        )

        text_y = image_y + image_height + 8

        if column == "lr":
            info_lines = [
                "Input image",
                "Size: 64 x 16",
                "OCR: not evaluated separately",
            ]
        else:
            method_key = column
            if method_key == "hr":
                method_data = record["methods"]["hr"]
            else:
                method_data = record["methods"][method_key]

            prediction = method_data["prediction"]
            confidence = method_data["confidence"]
            similarity = method_data["character_similarity"]
            exact_match = method_data["exact_match"]

            info_lines = [
                f"OCR: {wrap_text(prediction)}",
                f"Conf: {confidence:.4f}",
                f"Char sim: {similarity:.1%}",
                f"Exact match: {'Yes' if exact_match else 'No'}",
            ]

        offset = 0
        for line in info_lines:
            draw.multiline_text(
                (x + 8, text_y + offset),
                line,
                fill="black",
                font=font,
                spacing=2,
            )
            line_count = line.count("\n") + 1
            offset += 16 * line_count

    canvas.save(output_path)
    return output_path



def create_overview(panel_paths: list[Path]) -> None:
    images = [Image.open(path).convert("RGB") for path in panel_paths]

    thumb_width = 900
    thumb_height = int(images[0].height * (thumb_width / images[0].width))

    thumbs = []
    for image in images:
        thumbs.append(
            image.resize(
                (thumb_width, thumb_height),
                Image.Resampling.BICUBIC,
            )
        )

    cols = 1
    rows = len(thumbs)

    margin = 20
    canvas_width = thumb_width + margin * 2
    canvas_height = rows * thumb_height + (rows + 1) * margin

    overview = Image.new("RGB", (canvas_width, canvas_height), "white")

    for i, thumb in enumerate(thumbs):
        x = margin
        y = margin + i * (thumb_height + margin)
        overview.paste(thumb, (x, y))

    OVERVIEW_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    overview.save(OVERVIEW_IMAGE)

    for image in images:
        image.close()



def create_group_overview(
    panel_paths: list[Path],
    output_path: Path,
) -> None:
    """Stack three full-width panels into one readable overview."""

    images = [
        Image.open(path).convert("RGB")
        for path in panel_paths
    ]

    target_width = 1500
    resized_images = []

    for image in images:
        target_height = round(
            image.height
            * target_width
            / image.width
        )

        resized_images.append(
            image.resize(
                (
                    target_width,
                    target_height,
                ),
                Image.Resampling.LANCZOS,
            )
        )

    margin = 30

    canvas_width = target_width + margin * 2

    canvas_height = (
        sum(image.height for image in resized_images)
        + margin * (len(resized_images) + 1)
    )

    canvas = Image.new(
        "RGB",
        (
            canvas_width,
            canvas_height,
        ),
        "white",
    )

    y = margin

    for image in resized_images:
        canvas.paste(
            image,
            (
                margin,
                y,
            ),
        )

        y += image.height + margin

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(output_path)

    for image in images:
        image.close()

def main() -> None:
    records = build_sample_records()
    selected = select_representative_samples(records)
    write_selection_csv(selected)

    panel_paths = []

    for record in selected:
        panel_path = render_panel(record)
        panel_paths.append(panel_path)

    create_overview(panel_paths)

    improvement_paths = [
        panel_paths[0],
        panel_paths[2],
        panel_paths[4],
    ]

    limitation_paths = [
        panel_paths[1],
        panel_paths[3],
        panel_paths[5],
    ]

    create_group_overview(
        improvement_paths,
        OUTPUT_ROOT
        / "canonical_tsrn_improvements_overview.png",
    )

    create_group_overview(
        limitation_paths,
        OUTPUT_ROOT
        / "canonical_tsrn_limitations_overview.png",
    )

    improvement_paths = [
        panel_paths[0],
        panel_paths[2],
        panel_paths[4],
    ]

    limitation_paths = [
        panel_paths[1],
        panel_paths[3],
        panel_paths[5],
    ]

    create_group_overview(
        improvement_paths,
        OUTPUT_ROOT
        / "canonical_tsrn_improvements_overview.png",
    )

    create_group_overview(
        limitation_paths,
        OUTPUT_ROOT
        / "canonical_tsrn_limitations_overview.png",
    )

    print("Selected samples:", len(selected))
    print("Saved selection CSV:", SELECTION_CSV)
    print("Saved overview image:", OVERVIEW_IMAGE)
    print()
    print("Individual panel images:")
    for path in panel_paths:
        print(path)

def create_group_overview(
    panel_paths: list[Path],
    output_path: Path,
) -> None:
    """Stack three full-width panels into one readable overview."""

    images = [
        Image.open(path).convert("RGB")
        for path in panel_paths
    ]

    target_width = 1500

    resized_images = []

    for image in images:
        target_height = round(
            image.height
            * target_width
            / image.width
        )

        resized_images.append(
            image.resize(
                (
                    target_width,
                    target_height,
                ),
                Image.Resampling.LANCZOS,
            )
        )

    margin = 30

    canvas_width = target_width + margin * 2

    canvas_height = (
        sum(image.height for image in resized_images)
        + margin * (len(resized_images) + 1)
    )

    canvas = Image.new(
        "RGB",
        (
            canvas_width,
            canvas_height,
        ),
        "white",
    )

    y = margin

    for image in resized_images:
        canvas.paste(
            image,
            (
                margin,
                y,
            ),
        )

        y += image.height + margin

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        output_path,
        quality=95,
    )

    for image in images:
        image.close()
        
if __name__ == "__main__":
    main()
