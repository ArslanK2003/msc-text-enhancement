"""Create clear LR | Real-ESRGAN | HR comparisons for 300 TextZoom samples.

Every word crop receives an amber outline so the text region is visually
highlighted.  The script also creates a balanced six-sample overview: the
largest OCR improvement and largest OCR decline from each difficulty group.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SUBSET_DIR = Path("datasets/textzoom_subset_300")
SR_DIR = Path("experiments/outputs/textzoom_realesrgan_300")
LR_HR_OCR_CSV = Path("experiments/metrics/easyocr_textzoom_subset_300_results.csv")
SR_OCR_CSV = Path("experiments/metrics/easyocr_realesrgan_300_results.csv")
LABELS_CSV = Path("datasets/textzoom_subset_300/textzoom_subset_300_labels.csv")

OUTPUT_DIR = Path("experiments/visual_comparisons/realesrgan_300")
SELECTION_CSV = Path("experiments/metrics/realesrgan_qualitative_selection_300.csv")

DIFFICULTIES = ("easy", "medium", "hard")
METHODS = ("lr", "sr_realesrgan", "hr")
METHOD_LABELS = {
    "lr": "LR input",
    "sr_realesrgan": "Real-ESRGAN SR",
    "hr": "HR ground truth",
}
METHOD_COLOURS = {
    "lr": (66, 103, 178),
    "sr_realesrgan": (230, 126, 34),
    "hr": (46, 139, 87),
}

PANEL_WIDTH = 1380
PANEL_HEIGHT = 340
COLUMN_WIDTH = 440
COLUMN_GAP = 20
LEFT_MARGIN = 20
IMAGE_AREA_TOP = 105
IMAGE_AREA_HEIGHT = 150
MAX_IMAGE_WIDTH = 400
MAX_IMAGE_HEIGHT = 125
HIGHLIGHT_COLOUR = (255, 193, 7)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


TITLE_FONT = load_font(24, bold=True)
LABEL_FONT = load_font(22, bold=True)
TEXT_FONT = load_font(18)
SMALL_FONT = load_font(16)


def normalise_text(value: str) -> str:
    return "".join(character for character in value.upper() if character.isalnum())


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def character_similarity(prediction: str, ground_truth: str) -> float:
    prediction = normalise_text(prediction)
    ground_truth = normalise_text(ground_truth)
    denominator = max(len(prediction), len(ground_truth), 1)
    score = 1.0 - levenshtein_distance(prediction, ground_truth) / denominator
    return max(0.0, score)


def sample_key(difficulty: str, image_name: str) -> tuple[str, int]:
    match = re.search(r"_(\d+)_", image_name)
    if not match:
        raise ValueError(f"Cannot find sample index in: {image_name}")
    return difficulty.lower(), int(match.group(1))


def read_labels() -> dict[tuple[str, int], str]:
    labels = {}
    with LABELS_CSV.open("r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            labels[(row["difficulty"].lower(), int(row["index"]))] = row["label"]
    return labels


def read_ocr_results() -> dict[tuple[str, int, str], dict[str, object]]:
    results = {}
    for csv_path in (LR_HR_OCR_CSV, SR_OCR_CSV):
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                difficulty, index = sample_key(row["difficulty"], row["image_name"])
                method = row["image_type"].lower()
                key = (difficulty, index, method)
                if key in results:
                    raise ValueError(f"Duplicate OCR row for {key}")
                results[key] = {
                    "prediction": row["ocr_text"],
                    "confidence": float(row["confidence"] or 0),
                    "no_text": row["notes"].strip().lower() == "no text detected",
                }
    return results


def image_paths(difficulty: str, index: int) -> dict[str, Path]:
    sample_number = f"{index:03d}"
    return {
        "lr": SUBSET_DIR / difficulty / "lr" / f"{difficulty}_{sample_number}_lr.png",
        "sr_realesrgan": SR_DIR / difficulty / f"{difficulty}_{sample_number}_lr_sr.png",
        "hr": SUBSET_DIR / difficulty / "hr" / f"{difficulty}_{sample_number}_hr.png",
    }


def fit_image(image: Image.Image) -> Image.Image:
    scale = min(MAX_IMAGE_WIDTH / image.width, MAX_IMAGE_HEIGHT / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def display_prediction(result: dict[str, object]) -> str:
    prediction = str(result["prediction"]).strip() or "[no text]"
    if len(prediction) > 34:
        prediction = prediction[:31] + "..."
    return prediction


def create_comparison(
    difficulty: str,
    index: int,
    label: str,
    ocr_results: dict[tuple[str, int, str], dict[str, object]],
) -> Image.Image:
    lr_similarity = character_similarity(
        str(ocr_results[(difficulty, index, "lr")]["prediction"]), label
    )
    sr_similarity = character_similarity(
        str(ocr_results[(difficulty, index, "sr_realesrgan")]["prediction"]),
        label,
    )
    delta = sr_similarity - lr_similarity

    canvas = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, PANEL_WIDTH, 58), fill=(247, 249, 252))
    draw.rectangle((0, 58, PANEL_WIDTH, 62), fill=HIGHLIGHT_COLOUR)
    title = (
        f"{difficulty.title()} #{index:03d} | Ground truth: {label} | "
        f"SR-LR character similarity change: {delta:+.3f}"
    )
    draw.text((LEFT_MARGIN, 15), title, font=TITLE_FONT, fill=(25, 32, 44))

    paths = image_paths(difficulty, index)
    for column_index, method in enumerate(METHODS):
        x0 = LEFT_MARGIN + column_index * (COLUMN_WIDTH + COLUMN_GAP)
        x1 = x0 + COLUMN_WIDTH
        draw.rounded_rectangle(
            (x0, 75, x1, PANEL_HEIGHT - 12),
            radius=10,
            fill=(252, 252, 252),
            outline=METHOD_COLOURS[method],
            width=3,
        )

        label_width = draw.textbbox((0, 0), METHOD_LABELS[method], font=LABEL_FONT)[2]
        draw.text(
            (x0 + (COLUMN_WIDTH - label_width) // 2, 79),
            METHOD_LABELS[method],
            font=LABEL_FONT,
            fill=METHOD_COLOURS[method],
        )

        path = paths[method]
        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as source:
            displayed = fit_image(source.convert("RGB"))

        image_x = x0 + (COLUMN_WIDTH - displayed.width) // 2
        image_y = IMAGE_AREA_TOP + (IMAGE_AREA_HEIGHT - displayed.height) // 2
        canvas.paste(displayed, (image_x, image_y))
        draw.rectangle(
            (
                image_x - 5,
                image_y - 5,
                image_x + displayed.width + 4,
                image_y + displayed.height + 4,
            ),
            outline=HIGHLIGHT_COLOUR,
            width=5,
        )

        result = ocr_results[(difficulty, index, method)]
        prediction = display_prediction(result)
        confidence = float(result["confidence"])
        similarity = character_similarity(str(result["prediction"]), label)
        draw.text(
            (x0 + 14, 267),
            f"OCR: {prediction}",
            font=TEXT_FONT,
            fill=(30, 30, 30),
        )
        draw.text(
            (x0 + 14, 298),
            f"Confidence: {confidence:.4f} | Character similarity: {similarity:.3f}",
            font=SMALL_FONT,
            fill=(75, 75, 75),
        )
    return canvas


def select_representative_samples(
    labels: dict[tuple[str, int], str],
    ocr_results: dict[tuple[str, int, str], dict[str, object]],
) -> list[dict[str, object]]:
    selected = []
    for difficulty in DIFFICULTIES:
        candidates = []
        for index in range(1, 101):
            label = labels[(difficulty, index)]
            lr_prediction = str(ocr_results[(difficulty, index, "lr")]["prediction"])
            sr_prediction = str(
                ocr_results[(difficulty, index, "sr_realesrgan")]["prediction"]
            )
            lr_similarity = character_similarity(lr_prediction, label)
            sr_similarity = character_similarity(sr_prediction, label)
            candidates.append(
                {
                    "difficulty": difficulty,
                    "index": index,
                    "ground_truth": label,
                    "lr_ocr": lr_prediction,
                    "sr_ocr": sr_prediction,
                    "lr_character_similarity": lr_similarity,
                    "sr_character_similarity": sr_similarity,
                    "change": sr_similarity - lr_similarity,
                }
            )

        candidates.sort(key=lambda row: (row["change"], row["index"]))
        decline = dict(candidates[0])
        improvement = dict(candidates[-1])
        decline["selection_reason"] = "largest SR decline in difficulty group"
        improvement["selection_reason"] = "largest SR improvement in difficulty group"
        selected.extend((improvement, decline))
    return selected


def write_selection_csv(selected: list[dict[str, object]]) -> None:
    SELECTION_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "difficulty",
        "index",
        "selection_reason",
        "ground_truth",
        "lr_ocr",
        "sr_ocr",
        "lr_character_similarity",
        "sr_character_similarity",
        "change",
    ]
    with SELECTION_CSV.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            output_row = dict(row)
            for field in (
                "lr_character_similarity",
                "sr_character_similarity",
                "change",
            ):
                output_row[field] = round(float(output_row[field]), 4)
            writer.writerow(output_row)


def create_overview(
    selected: list[dict[str, object]],
    labels: dict[tuple[str, int], str],
    ocr_results: dict[tuple[str, int, str], dict[str, object]],
) -> None:
    panels = [
        create_comparison(
            str(row["difficulty"]),
            int(row["index"]),
            labels[(str(row["difficulty"]), int(row["index"]))],
            ocr_results,
        )
        for row in selected
    ]
    gap = 20
    overview = Image.new(
        "RGB",
        (PANEL_WIDTH * 2 + gap, PANEL_HEIGHT * 3 + gap * 2),
        (232, 236, 241),
    )
    for panel_index, panel in enumerate(panels):
        x = (panel_index % 2) * (PANEL_WIDTH + gap)
        y = (panel_index // 2) * (PANEL_HEIGHT + gap)
        overview.paste(panel, (x, y))
    overview.save(OUTPUT_DIR / "selected_representative_grid.png")


def main() -> None:
    labels = read_labels()
    ocr_results = read_ocr_results()

    expected_ocr_rows = 300 * 3
    if len(ocr_results) != expected_ocr_rows:
        raise ValueError(
            f"Expected {expected_ocr_rows} OCR rows; found {len(ocr_results)}"
        )

    created = 0
    for difficulty in DIFFICULTIES:
        for index in range(1, 101):
            panel = create_comparison(
                difficulty, index, labels[(difficulty, index)], ocr_results
            )
            output_path = (
                OUTPUT_DIR
                / difficulty
                / f"{difficulty}_{index:03d}_comparison.png"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            panel.save(output_path)
            created += 1

    selected = select_representative_samples(labels, ocr_results)
    write_selection_csv(selected)
    create_overview(selected, labels, ocr_results)

    print(f"Created {created} individual comparison images in: {OUTPUT_DIR}")
    print(f"Created presentation overview: {OUTPUT_DIR / 'selected_representative_grid.png'}")
    print(f"Saved transparent selection criteria: {SELECTION_CSV}")


if __name__ == "__main__":
    main()
