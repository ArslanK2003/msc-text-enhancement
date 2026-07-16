"""Compare OCR readability for LR, Real-ESRGAN, TSRN and HR TextZoom images.

The script keeps the requested confidence table and also reports metrics that
can be checked against the TextZoom ground-truth labels.  It uses only the
Python standard library so it can run in the existing project environment.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path


LR_HR_CSV = Path("experiments/metrics/easyocr_textzoom_subset_300_results.csv")
REALESRGAN_CSV = Path("experiments/metrics/easyocr_realesrgan_300_results.csv")
TSRN_CSV = Path("experiments/metrics/easyocr_tsrn_300_results.csv")
LABELS_CSV = Path("datasets/textzoom_subset_300/textzoom_subset_300_labels.csv")

CONFIDENCE_OUTPUT_CSV = Path(
    "experiments/metrics/ocr_confidence_summary_all_methods_300.csv"
)
READABILITY_OUTPUT_CSV = Path(
    "experiments/metrics/ocr_readability_summary_all_methods_300.csv"
)

DIFFICULTIES = ("easy", "medium", "hard")
METHODS = ("lr", "sr_realesrgan", "sr_tsrn", "hr")
METHOD_LABELS = {
    "lr": "LR",
    "sr_realesrgan": "Real-ESRGAN SR",
    "sr_tsrn": "TSRN SR",
    "hr": "HR",
}


def normalise_text(value: str) -> str:
    """Match text case-insensitively and ignore spaces/punctuation."""

    return "".join(character for character in value.upper() if character.isalnum())


def levenshtein_distance(left: str, right: str) -> int:
    """Return the character-level edit distance between two strings."""

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
    """Return 1 - normalised edit distance, clipped to the range 0 to 1."""

    denominator = max(len(prediction), len(ground_truth), 1)
    score = 1.0 - levenshtein_distance(prediction, ground_truth) / denominator
    return max(0.0, score)


def sample_key(difficulty: str, image_name: str) -> tuple[str, int]:
    match = re.search(r"_(\d+)_", image_name)
    if not match:
        raise ValueError(f"Cannot find a sample index in image name: {image_name}")
    return difficulty.lower(), int(match.group(1))


def read_labels() -> dict[tuple[str, int], str]:
    labels = {}
    with LABELS_CSV.open("r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            key = (row["difficulty"].lower(), int(row["index"]))
            if key in labels:
                raise ValueError(f"Duplicate label row for {key}")
            labels[key] = row["label"]
    return labels


def read_ocr_rows(csv_path: Path) -> list[dict[str, object]]:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            try:
                confidence = float(row["confidence"] or 0)
            except ValueError as error:
                raise ValueError(
                    f"Invalid confidence in {csv_path}: {row['confidence']!r}"
                ) from error

            rows.append(
                {
                    "key": sample_key(row["difficulty"], row["image_name"]),
                    "difficulty": row["difficulty"].lower(),
                    "method": row["image_type"].lower(),
                    "prediction": row["ocr_text"],
                    "confidence": confidence,
                    "no_text": row["notes"].strip().lower() == "no text detected",
                }
            )
    return rows


def validate_rows(rows: list[dict[str, object]], labels: dict[tuple[str, int], str]) -> None:
    seen = set()
    counts = defaultdict(int)

    for row in rows:
        key = (row["key"], row["method"])
        if key in seen:
            raise ValueError(f"Duplicate OCR result for {key}")
        seen.add(key)

        if row["key"] not in labels:
            raise ValueError(f"OCR result has no ground-truth label: {row['key']}")
        if row["method"] not in METHODS:
            raise ValueError(f"Unexpected image type: {row['method']}")
        counts[(row["difficulty"], row["method"])] += 1

    expected = 100
    problems = [
        f"{difficulty}/{method}: {counts[(difficulty, method)]} rows"
        for difficulty in DIFFICULTIES
        for method in METHODS
        if counts[(difficulty, method)] != expected
    ]
    if problems:
        raise ValueError("Expected 100 results per group; " + ", ".join(problems))


def calculate_group_metrics(
    rows: list[dict[str, object]], labels: dict[tuple[str, int], str]
) -> dict[str, float | int]:
    confidences = [float(row["confidence"]) for row in rows]
    no_text_count = sum(bool(row["no_text"]) for row in rows)
    exact_matches = 0
    similarities = []

    for row in rows:
        prediction = normalise_text(str(row["prediction"]))
        ground_truth = normalise_text(labels[row["key"]])
        exact_matches += prediction == ground_truth
        similarities.append(character_similarity(prediction, ground_truth))

    count = len(rows)
    return {
        "num_images": count,
        "average_confidence": sum(confidences) / count,
        "no_text_detected": no_text_count,
        "detection_rate": (count - no_text_count) / count,
        "exact_match_accuracy": exact_matches / count,
        "mean_character_similarity": sum(similarities) / count,
    }


def write_confidence_summary(group_metrics: dict[tuple[str, str], dict]) -> None:
    rows = []
    for difficulty in (*DIFFICULTIES, "overall"):
        row = {"difficulty": difficulty}
        for method in METHODS:
            row[method] = round(
                group_metrics[(difficulty, method)]["average_confidence"], 4
            )
        rows.append(row)

    CONFIDENCE_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with CONFIDENCE_OUTPUT_CSV.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["difficulty", "lr", "sr_realesrgan", "sr_tsrn", "hr"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_readability_summary(group_metrics: dict[tuple[str, str], dict]) -> None:
    fieldnames = [
        "difficulty",
        "method",
        "num_images",
        "average_confidence",
        "no_text_detected",
        "detection_rate",
        "exact_match_accuracy",
        "mean_character_similarity",
    ]
    rows = []
    for difficulty in (*DIFFICULTIES, "overall"):
        for method in METHODS:
            metrics = group_metrics[(difficulty, method)]
            rows.append(
                {
                    "difficulty": difficulty,
                    "method": method,
                    "num_images": metrics["num_images"],
                    "average_confidence": round(metrics["average_confidence"], 4),
                    "no_text_detected": metrics["no_text_detected"],
                    "detection_rate": round(metrics["detection_rate"], 4),
                    "exact_match_accuracy": round(metrics["exact_match_accuracy"], 4),
                    "mean_character_similarity": round(
                        metrics["mean_character_similarity"], 4
                    ),
                }
            )

    READABILITY_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with READABILITY_OUTPUT_CSV.open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(group_metrics: dict[tuple[str, str], dict]) -> None:
    print("\nOCR confidence summary")
    print("Difficulty | LR | Real-ESRGAN SR | TSRN SR | HR")
    for difficulty in (*DIFFICULTIES, "overall"):
        values = [
            group_metrics[(difficulty, method)]["average_confidence"]
            for method in METHODS
        ]
        print(
            f"{difficulty.title():10} | "
            + " | ".join(f"{value:.4f}" for value in values)
        )

    print("\nGround-truth readability summary")
    print("Difficulty | Method | Detection | Exact match | Character similarity")
    for difficulty in (*DIFFICULTIES, "overall"):
        for method in METHODS:
            metrics = group_metrics[(difficulty, method)]
            print(
                f"{difficulty.title():10} | {METHOD_LABELS[method]:16} | "
                f"{metrics['detection_rate']:.1%} | "
                f"{metrics['exact_match_accuracy']:.1%} | "
                f"{metrics['mean_character_similarity']:.1%}"
            )


def main() -> None:
    labels = read_labels()
    rows = (
        read_ocr_rows(LR_HR_CSV)
        + read_ocr_rows(REALESRGAN_CSV)
        + read_ocr_rows(TSRN_CSV)
    )
    validate_rows(rows, labels)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["difficulty"], row["method"])].append(row)

    group_metrics = {}
    for difficulty in DIFFICULTIES:
        for method in METHODS:
            group_metrics[(difficulty, method)] = calculate_group_metrics(
                grouped[(difficulty, method)], labels
            )

    for method in METHODS:
        overall_rows = [row for row in rows if row["method"] == method]
        group_metrics[("overall", method)] = calculate_group_metrics(
            overall_rows, labels
        )

    write_confidence_summary(group_metrics)
    write_readability_summary(group_metrics)
    print_summary(group_metrics)
    print(f"\nSaved: {CONFIDENCE_OUTPUT_CSV}")
    print(f"Saved: {READABILITY_OUTPUT_CSV}")


if __name__ == "__main__":
    main()

