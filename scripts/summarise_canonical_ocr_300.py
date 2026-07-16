"""Summarise controlled EasyOCR results at the common 128 x 32 resolution."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path


OCR_CSV = Path(
    "experiments/metrics/easyocr_canonical_all_methods_300_results.csv"
)

LABELS_CSV = Path(
    "datasets/textzoom_subset_300/textzoom_subset_300_labels.csv"
)

CONFIDENCE_OUTPUT_CSV = Path(
    "experiments/metrics/canonical_ocr_confidence_summary_300.csv"
)

READABILITY_OUTPUT_CSV = Path(
    "experiments/metrics/canonical_ocr_readability_summary_300.csv"
)

DIFFICULTIES = ("easy", "medium", "hard")

METHODS = (
    "bicubic",
    "sr_realesrgan",
    "sr_tsrn",
    "hr",
)

METHOD_LABELS = {
    "bicubic": "Bicubic",
    "sr_realesrgan": "Real-ESRGAN",
    "sr_tsrn": "TSRN",
    "hr": "HR",
}


def normalise_text(value: str) -> str:
    return "".join(
        character
        for character in value.upper()
        if character.isalnum()
    )


def levenshtein_distance(
    left: str,
    right: str,
) -> int:
    if len(left) < len(right):
        left, right = right, left

    if not right:
        return len(left)

    previous = list(
        range(len(right) + 1)
    )

    for left_index, left_character in enumerate(
        left,
        start=1,
    ):
        current = [left_index]

        for right_index, right_character in enumerate(
            right,
            start=1,
        ):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (
                        left_character
                        != right_character
                    ),
                )
            )

        previous = current

    return previous[-1]


def character_similarity(
    prediction: str,
    ground_truth: str,
) -> float:
    denominator = max(
        len(prediction),
        len(ground_truth),
        1,
    )

    similarity = (
        1.0
        - levenshtein_distance(
            prediction,
            ground_truth,
        )
        / denominator
    )

    return max(0.0, similarity)


def sample_key(
    difficulty: str,
    image_name: str,
) -> tuple[str, int]:
    match = re.search(
        r"_(\d+)_",
        image_name,
    )

    if not match:
        raise ValueError(
            f"Cannot find sample index in: {image_name}"
        )

    return (
        difficulty.lower(),
        int(match.group(1)),
    )


def read_labels() -> dict[tuple[str, int], str]:
    labels = {}

    with LABELS_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        for row in csv.DictReader(csv_file):
            key = (
                row["difficulty"].lower(),
                int(row["index"]),
            )

            if key in labels:
                raise ValueError(
                    f"Duplicate label: {key}"
                )

            labels[key] = row["label"]

    if len(labels) != 300:
        raise ValueError(
            f"Expected 300 labels; found {len(labels)}"
        )

    return labels


def read_ocr_rows() -> list[dict[str, object]]:
    rows = []

    with OCR_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        for row in csv.DictReader(csv_file):
            try:
                confidence = float(
                    row["confidence"] or 0
                )
            except ValueError as error:
                raise ValueError(
                    f"Invalid confidence: "
                    f"{row['confidence']!r}"
                ) from error

            rows.append(
                {
                    "key": sample_key(
                        row["difficulty"],
                        row["image_name"],
                    ),
                    "difficulty": (
                        row["difficulty"].lower()
                    ),
                    "method": (
                        row["image_type"].lower()
                    ),
                    "prediction": row["ocr_text"],
                    "confidence": confidence,
                    "no_text": (
                        row["notes"]
                        .strip()
                        .lower()
                        == "no text detected"
                    ),
                }
            )

    return rows


def validate_rows(
    rows: list[dict[str, object]],
    labels: dict[tuple[str, int], str],
) -> None:
    seen = set()
    counts = defaultdict(int)

    for row in rows:
        unique_key = (
            row["key"],
            row["method"],
        )

        if unique_key in seen:
            raise ValueError(
                f"Duplicate OCR result: {unique_key}"
            )

        seen.add(unique_key)

        if row["key"] not in labels:
            raise ValueError(
                f"No ground-truth label for: "
                f"{row['key']}"
            )

        if row["difficulty"] not in DIFFICULTIES:
            raise ValueError(
                f"Unexpected difficulty: "
                f"{row['difficulty']}"
            )

        if row["method"] not in METHODS:
            raise ValueError(
                f"Unexpected method: "
                f"{row['method']}"
            )

        counts[
            (
                row["difficulty"],
                row["method"],
            )
        ] += 1

    problems = []

    for difficulty in DIFFICULTIES:
        for method in METHODS:
            count = counts[
                (
                    difficulty,
                    method,
                )
            ]

            if count != 100:
                problems.append(
                    f"{difficulty}/{method}: "
                    f"{count}"
                )

    if problems:
        raise ValueError(
            "Expected 100 rows per group; "
            + ", ".join(problems)
        )

    if len(rows) != 1200:
        raise ValueError(
            f"Expected 1,200 rows; "
            f"found {len(rows)}"
        )


def calculate_metrics(
    rows: list[dict[str, object]],
    labels: dict[tuple[str, int], str],
) -> dict[str, float | int]:
    confidences = [
        float(row["confidence"])
        for row in rows
    ]

    no_text_count = sum(
        bool(row["no_text"])
        for row in rows
    )

    exact_matches = 0
    similarities = []

    for row in rows:
        prediction = normalise_text(
            str(row["prediction"])
        )

        ground_truth = normalise_text(
            labels[row["key"]]
        )

        exact_matches += (
            prediction == ground_truth
        )

        similarities.append(
            character_similarity(
                prediction,
                ground_truth,
            )
        )

    count = len(rows)

    return {
        "num_images": count,
        "average_confidence": (
            sum(confidences) / count
        ),
        "no_text_detected": no_text_count,
        "detection_rate": (
            count - no_text_count
        ) / count,
        "exact_match_accuracy": (
            exact_matches / count
        ),
        "mean_character_similarity": (
            sum(similarities) / count
        ),
    }


def build_summary(
    rows: list[dict[str, object]],
    labels: dict[tuple[str, int], str],
) -> dict[tuple[str, str], dict]:
    grouped = defaultdict(list)

    for row in rows:
        grouped[
            (
                row["difficulty"],
                row["method"],
            )
        ].append(row)

    metrics = {}

    for difficulty in DIFFICULTIES:
        for method in METHODS:
            metrics[
                (
                    difficulty,
                    method,
                )
            ] = calculate_metrics(
                grouped[
                    (
                        difficulty,
                        method,
                    )
                ],
                labels,
            )

    for method in METHODS:
        method_rows = [
            row
            for row in rows
            if row["method"] == method
        ]

        metrics[
            (
                "overall",
                method,
            )
        ] = calculate_metrics(
            method_rows,
            labels,
        )

    return metrics


def write_confidence_summary(
    metrics: dict[tuple[str, str], dict],
) -> None:
    rows = []

    for difficulty in (
        *DIFFICULTIES,
        "overall",
    ):
        row = {
            "difficulty": difficulty,
        }

        for method in METHODS:
            row[method] = round(
                metrics[
                    (
                        difficulty,
                        method,
                    )
                ]["average_confidence"],
                4,
            )

        rows.append(row)

    CONFIDENCE_OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CONFIDENCE_OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "difficulty",
                *METHODS,
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def write_readability_summary(
    metrics: dict[tuple[str, str], dict],
) -> None:
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

    for difficulty in (
        *DIFFICULTIES,
        "overall",
    ):
        for method in METHODS:
            values = metrics[
                (
                    difficulty,
                    method,
                )
            ]

            rows.append(
                {
                    "difficulty": difficulty,
                    "method": method,
                    "num_images": (
                        values["num_images"]
                    ),
                    "average_confidence": round(
                        values[
                            "average_confidence"
                        ],
                        4,
                    ),
                    "no_text_detected": (
                        values[
                            "no_text_detected"
                        ]
                    ),
                    "detection_rate": round(
                        values[
                            "detection_rate"
                        ],
                        4,
                    ),
                    "exact_match_accuracy": round(
                        values[
                            "exact_match_accuracy"
                        ],
                        4,
                    ),
                    "mean_character_similarity": round(
                        values[
                            "mean_character_similarity"
                        ],
                        4,
                    ),
                }
            )

    with READABILITY_OUTPUT_CSV.open(
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


def print_summary(
    metrics: dict[tuple[str, str], dict],
) -> None:
    print(
        "\nControlled OCR confidence summary"
    )

    print(
        "Difficulty | Bicubic | "
        "Real-ESRGAN | TSRN | HR"
    )

    for difficulty in (
        *DIFFICULTIES,
        "overall",
    ):
        values = [
            metrics[
                (
                    difficulty,
                    method,
                )
            ]["average_confidence"]
            for method in METHODS
        ]

        print(
            f"{difficulty.title():10} | "
            + " | ".join(
                f"{value:.4f}"
                for value in values
            )
        )

    print(
        "\nControlled ground-truth "
        "readability summary"
    )

    print(
        "Difficulty | Method       | "
        "Detection | Exact match | "
        "Character similarity"
    )

    for difficulty in (
        *DIFFICULTIES,
        "overall",
    ):
        for method in METHODS:
            values = metrics[
                (
                    difficulty,
                    method,
                )
            ]

            print(
                f"{difficulty.title():10} | "
                f"{METHOD_LABELS[method]:12} | "
                f"{values['detection_rate']:.1%} | "
                f"{values['exact_match_accuracy']:.1%} | "
                f"{values['mean_character_similarity']:.1%}"
            )


def main() -> None:
    labels = read_labels()
    rows = read_ocr_rows()

    validate_rows(
        rows,
        labels,
    )

    metrics = build_summary(
        rows,
        labels,
    )

    write_confidence_summary(metrics)
    write_readability_summary(metrics)
    print_summary(metrics)

    print()
    print(
        "Saved:",
        CONFIDENCE_OUTPUT_CSV,
    )

    print(
        "Saved:",
        READABILITY_OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()
