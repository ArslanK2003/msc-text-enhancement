from pathlib import Path
import csv
from collections import defaultdict

INPUT_CSV = Path("experiments/metrics/easyocr_textzoom_subset_300_results.csv")
OUTPUT_CSV = Path("experiments/metrics/easyocr_textzoom_subset_300_summary.csv")


def main():
    grouped = defaultdict(list)
    counts = defaultdict(int)
    no_text_counts = defaultdict(int)

    with open(INPUT_CSV, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            difficulty = row["difficulty"]
            image_type = row["image_type"]
            key = (difficulty, image_type)

            confidence = float(row["confidence"])
            grouped[key].append(confidence)
            counts[key] += 1

            if row["notes"] == "No text detected":
                no_text_counts[key] += 1

    difficulties = ["easy", "medium", "hard"]
    image_types = ["lr", "hr"]

    rows = []

    for difficulty in difficulties:
        for image_type in image_types:
            key = (difficulty, image_type)
            values = grouped[key]

            avg_confidence = sum(values) / len(values) if values else 0

            rows.append({
                "difficulty": difficulty,
                "image_type": image_type,
                "num_images": counts[key],
                "average_confidence": round(avg_confidence, 4),
                "no_text_detected": no_text_counts[key]
            })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "difficulty",
            "image_type",
            "num_images",
            "average_confidence",
            "no_text_detected"
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nEasyOCR 300 Subset Summary")
    print("--------------------------")
    print("Difficulty | Type | Images | Avg Confidence | No Text Detected")

    for row in rows:
        print(
            f"{row['difficulty']} | "
            f"{row['image_type']} | "
            f"{row['num_images']} | "
            f"{row['average_confidence']} | "
            f"{row['no_text_detected']}"
        )

    print(f"\nSaved summary to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()