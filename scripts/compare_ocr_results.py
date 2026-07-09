from pathlib import Path
import csv
from collections import defaultdict

LR_HR_CSV = Path("experiments/metrics/easyocr_textzoom_subset_results.csv")
SR_CSV = Path("experiments/metrics/easyocr_realesrgan_results.csv")
OUTPUT_CSV = Path("experiments/metrics/ocr_confidence_summary.csv")


def read_rows(csv_path):
    rows = []

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                confidence = float(row["confidence"])
            except ValueError:
                confidence = 0.0

            rows.append({
                "difficulty": row["difficulty"],
                "image_type": row["image_type"],
                "confidence": confidence
            })

    return rows


def main():
    all_rows = read_rows(LR_HR_CSV) + read_rows(SR_CSV)

    grouped = defaultdict(list)

    for row in all_rows:
        key = (row["difficulty"], row["image_type"])
        grouped[key].append(row["confidence"])

    difficulties = ["easy", "medium", "hard"]
    image_types = ["lr", "sr_realesrgan", "hr"]

    summary_rows = []

    for difficulty in difficulties:
        summary_row = {"difficulty": difficulty}

        for image_type in image_types:
            values = grouped.get((difficulty, image_type), [])

            if values:
                average_confidence = sum(values) / len(values)
                summary_row[image_type] = round(average_confidence, 4)
            else:
                summary_row[image_type] = ""

        summary_rows.append(summary_row)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        fieldnames = ["difficulty", "lr", "sr_realesrgan", "hr"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nOCR Confidence Summary")
    print("----------------------")
    print("Difficulty | LR | Real-ESRGAN SR | HR")

    for row in summary_rows:
        print(
            f"{row['difficulty']} | "
            f"{row['lr']} | "
            f"{row['sr_realesrgan']} | "
            f"{row['hr']}"
        )

    print(f"\nSaved summary to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()