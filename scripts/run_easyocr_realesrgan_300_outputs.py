from pathlib import Path
import csv
import easyocr

SR_DIR = Path("experiments/outputs/textzoom_realesrgan_300")
OUTPUT_CSV = Path("experiments/metrics/easyocr_realesrgan_300_results.csv")

DIFFICULTIES = ["easy", "medium", "hard"]


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    reader = easyocr.Reader(["en"], gpu=False)

    rows = []

    for difficulty in DIFFICULTIES:
        image_dir = SR_DIR / difficulty

        if not image_dir.exists():
            print(f"Missing folder: {image_dir}")
            continue

        image_paths = sorted(image_dir.glob("*.png"))

        for image_path in image_paths:
            print(f"Running OCR on: {image_path}")

            results = reader.readtext(str(image_path))

            if not results:
                rows.append({
                    "difficulty": difficulty,
                    "image_type": "sr_realesrgan",
                    "image_name": image_path.name,
                    "ocr_text": "",
                    "confidence": 0,
                    "notes": "No text detected"
                })
                continue

            detected_texts = []
            confidences = []

            for _, text, confidence in results:
                detected_texts.append(text)
                confidences.append(confidence)

            combined_text = " ".join(detected_texts)
            average_confidence = sum(confidences) / len(confidences)

            rows.append({
                "difficulty": difficulty,
                "image_type": "sr_realesrgan",
                "image_name": image_path.name,
                "ocr_text": combined_text,
                "confidence": round(average_confidence, 4),
                "notes": ""
            })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["difficulty", "image_type", "image_name", "ocr_text", "confidence", "notes"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()