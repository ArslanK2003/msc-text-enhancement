from pathlib import Path
from io import BytesIO
import csv

import lmdb
from PIL import Image


BASE_LMDB_DIR = Path("datasets/textzoom_lmdb")
OUTPUT_DIR = Path("datasets/textzoom_subset")
DIFFICULTIES = ["easy", "medium", "hard"]
SAMPLES_PER_SPLIT = 5


def buf_to_pil(txn, key: bytes, image_type: str = "RGB") -> Image.Image:
    image_buffer = txn.get(key)
    if image_buffer is None:
        raise KeyError(f"Missing key: {key}")

    buffer = BytesIO()
    buffer.write(image_buffer)
    buffer.seek(0)

    return Image.open(buffer).convert(image_type)


def extract_split(difficulty: str) -> list[dict]:
    lmdb_path = BASE_LMDB_DIR / difficulty
    output_path = OUTPUT_DIR / difficulty

    lr_output = output_path / "lr"
    hr_output = output_path / "hr"

    lr_output.mkdir(parents=True, exist_ok=True)
    hr_output.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    rows = []

    with env.begin(write=False) as txn:
        n_samples_raw = txn.get(b"num-samples")
        if n_samples_raw is None:
            raise KeyError(f"Could not find num-samples in {lmdb_path}")

        n_samples = int(n_samples_raw)
        limit = min(SAMPLES_PER_SPLIT, n_samples)

        print(f"{difficulty}: found {n_samples} samples, extracting {limit}")

        for index in range(1, limit + 1):
            label_key = f"label-{index:09d}".encode()
            lr_key = f"image_lr-{index:09d}".encode()
            hr_key = f"image_hr-{index:09d}".encode()

            label_raw = txn.get(label_key)
            label = label_raw.decode("utf-8") if label_raw else ""

            lr_image = buf_to_pil(txn, lr_key)
            hr_image = buf_to_pil(txn, hr_key)

            lr_filename = f"{difficulty}_{index:02d}_lr.png"
            hr_filename = f"{difficulty}_{index:02d}_hr.png"

            lr_path = lr_output / lr_filename
            hr_path = hr_output / hr_filename

            lr_image.save(lr_path)
            hr_image.save(hr_path)

            rows.append(
                {
                    "difficulty": difficulty,
                    "index": index,
                    "label": label,
                    "lr_image": str(lr_path),
                    "hr_image": str(hr_path),
                }
            )

    env.close()
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for difficulty in DIFFICULTIES:
        all_rows.extend(extract_split(difficulty))

    csv_path = OUTPUT_DIR / "textzoom_subset_labels.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["difficulty", "index", "label", "lr_image", "hr_image"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. Saved subset labels to: {csv_path}")


if __name__ == "__main__":
    main()