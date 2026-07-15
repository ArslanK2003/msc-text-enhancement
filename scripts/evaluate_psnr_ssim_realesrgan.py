from pathlib import Path
import csv
from PIL import Image
import math

import numpy as np


HR_DIR = Path("datasets/textzoom_subset")
SR_DIR = Path("experiments/outputs/realesrgan")
OUTPUT_CSV = Path("experiments/metrics/psnr_ssim_realesrgan_results.csv")

DIFFICULTIES = ["easy", "medium", "hard"]


def calculate_psnr(img1, img2):
    img1 = np.array(img1).astype(np.float64)
    img2 = np.array(img2).astype(np.float64)

    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return 100.0

    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim_simple(img1, img2):
    # Simple SSIM approximation for initial experiment.
    # This is enough for early W29 progress, but can later be replaced with skimage.metrics.structural_similarity.
    img1 = np.array(img1).astype(np.float64)
    img2 = np.array(img2).astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()

    sigma1 = img1.var()
    sigma2 = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2)
    )

    return ssim


def resize_to_match(sr_img, hr_img):
    if sr_img.size != hr_img.size:
        sr_img = sr_img.resize(hr_img.size, Image.BICUBIC)
    return sr_img, hr_img


def main():
    rows = []

    for difficulty in DIFFICULTIES:
        hr_folder = HR_DIR / difficulty / "hr"
        sr_folder = SR_DIR / difficulty

        for hr_path in sorted(hr_folder.glob("*_hr.png")):
            index = hr_path.stem.split("_")[1]

            sr_name = f"{difficulty}_{index}_lr_sr.png"
            sr_path = sr_folder / sr_name

            if not sr_path.exists():
                print(f"Missing SR image: {sr_path}")
                continue

            hr_img = Image.open(hr_path).convert("RGB")
            sr_img = Image.open(sr_path).convert("RGB")

            sr_img, hr_img = resize_to_match(sr_img, hr_img)

            psnr = calculate_psnr(sr_img, hr_img)
            ssim = calculate_ssim_simple(sr_img, hr_img)

            rows.append({
                "difficulty": difficulty,
                "image_name": sr_name,
                "psnr": round(psnr, 4),
                "ssim": round(ssim, 4)
            })

            print(f"{difficulty} {sr_name} | PSNR: {psnr:.4f} | SSIM: {ssim:.4f}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        fieldnames = ["difficulty", "image_name", "psnr", "ssim"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()