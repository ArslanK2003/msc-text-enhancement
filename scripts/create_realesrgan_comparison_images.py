from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


SUBSET_DIR = Path("datasets/textzoom_subset")
SR_DIR = Path("experiments/outputs/realesrgan")
OUTPUT_DIR = Path("experiments/visual_comparisons/realesrgan")

DIFFICULTIES = ["easy", "medium", "hard"]


def add_label(image, label):
    label_height = 30
    new_img = Image.new("RGB", (image.width, image.height + label_height), "white")
    new_img.paste(image, (0, label_height))

    draw = ImageDraw.Draw(new_img)
    draw.text((10, 8), label, fill="black")

    return new_img


def resize_for_display(image, target_height=120):
    ratio = target_height / image.height
    new_width = int(image.width * ratio)
    return image.resize((new_width, target_height), Image.BICUBIC)


def create_comparison(lr_path, sr_path, hr_path, output_path):
    lr = Image.open(lr_path).convert("RGB")
    sr = Image.open(sr_path).convert("RGB")
    hr = Image.open(hr_path).convert("RGB")

    lr = resize_for_display(lr)
    sr = resize_for_display(sr)
    hr = resize_for_display(hr)

    lr = add_label(lr, "LR input")
    sr = add_label(sr, "Real-ESRGAN SR")
    hr = add_label(hr, "HR ground truth")

    spacing = 20
    total_width = lr.width + sr.width + hr.width + spacing * 2
    total_height = max(lr.height, sr.height, hr.height)

    combined = Image.new("RGB", (total_width, total_height), "white")

    x = 0
    combined.paste(lr, (x, 0))
    x += lr.width + spacing
    combined.paste(sr, (x, 0))
    x += sr.width + spacing
    combined.paste(hr, (x, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path)


def main():
    for difficulty in DIFFICULTIES:
        lr_folder = SUBSET_DIR / difficulty / "lr"
        hr_folder = SUBSET_DIR / difficulty / "hr"
        sr_folder = SR_DIR / difficulty

        for lr_path in sorted(lr_folder.glob("*_lr.png")):
            index = lr_path.stem.split("_")[1]

            hr_path = hr_folder / f"{difficulty}_{index}_hr.png"
            sr_path = sr_folder / f"{difficulty}_{index}_lr_sr.png"

            if not hr_path.exists() or not sr_path.exists():
                print(f"Skipping missing pair for {lr_path.name}")
                continue

            output_path = OUTPUT_DIR / difficulty / f"{difficulty}_{index}_comparison.png"

            create_comparison(lr_path, sr_path, hr_path, output_path)
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()