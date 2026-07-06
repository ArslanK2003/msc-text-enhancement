import easyocr
from pathlib import Path

image_paths = [
    Path("repos/TextZoom/real_sr.jpg"),
    Path("repos/TextZoom/sr_raw.jpg"),
    Path("repos/TextZoom/syn_real.jpg"),
]

reader = easyocr.Reader(["en"], gpu=False)

for image_path in image_paths:
    print(f"\n--- OCR for {image_path.name} ---")

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        continue

    results = reader.readtext(str(image_path))

    if not results:
        print("No text detected.")
        continue

    for box, text, confidence in results:
        print(f"Text: {text} | Confidence: {confidence:.2f}")