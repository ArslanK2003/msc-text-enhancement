"""Run the official TextZoom TSRN model on the balanced 300-image subset.

The official repository does not include a TSRN checkpoint.  Use --smoke-test
to verify model compatibility, or pass a trained checkpoint with --checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DIFFICULTIES = ("easy", "medium", "hard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official TSRN inference on the TextZoom 300 subset."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="TSRN .pth checkpoint containing state_dict_G, state_dict or raw weights.",
    )
    parser.add_argument(
        "--repo-src",
        type=Path,
        default=Path("repos/TextZoom/src"),
        help="Path to the official TextZoom src folder.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("datasets/textzoom_subset_300"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/outputs/textzoom_tsrn_300"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Use only RGB input. Official TSRN training normally uses --mask.",
    )
    parser.add_argument(
        "--no-stn",
        action="store_true",
        help="Build TSRN without STN parameters if the checkpoint was trained that way.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Test model construction and one forward pass without a checkpoint.",
    )
    return parser.parse_args()


def import_tsrn(repo_src: Path):
    repo_src = repo_src.resolve()
    if not (repo_src / "model" / "tsrn.py").exists():
        raise FileNotFoundError(f"Official TSRN source not found: {repo_src}")
    sys.path.insert(0, str(repo_src))
    from model.tsrn import TSRN

    return TSRN


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot see a CUDA GPU.")
    return torch.device(requested)


def build_model(TSRN, use_mask: bool, use_stn: bool, device: torch.device):
    model = TSRN(
        scale_factor=2,
        width=128,
        height=32,
        STN=use_stn,
        mask=use_mask,
        srb_nums=5,
        hidden_units=32,
    )
    return model.to(device).eval()


def extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary.")

    if "state_dict_G" in checkpoint:
        state_dict = checkpoint["state_dict_G"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError("Could not find model weights in the checkpoint.")
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    if not path.exists():
        raise FileNotFoundError(f"TSRN checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)


def preprocess(path: Path, use_mask: bool) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB").resize((64, 16), Image.Resampling.BICUBIC)

    rgb = np.asarray(image, dtype=np.float32) / 255.0
    channels = [torch.from_numpy(rgb.transpose(2, 0, 1))]

    if use_mask:
        grey = np.asarray(image.convert("L"), dtype=np.float32)
        threshold = float(grey.mean())
        mask = (grey <= threshold).astype(np.float32)
        channels.append(torch.from_numpy(mask).unsqueeze(0))

    return torch.cat(channels, dim=0)


def save_output(tensor: torch.Tensor, path: Path) -> None:
    rgb = tensor[:3].detach().cpu().clamp(0, 1)
    array = (rgb.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def collect_inputs(root: Path) -> list[tuple[str, Path]]:
    inputs = []
    for difficulty in DIFFICULTIES:
        paths = sorted((root / difficulty / "lr").glob("*_lr.png"))
        if len(paths) != 100:
            raise ValueError(
                f"Expected 100 {difficulty} LR images; found {len(paths)}"
            )
        inputs.extend((difficulty, path) for path in paths)
    return inputs


def run_inference(
    model: torch.nn.Module,
    inputs: list[tuple[str, Path]],
    output_root: Path,
    use_mask: bool,
    device: torch.device,
    batch_size: int,
) -> None:
    created = 0
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batch_items = inputs[start : start + batch_size]
            batch = torch.stack(
                [preprocess(path, use_mask) for _, path in batch_items]
            ).to(device)
            outputs = model(batch)

            for output, (difficulty, input_path) in zip(outputs, batch_items):
                output_name = input_path.name.replace("_lr.png", "_lr_tsrn.png")
                save_output(output, output_root / difficulty / output_name)
                created += 1
            print(f"Processed {created}/{len(inputs)}")

    if created != 300:
        raise ValueError(f"Expected 300 TSRN outputs; created {created}")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    device = choose_device(args.device)
    use_mask = not args.no_mask
    use_stn = not args.no_stn
    TSRN = import_tsrn(args.repo_src)
    model = build_model(TSRN, use_mask, use_stn, device)

    if args.smoke_test:
        channels = 4 if use_mask else 3
        sample = torch.rand(1, channels, 16, 64, device=device)
        with torch.inference_mode():
            output = model(sample)
        print(f"PyTorch: {torch.__version__}")
        print(f"Device: {device}")
        print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters())}")
        print(f"Input shape: {tuple(sample.shape)}")
        print(f"Output shape: {tuple(output.shape)}")
        print("Smoke test passed. This does not count as a trained TSRN result.")
        return

    if args.checkpoint is None:
        raise ValueError(
            "--checkpoint is required for inference. The official repository does "
            "not include a trained TSRN checkpoint. Use --smoke-test only for "
            "compatibility checking."
        )

    load_checkpoint(model, args.checkpoint, device)
    inputs = collect_inputs(args.input_root)
    run_inference(
        model,
        inputs,
        args.output_root,
        use_mask,
        device,
        args.batch_size,
    )
    print(f"Saved 300 TSRN outputs to: {args.output_root}")


if __name__ == "__main__":
    main()
