# MSc Text Enhancement Dissertation

This repository contains implementation work for my MSc dissertation project:

**Deep Learning for Text Enhancement: Improving Text Readability in Degraded Images using Super-Resolution Methods**

## Project Aim

The aim of this project is to investigate deep learning methods for improving the readability of degraded scene text images using super-resolution techniques.

## W29 experiment status

- Balanced TextZoom diagnostic subset: 300 official test pairs (100 easy, 100 medium and 100 hard).
- General SR baseline: Real-ESRGAN completed for all 300 LR images.
- OCR evaluation: LR, Real-ESRGAN SR and HR completed with EasyOCR.
- Full-reference evaluation: PSNR and SSIM completed for all 300 Real-ESRGAN outputs.
- Qualitative evaluation: 300 annotated LR | Real-ESRGAN SR | HR panels plus a balanced six-sample overview.
- Text-specific SR: the official TSRN core passes a modern PyTorch smoke test; trained weights are still required for inference.

## Reproduce the W29 summaries

Run these commands from the repository root:

```powershell
python scripts\compare_ocr_results_300.py
python scripts\evaluate_psnr_ssim_realesrgan_300.py
python scripts\create_realesrgan_300_comparison_images.py
python scripts\run_tsrn_textzoom_300.py --smoke-test --device cpu
```

Important CSV outputs are stored under `experiments/metrics/`. Datasets, generated images, model weights and cloned external repositories are intentionally ignored by Git.

## Current 300-sample headline results

| Measure | LR | Real-ESRGAN SR | HR |
|---|---:|---:|---:|
| Mean OCR confidence | 0.0903 | 0.1379 | 0.5236 |
| OCR detection rate | 33.0% | 31.0% | 82.0% |
| Exact-match accuracy | 3.3% | 4.0% | 39.3% |
| Mean character similarity | 14.38% | 13.76% | 62.60% |

Real-ESRGAN achieved an overall mean PSNR of 21.9638 dB and SSIM of 0.6336 after its 4x outputs were bicubically resized to the paired HR dimensions.
