# TSRN attempt - W29

## Status

The official TextZoom/TSRN repository was inspected and the core TSRN model was
tested locally.  The model builds and completes a forward pass with the installed
PyTorch 2.12.1 CPU build.

- Input shape tested: `1 x 4 x 16 x 64` (RGB plus text mask).
- Output shape obtained: `1 x 4 x 32 x 128`.
- Model size: 2,681,677 parameters.
- Result: core-model compatibility smoke test passed.

This is not yet a completed TSRN experimental result.  Random weights were used
only to test compatibility.

## Confirmed blockers

- No trained TSRN `.pth` checkpoint is present in the project.
- The official repository README does not provide a trained TSRN checkpoint.
- The official environment is old: PyTorch 1.0.1 to below 1.6.0.
- This computer has a CPU-only PyTorch build, so training the official 500-epoch
  setting locally is not practical.

## Ready inference command

After placing a compatible checkpoint outside Git, run from the project root:

```powershell
python scripts\run_tsrn_textzoom_300.py `
  --checkpoint C:\path\to\model_best.pth `
  --device cuda
```

Compatibility can be checked without weights:

```powershell
python scripts\run_tsrn_textzoom_300.py --smoke-test --device cpu
```

The inference script will create 100 easy, 100 medium and 100 hard outputs under
`experiments/outputs/textzoom_tsrn_300/`.  That folder and model weights remain
ignored by Git.

## Honest meeting wording

"The official TSRN model has been integrated far enough to pass a modern PyTorch
forward test.  Full inference is blocked by the missing trained TSRN checkpoint;
the next action is checkpoint acquisition or GPU training.  I have not counted
this as a completed SOTA experiment."
