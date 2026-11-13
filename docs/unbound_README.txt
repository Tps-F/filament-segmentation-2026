Own S-full + deterministic B5/Conv semantic boundary + frozen q20, portable v2

STATUS: PREPARATION ONLY. Final S20 weights, canonical candidate and completion
hashes are unbound. This package refuses execution until the author binds
actual completed, reviewed artifacts. No GPU TEST reproduction has run.

This package is separate from the preserved original S/pair/three packages.
The old pair's fresh GPU reproduction failed exact CSV identity (1311 versus
1313 masks); its artifacts remain evidence. A later fixed one-image probe
found exact repeated deterministic B5/Conv outputs on the same NVIDIA GB10,
at both float32 and float16. That is not proof of full180 or crossGPU parity.

RUNTIME
Linux/POSIX, Python3.12 and the exact requirements.txt CUDA13 environment.
A native BF16 CUDA GPU is required. NVIDIA GB10 is the reference device.
Kaggle T4 lacks native BF16 and is rejected; emulation or switching to FP16
does not reproduce this recipe. Other GPU byte parity is not established.
No weights or data are downloaded by run.py. Supply the official180 TEST JPEGs.

PROCESS ISOLATION
run.py remains CPU-only and starts three fresh exec-created Python processes:
  1. YOLO S20: original unset CUBLAS_WORKSPACE_CONFIG, cudnn benchmark=False,
     default non-deterministic mode, default FP32 model.predict; native1536,
     raw confidence.03, boxNMS.5, retina masks, max100, no TTA.
  2. B5 semantic worker: CUBLAS_WORKSPACE_CONFIG=:4096:8 before torch import;
     cudnn benchmark=False, cudnn deterministic=True, strict deterministic
     algorithms, nativeBF16, channels_last, TF32 settings recorded in recipe.
  3. ConvNeXt semantic worker: a separate fresh process with the same policy.
The environment sanitizer removes inherited CUBLAS/TF32/allocator overrides.
The YOLO worker does not inherit semantic backend settings. Each worker exits
before the next begins. The CPU parent computes the mean, geometry and q20.

NUMERICAL AND INSTANCE RECIPE
Each semantic model: native2048 image, tile768/stride512/batch4/TTA4, independently
rounded to float16. Equal float32 weighted mean is then rounded to float16.
Do not average before member rounding. Boundary is .75semantic + .25Gaussian
of the original mask, sigma3/kernel25/zero border, threshold.5 inside ellipseR12,
minarea5, pre/post confidence-ordered maskNMS>.5. Natural overlaps may remain.
The original detector confidence is the27th feature. The frozen quality model
05ef8e57... predicts clipped[0,1] scores, selected at q>=.20; no extra detector
confidence.20 filter. No fitting, annotations, threshold search or upload.

AFTER ACTUAL AUTHOR BINDING
  python run.py                         # verify and show the plan only
  CUDA_VISIBLE_DEVICES='' python test_release.py
  python run.py --execute --test-dir /path/to/official/test/test_images \
    --output /path/to/new/output
Or use notebooks/reproduce_own_s_full_deterministic_v2.ipynb. It explicitly
spawns the CLI parent; setting RUN_FULL=True is required for full execution.
The whole run is bounded to3hours, with process-group cleanup on failures,
timeout and catchable termination. SIGKILL cannot be caught. Existing output
directories are refused; partial runs are not automatically retried.

Every detector raw JSON and member/equal probability file must match the
bound canonical per-image hash, followed by exact candidate CSV identity.
A mismatch stops the run and preserves artifacts. A completed CSV is not
claimed reproduced unless all checks pass in that actual run. This package
does not claim the preserved YOLO runtime itself is deterministic.

AUTHOR-ONLY BINDING
build_local.py bind --review <root-review.json> is a CPU-only operation. It
requires reviewed actual final20 training/capture and deterministic candidate
completion chains. It exports exact EMA tensor bytes without dtype conversion,
checks every tensor, strict native YOLO CPU128 output parity, and64MiB archive
roundtrip. It then binds actual raw/semantic/candidate hashes. No export runs
during preparation. No repository manifest paths are required by the portable
runner; provenance paths in documentation are historical evidence only.
