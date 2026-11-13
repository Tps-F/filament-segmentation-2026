Own S20 + deterministic B5/Conv + fixed q20 + disjoint export (v3.2)

STATUS: Derived runner PREPARED, no v3.2 GPU execution or cross-hardware equality
claim. Actual weights and reference candidate remain bound to the unchanged
v3.1 sources. Canonical submitted CSV SHA256:
9dd513669eee8b12910e001732eea9fdd0f8a8adeb9eacab4421a544da5d9c3c.
The v3.1 package, its actual runs, failures and verification proofs are preserved
separately and unchanged. Historical copied documents describe those versions.

Choose a reference mode explicitly for full inference:

  python run.py --execute --reference-mode strict \
    --test-dir /official/test/test_images --output /new/strict-output

  python run.py --execute --reference-mode report-only \
    --test-dir /official/test/test_images --output /new/inference-output

The CLI defaults to strict and does not execute without --execute. Strict mode
is our exact reference-environment verification: any canonical detector,
semantic-member, equal-mean or final CSV hash mismatch stops the run. Report-only
is useful on another compatible native-BF16 GPU: these same comparisons are
recorded but do not abort inference. A successful report-only run does NOT certify
matching the submitted CSV on another GPU. Inspect reference_comparison.json and
reproduction_receipt.json: exact_csv_reproduction and all_reference_hashes_exact
are actual comparisons, never assumed. Output instance counts can differ.

Both modes perform exactly the same model calls, arithmetic, preprocessing,
precision, TTA, geometry, feature extraction, frozen quality q20, and final
ownership. Neither mode relaxes source/weight/input hashes, actual-output hash
integrity, native shapes/finiteness, native CUDA BF16 and backend settings,
original source identity/confidence/R12 support, CSV round trips, or zero overlap.
There is no weight/quality refit, threshold search, precision fallback, or silent
change of backend. Unsupported deterministic operations fail in both modes.

The unchanged boundary and quality stages select instances at q20. Afterward,
stable descending ORIGINAL detector confidence assigns pixels. Retained
occupancy is subtracted; a remainder under5pixels is dropped and claims no pixels.
Retained clipped parts remain one object. No new morphology, NMS, quality ordering
or splitting occurs. Final native area sum equals union exactly.

Install requirements.txt in a compatible environment; its existing PyTorch
cu130 extra-index URL is unchanged. The notebook has1Markdown+3code cells: set
its3paths and REFERENCE_MODE, then explicitly set RUN_FULL=True. It recommends
report-only for ordinary inference; set strict for canonical verification.
CPU contracts: python -m unittest test_release test_reference_mode .
Original author binding tools under build_local.py/docs are historical records,
not necessary runtime steps; docs/build_v32.py records this exact derivation.

Reference GPU: NVIDIA GB10. Native CUDA BF16 is required (capability>=8 and no
emulation); T4/FP16 fallback is unsupported. Detector workers start in a fresh
process with unset CUBLAS and FP32 defaults. The native Ultralytics constructor
sets CUBLAS_WORKSPACE_CONFIG=:4096:8; its deterministic flags remain false.
Each separate semantic worker sets its strict deterministic backend before Torch
import. The CPU parent never owns CUDA. All model workers are fresh processes,
with the original10800-second total budget, process-group cleanup, no automatic
retry, and an additional caller alarm mask through child publication/cleanup.

No images, labels, credentials, reference probability arrays, or canonical CSV
prediction rows are distributed here. Obtain official competition inputs using
your own authorized access. The runner reads no labels, downloads no weights,
trains/fits nothing, and uploads nothing. Cross-hardware numerical/metric parity,
a new installation, and notebook-kernel execution are not established by this
preparation. Exact source/archive copies and synthetic CPU contracts establish
only the stated guard/computation invariants.

See THIRD_PARTY_NOTICES.txt. The exact official Ultralytics8.4.211 AGPL-3.0 text
is retained under licenses/ with version/source/hash attribution for its code
and model component. Other components retain their upstream terms; no blanket
license assertion about all assets is made.
