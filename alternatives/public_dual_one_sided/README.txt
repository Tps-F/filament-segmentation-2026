One-sided absent-model correction to HDJoJo's public dual-YOLO filament method

Algorithm attribution: https://www.kaggle.com/code/hdjojo/solar-filament-segmentation
This package contains an independent implementation of the published method,
not a redistribution of that notebook. Its original notebook license is not
asserted here. The exact source and model-version identities are recorded in
the recipe and provenance metadata.

The method keeps YOLO26x and YOLOv8x resident, predicts each image at2048,
confidence0.5, boxIoU0, max_det100 and the pinned runtime's default retina_masks
False. It concatenates26x first, uses the source's two default NumPy descending
sorts, maskNMS at0.5, ordered occupied-pixel subtraction, nearest native resize,
minimum5pixels and gapped IDs. An absent model contributes zero rows; the other model is retained unchanged.
The image stays empty when both models are absent. This is the only scientific
change from the literal algorithm.
No threshold search, calibration, validation or training occurs.

Run notebook_execute.py --execute with explicit --test-dir, --output, --python,
--entry-epoch and --deadline-epoch. Use the exact isolated interpreter described
by runtime_lock.json. The entire fresh attempt is bounded by2700seconds and
must regenerate both models' predictions on all180 external official JPEGs.
The package restores only its own pinned public checkpoint pieces. It performs
no model downloads or package installation. No image or annotation is shipped.

The explicit strict local backend is declared in recipe.json. Unsupported
deterministic operations stop the attempt; no backend or precision retry occurs.
Each pass seals its own raw masks/confidences/source IDs. A separate CPU process
reconstructs that pass's literal output using RLE integer intersections and
first-owner pixel labels, checking native shapes, gapped IDs and zero shared
pixels. The final ordered native rows and CSV must match the canonical result
byte for byte. Raw confidences and intermediate identities are not claimed to
match between passes; no floating-point tolerance is used.
This proves local pinned-runtime reproduction, not the unknown historical
Kaggle environment, a fresh package installation or a public-score guarantee.
There is no upload or submission command.

