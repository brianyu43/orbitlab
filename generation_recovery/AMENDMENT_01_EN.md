# Diversity-Evaluation Amendment Based on Execution-Time Diagnostics

2026-09-23. The original [plan](../planning/generation_recovery_v1/PLAN_EN.md) and `protocol.json` are preserved as the design draft from that time. This amendment reflects P0 results from reading actual clean images. It does not adjust evaluation thresholds after inspecting training results.

A zigzag (shape 3) has an outline indistinguishable after a 180-degree rotation. Counting four orientation bins for every shape, as the original plan proposed, would misjudge orientation even on clean zigzags. On 2,048 rotated clean development images tested with fixed templates, reading accuracy was 100% for each shape when using four bins for shapes 0–2 and two bins for shape 3. We therefore fix the number of orientation bins by shape at 4/4/4/2. [Reader and bin boundaries](reports/p0_diversity_reader_lock_v1.json)

The radius of a generated shape is not directly known, so size is measured by the fixed reader's bounding-box `scale`. The three-bin boundaries for center x, center y, and size were fixed using clean development images. We also record the share of positions and sizes outside the normal range. We will not choose new bins after inspecting the final model.

In the first diversity comparison, we match 80 clean images and 80 generated images in each of 24 conditions. We compare the marginal distributions of four attributes and inspect the mean per-condition bin occupancy, total variation (TV) distance, and out-of-range rate. The draft criteria are mean occupancy of at least 90%, mean TV at most 0.20, and mean out-of-range rate at most 10%. The results table retains every per-condition value. Where fewer than 32 images pass strictly in a combination, diversity among successful images is marked inconclusive and is checked separately in final confirmation.

This is a **correction of the measurement definition**. The existing IoU 0.8 success criterion is unchanged. Numerical checks made before training the new model are recorded in the [P0 results](reports/p0_data_c4_v1.json) and [clean-renderer evaluation](reports/p0_clean_renderer_v1.json).
