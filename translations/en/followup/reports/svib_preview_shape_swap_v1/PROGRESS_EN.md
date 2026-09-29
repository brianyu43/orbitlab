# SVIB public example follow-up experiment completed

2026-09-23. Completed separate verification for the learning and evaluation of 24 models, 500 pairs of public previews, and 32 baseline conditions. Played back 4,800 predictions for learning/verification/ID/common tests and 7,200 predictions for rotation input, and verified all pixel scores, aggregation, pair comparison, tables, and drawings.

The overall test pixel error for C4 was smaller than that of the general model, with alpha averages of 4/4 and initialization pairs of 11/12. However, both methods showed larger errors than simply maintaining the alpha averages of 4/4. The learning image error was small, while the hold-out image error increased significantly, revealing overfitting to small examples. Rotation consistency is not interpreted as an external transformation rule.

The external task for B06's reduction has been completed. The official overall benchmark, LPIPS, and meaningful Shape-Swap success rates were not achieved, and the existing OrbitLab checkpoint transfer was not executed, and it is not included in the achievement claims. The overall subsequent goals remain the comprehensive observation length, the full picture, the final research memo, the replication bundle, and the actual human judgment.

[Results and fixed examples](RESULTS_EN.md) · [Model verification](../../../../../followup/reports/svib_preview_shape_swap_v1/model_verification.json) · [Aggregation verification](../../../../../followup/reports/svib_preview_shape_swap_v1/aggregate_v1/verification.json) · [Report verification](../../../../../followup/reports/svib_preview_shape_swap_v1/report_verification.json) · [Execution records](../../../../../followup/reports/svib_preview_shape_swap_v1/execution_sessions.json)
