# R3 sampling amendment, before training

The first random-layout sampler could not fill all three contact strata at six objects within 20,000 candidates. The archived original protocol, code, completed data, and failure receipt are in `r3_sampling_attempt_v1/`. No R3 model was trained and no model-performance results were inspected.

The revised design uses explicit candidate families: slow/free flight; one disk directed toward a wall; and ordinary randomly moving disks for pair-contact examples. Actual simulator events still determine the accepted stratum over forecast steps 1–16. Rejections never change the requested stratum, quotas, contact definition or horizons. Slow-flight and wall families have smaller force/action magnitudes, so strata are controlled mechanism diagnostics, not an unbiased draw from one shared population. All compared models receive the identical retained episodes. Force/action distributions and candidate counts are retained. Development and confirmation use fresh data seeds; previous attempt data are excluded from training and model selection.

The physics, architectures, budgets and scoring gates are unchanged. The amended protocol will be frozen in a fresh `r3/` directory. The previous frozen record is not overwritten.
