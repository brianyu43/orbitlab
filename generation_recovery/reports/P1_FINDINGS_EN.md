# P1 Development Findings (Data Seed 770101, One Initialization)

Training and evaluation are complete. [Independent re-verification](../evaluation/p1_v1/verification.json) recomputed 20 outcomes for four flows, ten generation settings, and two decoders from raw data. These are **development-data results**. Reproducibility has not been checked with a new data seed, another initialization, or human evaluation.

| Generation setting / decoder | Seen strict pass | Excluded strict pass | Paired difference from F0, seen/excluded | Size-distribution TV | Diversity pass |
| --- | ---: | ---: | ---: | ---: | --- |
| F0 / pixel mean | 17.06% | 10.63% | Reference | 0.309 | Fail |
| F0 / logit mean | 23.31% | 14.38% | Reference | 0.221 | Fail |
| F1 FiLM / logit mean | 28.50% | 20.00% | +5.19/+5.63 pp | 0.208 | Fail |
| F3 FiLM+CFG 2.0 / logit mean | 32.50% | 20.31% | +9.19/+5.94 pp | 0.322 | Fail |

The differences in this table are **paired comparisons with F0** using the same development conditions and generation noise. They are not gains computed by subtracting the 15.6% result of a previous independent experiment. Strict passing requires both the requested shape and color and the fixed shape-quality criterion. None of the 20 configurations passed the selection rule. In particular, every configuration exceeded the 0.20 total-variation limit for the size marginal distribution. F3+CFG 2.0 had good seen/excluded color accuracy (99.88%/99.06%) and shape accuracy (77.44%/77.50%), but outline and size remained bottlenecks.

When the same flow representation was rendered with two decoders, logit mean raised F0 strict success by +6.25 pp on seen combinations and +3.13 pp on excluded combinations. With a fixed encoder, reconstruction strict success was also higher for logit mean (91.80%) than pixel mean (76.17%). This is evidence that outline reconstruction is one real bottleneck. However, F3+CFG 2.0 strict success remained at 32.5% seen and 20.3% excluded, with large variation by shape. Changing the decoder alone did not resolve condition-specific shape failures.

For F3+CFG 2.0 with logit mean, strict success by condition was 0–17.5% for shape 2 versus 40–87.5% for shape 3. Because all 24 conditions should be covered, the mean alone is insufficient to judge success. Center and orientation diversity met their current criteria, but size-distribution TV averaged 0.322, above the 0.20 limit. [Per-condition raw results](../evaluation/p1_v1/F3_s2p0/results.json)

The next stage is the P2 direct-pixel generation baseline in the [plan](../../planning/generation_recovery_v1/PLAN_EN.md). It tests whether similar failures remain after removing the compression path. Model selection for P1 stops with these development results; final confirmation data must not be used to change settings.
