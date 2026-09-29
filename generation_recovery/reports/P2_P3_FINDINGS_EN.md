# P2–P3 Development Findings and Confirmation Candidate

The P2 direct-pixel U-Net had 797,443 parameters and trained for 4,000 steps in about 436 seconds. Among 24 conditions × 8 separately generated samples (192 total), strict-pass rates were 3.75% for seen and 3.13% for excluded combinations; shape accuracy was 30.63%/28.13%. This small diagnostic sample is not a final performance estimate, but it was far below the latent model at the same training budget. We therefore did not run its full development evaluation or the C4 pixel variant, which would cost about four times as much compute. [Training record](../runs/p2_d770101_s0/plain/run.json), [diagnostic results](../runs/p2_d770101_s0/plain/validation_8_per_pair.json)

P3 kept data count, AE, decoder, and model architecture fixed, increasing only flow training from 4,000 to 16,000 steps. We resumed the P1 optimizer and sample/noise/time RNG states and evaluated with the same development noise: 24 conditions × 80 generations. [Branch decision](../P3_DECISION_EN.md), [recomputation check](../evaluation/p3_steps16k_v1/verification.json)

| Setting / logit mean | Seen strict pass | Excluded strict pass | Size TV | Overall diversity |
| --- | ---: | ---: | ---: | --- |
| F0, 4,000 steps | 23.31% | 14.38% | 0.221 | Fail |
| F0, 16,000 steps | 39.00% | 29.06% | 0.196 | Pass |
| F3, 4,000 steps, CFG 2.0 | 32.50% | 20.31% | 0.322 | Fail |
| F3, 16,000 steps, CFG 2.0 | 56.19% | 46.25% | 0.310 | Fail |

Training the paired F0 control longer improved strict success by +15.69 pp on seen and +14.69 pp on excluded combinations. This experiment provides evidence that **insufficient training was an important cause of the low automatic success rate**. F3 had higher condition accuracy and strict success, but its size-distribution TV of 0.310 exceeded the prespecified 0.20 limit, so it was excluded from the final candidate set. Model selection ends with these development results. Only F0 at 4,000 and 16,000 steps will undergo a paired comparison on new data. The confirmation configuration is in the [pre-registered lock file](../confirmation_v1/lock.json).

These numbers come from one development data seed and one initialization. Whether the effect holds on new data, and whether size, position, and orientation diversity hold when considering only strict passes, still require confirmation.
