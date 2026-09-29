# Analysis of the cause of the single figure stage

2026-09-23. This is a preliminary conclusion based on automatic evaluation and control experiments. The human judgment A04 remains, and the overall research objectives continue.

## Verified effect

If only the generator is changed on the same C4 AE, the automatic evaluation advantage of accurate C4 flow is maintained. From three new data generation seeds × three initializations, the proportion of shapes that meet strict geometric criteria and shape/color requirements was 3.3% for the general flow and 15.6% for the C4 flow. Even if the general flow is trained for the same amount of time, it was still 5.3%, and in both the 9-pair comparison and the 9-pair comparison, C4 was higher. AE, initial raw flow weights, learning latent, and generation noise were tuned. [Confirmation aggregation](../../../../followup/reports/confirmation_v1/summary.json), [Verification](../../../../followup/reports/confirmation_v1/verification.json)

This is evidence that the rotational constraint of flow helps in this single-figure task. The original difference between the entire B1 and B2 sets cannot be explained by a single cause. There are three independent data seeds, and the inference cost was not adjusted. In the previous measurement, the generation of C4 was approximately 2.7 times slower than the general flow.

## Fixed the evaluation criteria

The existing IoU of 0.65 passed the test of passing both round-triangular, square, and random noise. The standard was corrected using a new calibration and independent comparison test. The previous 52.2% cannot be considered a high-quality generation success rate. After correcting the same original B2 sample, the value is 16.6%, and the value for the new data repetition is 15.6%. The corrected automatic standard also does not replace human quality evaluation. [Correction results](../../../../followup/reports/evaluator_calibration_v1/summary.json)

## There are bottlenecks in the part of creating pictures too.

The same encoder was fixed and the decoder was newly trained. Both methods matched the number of parameters, initial weights, the order of learning scenes, latent space, and samples, and set the number of steps to 6,000. One method converted each direction output to pixel values using the sigmoid function and then averaged it (pixel mean), while the other method averaged the values before the sigmoid function and then converted them to pixel values (logit mean). Both methods maintain the exact C4 rotational relationship.

| Input and measurement | Pixel mean | Logit mean |
| --- | ---: | ---: |
| Test restoration mask IoU from learned expressions | 0.843 | 0.893 |
| Strict standards, shape, and color of test restoration in learned expressions | 73.7% | 83.5% |
| Passed strict criteria, shape, and color in the test recovery state of the correct answer | 43.9% | 86.7% |
| Strict criteria, shape, and color passed by the combination based on the same flow generation latent | 17.4% | 21.5% |
| Strict criteria for combinations excluding the same flow generation latent in terms of shape, color, and pass | 9.0% | 13.2% |

The same flow sample results for the original decoder were 16.2% for this combination and 10.8% for the excluded combination. The original decoder was trained together with AE, and the two new decoders were trained separately on the fixed encoder, so the comparison with the original model is a reference value with different training histories. The strictly controlled pair is the new pixel/logit decoder.

Additionally, the shape and color consistency of the new pixel/logit decoder in this combination is 58.1%/57.4%, which did not improve compared to contour selection indicators. The problem of simply changing the drawing method to select and generate the desired shape was not resolved. There is still a significant difference between the high performance during restoration and the low performance during generation. The oracle in the correct state also uses a learned decoder, so it is not a perfect renderer limit and is influenced by optimization and structure.

Training 12 instances, recovery metric aggregation 216 instances, C4 numerical error, and frozen AE with weight and data matching were tested. The generation transfer used the same stored latent space of 576 × 3 initializations × 3 decoders, and no new flow training was performed. This decoder experiment is a search that reuses the existing diagnostic pool and evaluation noise. The 15.6% observed on new data and the 21.5% development results are not directly combined into a performance increase curve. [Recovery comparison](../../../../followup/reports/decoder_study_v1/summary.json), [Recovery verification](../../../../followup/reports/decoder_study_v1/verification.json), [Generation transfer](../../../../followup/reports/decoder_transfer_v1/summary.json), [Transfer verification](../../../../followup/reports/decoder_transfer_v1/verification.json)

## What was confirmed and what was left in the expression

The shape was read as 45.4% by a simple linear reader and 75.0% by a nonlinear reader. Therefore, we cannot conclude that there is no shape information at all, but we can also conclude that it has become a concept that can be edited independently. The four sets are not four objects but four rotation directions. [Expression diagnosis](../../../../followup/reports/probes_6000_v1/summary.json)

A standard MSE bound for the common recovery of invariant rotational expressions was separately derived and verified. This is not a new theorem, nor is it a bound that explains the entire source of generation errors for the entire C4 expression. [Bound explanation](invariant_bound_v1/EXPLANATION_EN.md)

## The choice to take to the next step

1. The standard for the single-object experiment is maintained using C4 AE+flow after completing independent repetitions. The logit mean is stored as the result of a promising decoder search and is not referred to as the final improved model after verifying new data.
2. Multiple object stages separate comparisons that give the correct state from comparisons that find objects in RGB. Direction bundles are not interpreted as object bundles.
3. Since the effect of object sharing rules has already been verified in a simple intervention where the state is already given, the next bottleneck is object separation and property reading. Evaluate this using only RGB, then extend it to environmental conditions and future prediction.
4. There are remaining human evaluation, external SVIB model results, dynamics, text and code comparison with close research, and final replication bundles.
