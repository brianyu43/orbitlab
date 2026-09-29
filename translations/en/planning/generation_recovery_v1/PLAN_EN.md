# OrbitLab Single Shape Creation Improvement Plan

Created: 2026-09-23 · Status: Investigation and design completed, implementation and learning not completed

First, the recommended approach is to **verify the improvement in contour restoration (logit mean) and condition transfer improvement (FiLM·CFG) separately, and compare it with a small direct image generator**. Based on current metrics, the most obvious problem is the ability to create the desired shape and the ability to produce clear contours. The lack of model size or training data has not yet been confirmed as the cause.

This plan addresses the problem of the strict automatic success rate of 15.6% for a single figure. Recognition and editing failures of multiple objects and long-term motion prediction divergence are separate problems, and we do not consider them to be resolved by the success of this plan. It uses category input of shapes and colors and does not evaluate natural language understanding.

## 1. Facts confirmed now

| Observation | Basis | Judgment that can be made here |
| --- | --- | --- |
| Combination color 98.5%, shape 56.6%, shape·color 55.9%, strict pass 15.6% | 3 new data × 3 initializations | A bottleneck where shape and contour are larger than color. |
| Strict pass of excluded combinations 11.4% | Same independent repetition | Even combinations that are not in the learning process still have problems. |
| Same time general flow 5.3%, C4 flow 15.6% | 9 comparisons of response, independent data are 3 | Rotation constraints helped but not enough. Inference cost was not the same. |
| Recovery 73.7→83.5% when replacing the decoder, generation 17.4→21.5% | Comparison of control between fixed encoder and new pixel/logit decoder | Contour recovery improvement is promising. A good recovery model does not guarantee a good generator. |
| Shape and color matching in the same decoder comparison 58.1→57.4% | Same storage generation latent | The problem of creating shapes that match the condition by only modifying the decoder has not been solved. |

Evidence: [Current capability](../../followup/reports/CURRENT_CAPABILITY_V2_EN.md), [Independent repeat material](../../../../followup/reports/confirmation_v1/summary.json), [Recovery comparison](../../../../followup/reports/decoder_study_v1/summary.json), [Decoder comparison of same generation representation](../../../../followup/reports/decoder_transfer_v1/summary.json).

15.6% are independent verification results, and 21.5% are other development experiments. These two values do not indicate the improvement in the preceding and subsequent experiments. The previous 52.2% are results of a loose judgment and are not used as the starting point for quality goals. The existing 9 human cases and 231 AI cases are supplementary reviews of separate development samples.

The current code compresses 64×64 images into 16-dimensional per-rotation and a total of 4×16-dimensional. Shapes and colors are already input as separate one-hot vectors, but they are only applied once at the flow MLP's input. The decoder averages the pixel values per rotation, and generation uses Euler 64 steps. Therefore, explanations such as "separating the conditions first" or "failed because there was no rotation data" are incorrect. The four groups are not four objects, but four rotations. [Current implementation](../../../../work/orbitlab.py)

## 2. The solution methods and application order investigated

| Priority | Method | Specific application in OrbitLab | Limitations of the basis |
| --- | --- | --- | --- |
| 1 | logit mean decoder | Average before converting rotation-specific outputs to the final pixel value. Compare with pixel mean using the same encoder/latent generation. | Local development results are available, but new data verification is required. |
| 1 | FiLM condition transmission | Shape and color conditions expand and shrink and move the characteristics of each hidden layer. Currently applied to 2 hidden layers. | The original paper is a visual inference research. The improvement of this generation problem is a hypothesis. |
| 1 | classifier-free guidance, CFG | In some learning, conditions are set aside, and conditional predictions are emphasized more during generation. | Instead of following the shape more closely, diversity may be reduced. It cannot simply be attached to existing models that do not learn unconditional branching. |
| 2 | Small U-Net flow in pixel space | Directly generates 64×64 images without compression or decompression. | It serves as a baseline for comparing expression bottlenecks. More calculations may be required. |
| Conditional | Spatial Broadcast Decoder | Spread the compressed values across the entire screen and draw them with a small CNN along with x·y coordinates. | The C4 relationship is not guaranteed by introducing coordinates alone. Branch rotation structure and numerical verification are required. |
| Conditional | Latent with expression regulation or spatial structure | Only tested when the difference between the compressed expression of the generated expression and the actual drawing is large. | There is no need to unconditionally make the latent distribution normal. There is still no basis that normalization itself is a solution. |

This is a proposal to apply affine transformations by feature of FiLM to the current condition delivery. [FiLM, AAAI 2018](https://arxiv.org/abs/1709.07871v2)

CFG learns both conditional and unconditional predictions together to control quality and diversity. Conditional removal learning and CFG are also used in the official image implementation of Flow Matching. [Original CFG paper](https://arxiv.org/abs/2207.12598v1), [Official training implementation](https://github.com/facebookresearch/flow_matching/blob/main/examples/image/training/train_loop.py), [Official generation implementation](https://github.com/facebookresearch/flow_matching/blob/main/examples/image/training/eval_loop.py)

The baseline for direct image generation is re-designed to match the size of this task based on official Flow Matching image examples. It is not called a replication of the official large-scale experiment. [Guide](https://arxiv.org/abs/2412.06264v1), [Official image examples](https://github.com/facebookresearch/flow_matching/tree/main/examples/image)

Decoders using coordinates showed improvements in reconstruction and representation in the VAE experiments of the original paper. Latent diffusion can be referenced when designing the relationship between compressed representations and generation quality. Neither study directly confirms the bottlenecks in OrbitLab at present. [Spatial Broadcast Decoder](https://arxiv.org/abs/1901.07017v2), [Latent Diffusion Models](https://arxiv.org/abs/2112.10752v2)

## 3. Step-by-step execution design

### P0. Separate evaluation, implementation, and numerical error first

1. Read the existing checkpoint, noise, and evaluation files to replay the saved scores. Write the new code in a separate directory and do not modify the existing results, ZIP, or status files.
2. Verify the order of shape and color labels, image range, latent's train-only mean and standard deviation, and the consistency of rotation effects and conditions. The maximum absolute error target for the FP32 C4 new numerical test is 1e-5; if it exceeds this, record precision and the underlying cause.
3. Design a diagnosis that trains only one AE to memorize 64 learning images and one small flow to create fixed figures for each condition. Each should have a maximum of 6,000/4,000 steps. Use strict pass-through of 95% as the diagnosis target on the same learning sample. This is not a generalization score. We do not require the stochastic flow's MSE to be exactly 0.
4. Compare Euler 64 times and Heun 32 steps (velocity evaluation 64 times) using the same noise in the storage model. Euler 256 times is for checking numerical sensitivity. The results that require more computation are not counted as a free improvement. Only 24 combinations × 32 for verification are generated first.
5. Compare the compressed values of the actual image with the generated latent space. Record the normalization distance and the difference between `E(D(z))` and `z` for the probe and train conditions. We cannot conclude that an image is outside the representation space just by using one distance.
6. Even images drawn directly according to the correct shape rules are placed in the same evaluation tool. This is used to check the pass rate of the normal drawings in the evaluation tool and the maximum limit of the renderer. Non-learning renderers that know the correct rules separate the achievement of the learning model from it.

If implementation errors or evaluation discrepancies are found, first fix them and then compare again. If the validation strictness passes by 10%p or more using only the integrator, compare the corresponding correction with the previous method first and reduce the scope of the structural experiment. If the normal plot also does not pass well, examine the evaluation version separately, and do not secretly lower the existing threshold to increase the success rate.

Output: `diagnostic_report.json`, fixed case contact sheet, separate time/memory measurements. Not generated at the time of this planning.

### P1. Separate the effect of conveying contours and conditions

Uses a basic picture of a single set of new data for development, 1,024 images, and the same 20 learning combinations plus 4 excluded combinations. The C4 rotation version is not counted as a separate independent scene. Only compare from initialization 0 and repeat with 1·2 when the direction of improvement is confirmed.

First, train one C4 AE with the existing condition of 6,000 steps. Fix the encoder, then train **two decoders** each for 6,000 steps. The initial weights, sample order, parameters, and optimization settings for both decoders are the same.

| ID | flow condition transmission | probability of removing conditions during learning | guidance during creation |
| --- | --- | ---: | --- |
| F0 | Existing input concat | 0 | None |
| F1 | FiLM by hidden layer | 0 | None |
| F2 | Existing input concat | 0.1 | Compare intensity 1, 1.5, 2, 3 in validation |
| F3 | FiLM by hidden layer | 0.1 | Same comparison |

Each flow is trained on the same fixed encoder for 4,000 steps, with a batch size of 128 and a learning rate of 0.001. F0/F2 and F1/F3 use the same initial weights per structure, and the RNG of the training scenes, noise, and time samples is separated from the dropout RNG. FiLM and concat have different structures and do not claim that all weights are identical. The number and width of the hidden layers start the same, and the parameters of the additional condition layer and the actual training time are disclosed. To claim that the improvement of FiLM is solely due to the condition transfer effect, add a maximum of three comparisons between concat with similar parameter budgets.

Each flow draws **the same latent space** with both decoders. It does not re-train the flow for decoder comparison. F0/F1 are general generation tasks that provide conditions. In F2/F3, the guidance strength is compared with that of guidance 1, allowing for the comparison of the effectiveness of condition removal learning and the emphasis effect during generation.

The definition of guidance is `v_null + s × (v_cond − v_null)`, and when s = 1, it is a general conditional generation. The formula code's `cfg_scale=w` corresponds to `(1+w)v_cond − w v_null`, so it applies to s = 1 + w here. These notations are not confused. A linear combination of conditions and unconditional velocity, while maintaining C4, is used.

**The initial comparison is AE 1 + decoder 2 + flow 4 = 7 training sessions.** The maximum memorization diagnosis for P0 is set separately to 2 sessions. If you repeat all three initializations, P1 will have 21 training sessions. You do not rotate all subsequent segments from the beginning.

Selection rules: validation uses 24 combinations × 80. Strict passing of this combination is at least 5%p higher than F0, and the excluded combinations do not deteriorate by more than 2%p, and if the diversity and color preservation conditions of the entire generated products below are satisfied, it is repeated with additional initialization. If the number of strict passes in this small sample is insufficient, the diversity of that part is left unconfirmed and judged in the final verification stage. The decoder alone effect is compared in the same flow·noise. If there is no improvement, only up to 2 combinations are left for P2/P3 diagnosis. Before final verification, the decoder type·guidance intensity·solver·learning termination point are confirmed.

### P2. Block the bottleneck with the baseline without the compression process

A small conditional U-Net flow is created by directly learning from 64×64 RGB. The initial configuration consists of channels 32/64/64, two sequential residual blocks, and time, shape, and color conditions. Spatial attention is omitted in the first comparison. The final detailed structure is recorded as the setting before training.

First, test one general U-Net that provides C4 rotation data. Measure the 100-step processing throughput and the 24 combinations × 8 generation/evaluation times, then record the expected cost for 4,000 steps. For each model's local training, set a 30-minute limit in priority. If the limit is exceeded, leave it as `budget_limited` and do not infer that the task or structure failed. The batch is fixed after memory measurement.

If promising, add a C4 version that averages the outputs of the same raw U-Net over four iterations. Each model is limited to a maximum of three initializations for both the general and C4 versions, totaling six iterations. Both models report actual completion steps, including a comparison of learning times. They also present the total encoder/decoder training and generation costs alongside the potential space model. They do not claim that the same cost is achieved solely by varying the number of channels or solver calls.

Initially, CFG is turned off and the presence or absence of the compression path is compared. Even with the condition strengthening verified in P1, they are not immediately all mixed together. U-Net's success is a basis for prioritizing the compression path, but since it changes both structure and capacity, it is not proof that 'compression is the only cause.'

| Result | Next decision |
| --- | --- |
| The latent improvement model has also improved sufficiently | Prioritize the low-cost model for final verification. |
| Direct pixel generation is good, but latent generation is low | We prioritize improving the interface between latent and decoder in P3. |
| Both methods cannot memorize even low and small data | Re-examine from labeling, scale, loss, numbers, and optimization. |
| Memorize small materials, but the low score in the new picture | Test the problems of data diversity, learning amount, and generalization of combinations separately. |

### P3. Add only one quarter according to observation

| Observation | Additional experiments to choose | Scale limit |
| --- | --- | --- |
| Even after compressing and reconstructing the actual image, the contours collapse significantly | Maintain Spatial Broadcast Decoder and C4 branch rotation above the frozen encoder | 3 times |
| Restoration is good and the difference between the generated latent and the actual encoding is large | compare from the decoder robustness of adding small Gaussian noise σ=0.01/0.03 to the standardizing latent in train | 2 candidates × 3 initialization = 6 times |
| decoder Kang Geon-hwa does not have any effect and the pixel baseline is superior | Design one of space latent or weak KL regulation AE in a separate protocol | Outside this execution range; not automatically extended |
| There is a clear indication of overfitting or insufficient learning | Compare the train 1,024→4,096 of the selection model using the same steps, or compare 4k→16k using the same data | Additional level on the selected axis × 3 initialization = 3 times |
| FiLM improvement must be separated from capacity increase | Concatenate comparison to match parameter budget | Max 3 times, combine with other branches P3 within 6 times |

Ganggeonhwa Jabeum is an experimental proposal for this project and not a replication of the LDM paper. It is tested to ensure that C4 does not break down using different normalization or coordinate processing for each rotation. The method of adding shape supervision to AE is a separate setting that provides additional label information, so it is excluded from the first comparison.

EMA can be used to verify raw/EMA by separately recording a weighted mean in the same learning process if necessary. If applied, the coefficients and warmup should be fixed in advance, and they should not be included in the mandatory changes for the first FiLM/CFG cause comparison. The assumption that "expanding the data significantly solves the problem" does not justify expanding all candidates.

### P4. Final confirmation with only two candidates using new data

Evaluation is conducted using the existing C4 benchmark + a maximum of 2 improvement candidates, with 3 seed for new data generation and 3 for initialization. **Maximum of 27 model paths**, and the units differ from the number of training iterations for AE, decoder, and flow respectively. Paths using the same encoder are shared. If there is only one candidate, there are 18 paths.

Save before evaluating the new seed/noise list and the final settings hash. The exact seed value is determined after duplicate checking and stored in the executable lock file, and it is cleared from the current draft. Duplicate checking is performed between existing development/verification materials and rotation orbits/complete images. The results of the already processed 43001/43002/43003 cannot be considered new verification materials.

24 combinations per model × 128 pieces = 3,072 generated (2,560 learning combinations, 512 excluded combinations) are used. Noise is adjusted between models of the same dimension. Since the latent and pixel models have different dimensions, they do not claim that they used the same noise tensor, and they only adjust the condition, sample ID, and evaluation budget.

Model selection is completed in the development validation phase. After reviewing the results of the final test and new noise, the guidance, steps, checkpoint, and seed are not reselected. The effect of each of the three datasets is first observed, and the initial average is calculated. The number of generated images is not counted as thousands, nor are the initial 9 initialization points counted as independent data.

The exclusion combination is the “shape and color combination excluded from the learning of each model.” The existing four exclusion combinations have already been observed by the researchers. Therefore, it is not called a verification of a combination design that has never seen new results. To make that claim, a separate verification is needed to change the exclusion combination pattern.

## 4. Evaluation and decision-making criteria

The primary metric is the existing **strict geometric criteria AND requested shape/color matching**. [Maintain IoU of 0.8, margin of 0, and maximum occupancy of 0.35 from the fixed evaluation settings](../../../../followup/reports/evaluator_calibration_v1/locked_threshold.json). Do not use reward/loss mechanisms to directly optimize the evaluator scores.

| Evaluation axis | Recording method |
| --- | --- |
| Meaning accuracy | Record separately shape, color, simultaneous matching. Table for 24 combinations and average of seen/excluded macro. |
| Contour | Strict pass rate, template suitability, ratio of broken components and empty images. Only for restoration, compare IoU of input image and mask with boundary error. |
| Diversity | The central, size, and direction distribution of each of the two generated products and strict pass-throughs. Comparison with recent images and identical copy ratio. |
| Cost | Separate learning, generation, decoder, evaluation, and storage times. wall time, memory, and real raw network evaluation data recorded. |
| Symmetry | Check the C4 relationship between encoder/decoder/velocity/final generation with fixed input. |
| Visual review | Small blind contact sheet selected with a fixed seed. Separate AI and human judgments recorded. No additional human judgments are required for 240 people. |

There is no single correct image that has the same position and size as the randomly generated picture. We do not use the pixel MSE to generate the accuracy for the separate correct image. The position and size estimated as a template may be incorrect in the faulty output, so we also leave the readability ratio. We do not change the denominator of the indicator only by selecting images with high scores.

The diversity guard proposal is as follows: match the number of new normal images reference and generated samples to each combination, and compare the **distribution around each surrounding region** of the fixed center within the 3×3 radius × 3 directions 4-segment intervals. The generated samples must fill at least 90% of the intervals filled by the normal reference, and the total variation distance (TV) must be less than 0.20 for each attribute. Apply this to both the entire generated samples and the strict pass-throughs, and if the number of pass-throughs per combination is less than 32, the pass-through diversity is recorded as insufficient for verification. In the final decision, for each data seed, three initial outputs are collected and the distribution is examined, which is not a procedure to increase the number of independent data. In development, repeated experiments are not blocked solely by a lack of pass-throughs, and for the P4 improvement completion judgment, pass-through diversity verification is required. No post-processing is performed by artificially filling in empty slots. The final bin boundary and reading method are verified from P0 to the normal reference and then locked in the lock file before training.

In CFG, condition/conditional evaluation is also added in the same solver stage. The C4 average across all directions also adds cost. For example, a C4 CFG with s>1 may correspond to 8 raw evaluations per velocity. The actual time depending on batching is measured separately. The NFE wrapper of the official implementation merges the two forward steps within the CFG, so this number alone does not match the inference cost. For the final comparison, the selected quality settings for each method and the settings within the measurement time of the reference model are presented together.

**The first improvement goal is to achieve 30% or more seen strict pass, and 20% or more excluded combination pass.** This is not a current achievement value or expected success rate, but a proposed goal to determine whether the execution is successful. At the same time, it requires that seen averages be at least +10%p higher than the newly learned response criteria, excluded combinations be at least +5%p higher, and that no data individually deteriorates by more than 2%p. The color consistency must be at least 95% for seen and excluded, and the diversity guard must be passed.

Even if this condition is met, it does not be called a “stable product level” or “person verification completed.” It is the first milestone that the improvement direction has been reproduced. It does not claim precise statistical confirmation of the effect or generalization to other geometric or real‑world images based on three data points.

## 5. Time and workload management

| Bundle | Maximum scale | Conditions of proceeding |
| --- | ---: | --- |
| P0 | Reading·Reassessment + Small learning 2 times | Assessment·Implementation diagnosis |
| P1 First comparison | 7 times learning | After P0 error resolution |
| P1 including repetition all | 21 times learning | Reset the rest when there are promising candidates |
| P2 | Up to 6 times of direct image model | When compression path comparison is needed |
| P3 | Maximum 6 additional learning sessions | Quarterly-based according to the observation of the chart |
| P4 | Maximum 27 total model paths | Maximum 2 candidates and after protocol confirmation |

These numbers do not combine to form an immediate task list for execution. The next segment is determined based on the initial implementation, diagnosis, and 7-time comparison results. The time for the new structure has not yet been measured. The record that the training of the existing small C4 AE+flow took only about 87 seconds per run is not used as an estimate for the total pipeline time or U-Net time.

During execution, the first 100 training steps are updated, and the time required per step is refreshed based on real-time small-scale generation and evaluation. Record the 30-minute local learning limit and disk space allowance, and store results in `budget_limited` if exceeded to plan for expansion separately. Remote GPU uploads and external uploads are not included in this plan.

Output is based on bundled tensors/NPZ and row-wise metrics, and only some for visual review are saved as PNG. The hash and replay commands for checkpoint, config, noise, and evaluation versions are retained. All normal/failure/timeout interruptions are recorded, and while viewing the results, the failure seed is not replaced.

## 6. Conclusion that should be left as a research

The goal is to explain **what changes corrected what** along with the increase in the score of the good drawing. It separates whether the decoder only improved the contours, whether FiLM·CFG more closely followed the requested shape, whether diversity decreased in exchange, and whether C4 still yields a benefit even with the same calculation budget.

If the direct pixel generator performs better, we can study the conditions of the latent expression, and if both methods fail in a specific exclusion combination, we can study combination generalization. In this way, controlled experiments that distinguish failures are more valuable for analysis than the existing “when we added symmetry, it improved a little” approach. This statement does not guarantee the interest of a specific research lab or the possibility of a paper being accepted.

The [protocol.json](../../../../planning/generation_recovery_v1/protocol.json) in this folder is a pre-execution design specification. It is not an executable program and is in the `draft_not_run` state. [evidence_snapshot.json](../../../../planning/generation_recovery_v1/evidence_snapshot.json) contains the verified local measurements, file hashes, and external evidence, while [verification.json](../../../../planning/generation_recovery_v1/verification.json) contains the static verification of the planning file. It does not indicate that new learning or performance verification has been completed.
