# Progress in follow-up research

2026-09-23 · The overall goal is still in progress. Several objects, external benchmarks, dynamics, and final deployment have not yet been completed.

Out of 31 task items, 21 are completed, 3 are in progress (SVIB, literature comparison, key figures), 1 is waiting for human assessment, and 6 are waiting for follow-up. These numbers do not refer to the model's capabilities or research completion. The easy explanations and latest values for the current capabilities are available in `CURRENT_CAPABILITY_KO.md`.

## Work actually performed

- We confirmed the hash matching of 1,324 files of the existing distribution materials at the start and this intermediate audit.
- Created 8,192 evaluation stress tests and 128 voice comparisons. Since rotation/deformation is shared, they are not counted as 8,192 independent scenes.
- The automatic rejection criterion was corrected by using the new calibration and independent test, and the sample of 3,456 from the existing 6-model generation was re-evaluated.
- 6,000-step AE examined 6 internal expressions and new restoration evaluation scenes.
- I completed 9 times of fixed-step 9 times of time like only changing flow in the same C4 AE, total 18 times of learning.
- The standard MSE threshold of invariant expressions was verified in 512 new scenes.
- I prepared 240 generated products for judging people who hide the source. There is no actual person response yet.

## Important corrections regarding the evaluator

The existing template IoU≥0.65 condition was passed by all 32 of the original circle, square, and random noise. Therefore, we cannot interpret the existing "valid & condition match 52.2%" as the success rate of a good shape that humans have guaranteed.
In the new calibration, IoU≥0.80 and foreground occupancy ≤0.35 were selected. In the independent test, clean shapes 256/256 and weak blurring 252/256 were maintained, and 64 of the 64 circles, squares, and noise were all rejected. This is also an automatic selection criterion tailored to the synthetic reference set and does not replace human verification.
The percentage of existing B2 that satisfied both new standards and shape and color consistency was seen 16.6%, OOD 12.5%. B1 was 3.8% and 1.7% respectively. The result of shape and color classification itself did not change 57.0%/34.0%.

## The result of changing only the generator in the same expression

| Budget | Generator | Shape/Color Match | New standard and shape/color match |
| --- | --- | ---: | ---: |
| 4,000 steps | General | 40.1% | 3.2% |
| 4,000 steps | Rotation enhancement | 39.9% | 2.7% |
| 4,000 steps | Accurate C4 | 58.2% | 16.2% |
| Same study time | General | 42.1% | 4.2% |
| Same learning time | Rotation enhancement | 42.2% | 3.5% |
| Same learning time | Accurate C4 | 56.5% | 15.0% |

The table above represents the average of three initializations that align the same AE·initial raw weights·latent pool·evaluation noise. The data generation seed is still the original one. Since the AE conditions at the same time are fixed in common, the AE cost is also the same, but the inference cost during generation is not the same. The exact C4 flow took approximately 2.7 times longer than the general flow in a 64-step generation.
Even in the latent space of the general flow, there are already four rotation learning samples. The results of additional rotation augmentation are not interpreted as “the difference between a model that did not see rotation data and the original model.”

The shape information of 6,000-step C4 AE was read as linear probe 45.4% and nonlinear RBF probe 75.0%. It cannot be concluded that the shape information is absent, nor does it mean that it has been separated into a concept that can be edited independently.

## Confirmation experiment of new data completed

Instead of including the existing seed 42 in the test sample, we created completely new data seeds 43001/43002/43003. We eliminated duplication of the rotation orbits between the original/existing follow-up evaluation scenes and the new split. Combining the initializations 0/1/2, we sequentially executed 9 AEs, 27 total of fixed-step general/C4 flow and time control general flow. The conditions were fixed in `configs/confirmation_v1.json` before execution.
The process terminated successfully, and the final aggregation of `confirmation_v1/summary.json` and additional validation of `confirmation_v1/verification.json` was confirmed. Hash and fixed AEs for 9 AEs and 27 flows, initial weights, latent, and evaluation noise alignment, and 5,376 new scene rotation trajectories were verified, along with material and indicator aggregation. No AE validation gates set in advance were bypassed.

| Generator | Match the shape and color of this combination | Match the strict criteria of this combination | Match the shape and color of the combination excluded | Match the strict criteria of the combination excluded |
| --- | ---: | ---: | ---: | ---: |
| General, 4,000 steps | 37.9% | 3.3% | 33.5% | 2.3% |
| C4, 4,000 steps | 55.9% | 15.6% | 52.0% | 11.4% |
| General, same learning time | 41.5% | 5.3% | 36.1% | 3.0% |

The C4 superiority of the major indicators was maintained in all 9 paired comparisons between fixed step/time and the original combination/excluded combination. Conditional intervals were recorded separately for the conditions where generated noise was re-collected within the fixed conditions of each model pair, and the effects per data seed were recorded separately. Since there are three independent data seeds, we do not claim statistical significance of the sample.

## decoder cause analysis completed

Same learning representation/ground truth state × pixel/logit mean decoder × 3 initializations, completed a total of 12 learning iterations. When restoring the learning representation, the logit mean improved from 0.843 to 0.893 for mask IoU, and from 73.7% to 83.5% for strict criteria, shape, and color simultaneous pass. When using two new decoders with the same generated latent space, the strict criteria pass improved from 17.4% to 21.5%, but the shape and color alignment did not improve. The original decoder's corresponding generated sample results were 16.2%.
The restoration count is 216, and rotational consistency, frozen AE, learning materials, and weight matching were verified against 5,184 generated samples. These decoder results are a development experiment where the existing diagnostic pool and noise were reused. They are not referred to as verification performance for new data. The cause analysis and next-step model selection are found in `STAGE_A_CONCLUSIONS_KO.md`.

## Complete multiple object data and answer state model

- Built and played back 1,792 basic scenes and 4,820 original intervention pairs. Checked for global rotation, non-target preservation, and overlap in source/target rotation trajectories between split.
- Created 6,356 tasks by combining commands under object count and occlusion conditions. Both models received the same correct state and target commands, and matched 34,991 parameters with 6,000-step learning sample and slot shuffling.
- The average success of editing the entire scene in the 3rd initialization was 99.54% flat for two objects, 99.93% for objects; 10.74%/100% for three objects; 0.78%/100% for four objects. This was the result of a simple independent intervention that provided the correct state.
- The success rate is sensitive to the criteria of allowing coordinates 1 pixel and radius 0.5 pixel. The flat success rate of the three objects is 52.5% at 2/1 pixel and 91.6% at 4/2 pixel. The continuous coordinate error was also recorded and did not replace the fixed main criteria.
- 6 real learning sessions, 6,356 correct task, 26,272 predictions, and 2,304 scene-level aggregation were verified. You can check them in `object_oracle_study_v1/RESULTS_KO.md`.

## RGB object models and external materials

We implemented a small flat/Slot Attention model that does not use the ground truth mask in training and tested its consistency in order, gradient, and separation metrics. The two models in the first 1,000-step pilot collapsed against a black background. We preserved the failure results and code. The second pilot completed each 3,000 steps using a linear RGB output and a loss of foreground/background balance calculated from the input. The IoU of the restored foreground improved from flat 0.392 to Slot Attention 0.455, but the object-slot correspondence mask IoU is low at 0.093/0.072. Just high foreground ARI does not interpret object detection success as such.

We re-ran the four models in both versions on the CPU and verified the stored masks and metric aggregation. This was a development learning based solely on validation, and we did not complete the full verification experiment for B04. We left the failure cause candidates and a longer learning and mask diagnosis plan in `object_perception_pilot_v2/RESULTS_KO.md`. Currently, the process for both of these pilots has all completed successfully.

We have secured and verified 500 public previews of the SVIB official code. External model evaluations and the full official protocol execution are still pending. We have begun comparing the source code and official code of Slot Attention, ISA, SysBinder, Dreamweaver, and SEN, and recorded the scope of verification and the unverified originality range in `NEAREST_METHODS_REVIEW_KO.md`.

## Remaining work and verification limits

- The automatic cause analysis in stage A is completed, but the human judgment A04 is still left.
- The RGB object model and the intervention/generalization of object count for the corresponding expression in the B stage, SVIB and failure analysis, the predictive model, image inference, long-term and counterfactual prediction in the C stage, and the literature comparison, conclusion, research memo, and replication bundle in the D stage are still remaining.
- The human review material is in `human_review_v1/review.html`. I verified the anonymization and JavaScript syntax, but the browser tool blocked the local file URL due to security policy, so I could not perform screen and operation checks. I did not bypass it.
- The error in the path for recording the status of the first evaluation run and the conflict between the first flow's evaluation variables were preserved and corrected to preserve the cause and failure record. The already completed learning was not restarted or replaced with a good seed.
- The verification of `initial_verification.json` is a intermediate consistency check for the hash of raw data, sample count, aggregation, and noise matching. It is not a complete certification that proves the overall research objectives or scientific hypotheses.

## Exercise and environmental data and prediction pilot

Completed C01/C02. Generated 2,240 basic trajectories for circular objects and 512 trajectories for counter-action response. Replayed 111,040 state values and 8,960 observation videos, and inspected 912 trajectories that rotate together with external forces, as well as 96 exceptions when external forces are fixed. Ranges and values are provided in `dynamics_world_v1/RESULTS_KO.md` and `verification.json`. It is not considered a verification of the performance of continuous physics or learning models.

We trained the 3 environments × 7 models of C03 for 1,000 steps. Subsequently, we added 9 comparison runs where only the sample extraction was changed, completing the actual 30 CPU training sessions and generating 122,880 predictions. All models performed worse than the baseline in terms of overall average speed deviation. When switching to a uniform sample for the training that emphasized contact, the overall average improved for all 9 models, but the contact moments deteriorated. The subsequent training curves, physical baseline, and independent evaluation were more clearly separated in `dynamics_pilot_v1/RESULTS_KO.md`. C03 is currently underway.

The connection area detection, which does not input the correct mask as a contrast group for RGB object separation, was applied to validation 128. We obtained a overlap ratio of 0.9973 and a number of objects of 128/128, but we confirmed the failure to merge the overlapping objects. This is a baseline for easy data and does not prove learned representations or occlusion restoration. Comparisons and validations were recorded in `object_component_diagnostic_v1/RESULTS_KO.md`.

## Object recognition additional learning ended

Each model was trained from the stored 3,000-step state to 6,000/12,000 steps. An additional 18,000 steps and four checkpoint evaluations were completed normally, and the parent optimizer, original random flow, and 512 CPU predictions were re-verified. The IoU of the slot attention's visual preview improved from 0.455 to 0.523, but the individual object mask IoU remained at 0.072 to 0.090. The flat IoU was 0.093 to 0.139. The B04 completion was not completed, and the diagnostic order by cause is in `object_perception_curve_v1/RESULTS_KO.md`. This additional training, 30 cycles of dynamic learning, and the data generation and verification processes all completed normally.

## Image model fixed evaluation completed and next state of this experiment

The evaluation range for B04 has been completed. The 12k model was fixed, and inputs, output masks, and RGB contributions were divided, and the common RGB foreground post-processing was evaluated under all holdout conditions. The output of the learning model was re-verified for 1,536 rows of object detection, 6,912 rows of object segmentation, and 1,280 rows of reconstruction. The Slot+foreground had a test IoU of 0.628, for the four objects 0.742, and for the occlusion 0.427. We do not claim to have achieved intervention capability. The range, errors, verification, and drawings are available in `object_layers_heldout_v1/RESULTS_KO.md`. We proceeded with the object count/occlusion/summation/fragmentation analysis for B07 and the first drawing for D02, but the intervention synthesis and the entire dynamic trajectory drawings are still missing.

C03 expanded the 9 identical initialization learning paths to 12k and reproduced 110,592 predictions from 27 checkpoints. Some exceeded the steady-state baseline, but the non-learning reference that only calculates known force and wall reflection does not meet the overall average. Subsequently, the common 12k configuration for 7 methods × 3 environments × 3 initializations was fixed. The existing reuse of 9 checkpoints and the evaluation of new 54 learning and hold conditions are currently in progress. Execution status is verified via `dynamics_state_study_v1/progress.json` and the actual process, and it has not yet been marked as complete.

## Status reading and the latest results of 63 exercise models

Six RGB status readers were trained, directional scoring errors were evaluated separately with a separate evaluator, and the predicted 6,912 lines were re-verified. The validation shape matching accuracy was 34.6% for flat and 40.6% for Slot Attention. Simultaneous success for all scene properties and counts was 0/128 for both initializations of the two methods. This is a pre-editing status reading and not the overall image manipulation result. B05 is currently underway. The revision history and interpretations are available in `object_readout_pose_corrected_v1/RESULTS_KO.md`.

The 63 model evaluations and hold condition predictions for C03 for 1,720,320 playback sessions were completed normally. The individual recalculation of physical reference original objects also passed 200,704 times. Models that rotate together with state and external forces had a test speed error of 0.0411 pixels/frame for varying external forces, which was lower than the general interaction model of 0.0698, but increased to 0.1821 for the four objects. The final model aggregation audit and research report are still pending. Video-based or continuous future predictions have not yet been evaluated. The current capability document has also been updated to align with this range.

## Final comparison of actual image editing and state movement

The fixed evaluation for B05/B07 was completed. New 640 pages and 2,961 tasks were created, and it was confirmed that there were no overlaps with the existing 8,148 object rotation orbits. The 1,280 RGB encoders, 3,840 readouts, and 68,103 editing/output operations for 115 comparison conditions, as well as 17,388 metrics and intervals, were re-verified. The 2,138 counts of additional 180 conditions, 10 failed classification, and 8 masked intervals also passed. The selection of two objects was 93.5% flat and 92.2% Slot, but the entire editing process that required preserving all properties and counts of the actual answers yielded 0 observations. The command execution based on high estimated states was distinguished from actual answer success. `object_image_edit_v1/RESULTS_KO.md` and `OBJECT_FAILURE_SYNTHESIS_KO.md` have numerical and limit issues. Comparison diagrams and step-by-step GIFs for the first fixed case were generated and verified.

The cross-model aggregation audit for C03 has been completed. For 63 CSV files, 910 metrics for 455 conditions were compared using paired t-tests, and `dynamics_state_study_v1/RESULTS_KO.md` was generated. C03 was marked as complete, but the ability of video inference and continuous prediction was not included. The next C04 is divided into 7 stages, starting from input leakage inspection, through video baseline, supervisor reading, identifiable environmental inference, comparison groups, replication, and rollout connections. At the time of this recording, a new video inference experiment had not yet been executed.

## Video status and environment estimation and one-step connection completed

Completed C04. We trained a direct CNN using four RGB action sequences and a video measurement+learning correction model with each initialized to two different initializations, and compared the non-learning reference with the reference that knows the circular renderer and physical integration rules. We re-verified 2,240 input trajectories, 8,960 RGB images, 5,760 development predictions, 307,200 one-step predictions for 156 hold-out conditions, and 93,456 aggregated and interval predictions. The hold-out data consists of 1,280 original trajectories, and repeated comparisons are not counted as independent data.

The position error for the general test was directly 3.71 pixels for CNN, with measurement+learning correction at 0.039 pixels. The speed error connected to the same joint rotation motion model was 0.0246 pixels for the ground truth input, 0.8833 pixels for the CNN input, 0.0638 pixels for measurement+learning correction, and 0.0529 pixels per frame for measurement+physical rules correction. Learning correction was worse than physical rules in terms of external forces not included in the training data. Input information, prior knowledge, error comparison, and limitations were recorded in `dynamics_observation_eval_v1/RESULTS_KO.md`.

We also constructed specific cases where different resistances and combinations of actions produced different past behaviors and shaped the future. This is an example of information limits in C07, and we have not completed the entire independent repetition and observation length study. The continuous future and action change evaluation for C05/C06 is still not implemented, and the overall goal is still ongoing.

## 61st stage self-prediction and behavior change evaluation completed

Completed C05/C06. Evaluated 1,404 conditions for the initial information four learning modes × seven learning modes and two non-learning reference × three initializations × 13 hold groups. Past and opposite behavior transfer of 3,373,056, similar to the 8,432,640 baseline transfer conditions that do not reset the correct state, were replayed using separate cyclic codes. Main state indicators of 10,450,944 and behavior response indicators of 2,654,208 were independently calculated, and they were validated for aggregation, intervals, and the C04 first-stage connection.

In the general external force test, the 61st-stage position error of the coupled rotation model increased from the initial correct input of 14.206 pixels to the input of 14.517 pixels for video measurement + learning correction. Large finite deviations were observed in the four objects, and 750 of the total baseline comparison paths were failures due to analogy. The failure ratio and finite conditional error were recorded together. The behavior response of non-target objects was worse than the non-response comparison under general conditions. The termination of evaluation does not imply the achievement of a stable world model.

The contact candidate detection for models with no collision output, which had been applied without separate correction to 78,080 correct transfers, revealed false positives and omissions. We created and visually inspected 62 frames of GIF images for the first fixed case, including long-term curve stability, stability across all methods, and behavioral response diagrams. `dynamics_autonomous_v1/RESULTS_KO.md` contains results and limitations. C07 was segmented into independent data repetition, observation length, condition omissions, identification boundaries, and playback/aggregation. The corresponding new learning has not yet been executed, and the overall goal is still underway.

## Start with two sets of independent data and repeat learning

Generated 4,480 trajectories, 222,080 states, and 17,920 video frames for new data seeds 997101 and 997102, and all were played. Confirmed 1,024 counter-actions, 1,824 rotation trajectories, and 608 object sequence changes. The initial physics and video rotation trajectories, combined with the original data, are 6,720 and have no duplicates. This is a new sample from the same discrete physics simulator and is not a real-world environment verification.

Starting new training on 126 models while maintaining the existing initialization, sample extraction, and learning budget from both data bundles. For completed models, we are verifying consistency with the order of saved predictions, optimizer steps, initial weights, and original sample extraction. The actual completion count is recorded in `dynamics_repeats_v1/state_progress.json`, and the validation count is recorded in `state_verification_progress.json`. We do not assess process survival based on the log files alone; instead, we verify the execution sessions separately.

The 12 new training iterations of the RGB CNN and measurement+learning correction and estimation models were completed, and 11,520 train/validation predictions were verified, along with hash, initialization, and optimizer step inputs and outputs for training. The 2,304 validation predictions from the new evaluation input path also matched the stored development predictions. The standalone evaluation for the new C04–C06 hold-out connections has not yet been executed. The prepared and trained work completed in `dynamics_repeats_v1/PROGRESS_KO.md` and the remaining evaluation work were separated. The C07 project overall, observation length and condition missingness, and external evaluation and final research bundle are still ongoing.


## Independent Repeat: Complete full evaluation replay of the first new data

126 new state predictors have completed their training and are now conducting the final playback verification. All 156 video connections under condition 997101 and 1,404 conditions under condition autonomous prediction and counter-behavior were played back. The 327,680 non-learning physical reference transfer between the two new datasets were also verified using separate code. The second batch evaluation and data seed-level aggregation and reporting will be performed next. The entire subsequent scope, including observation length, missing environmental information, external evaluation, and human judgment, remains incomplete.


## Independent repetition reporting completed and start of re-learning due to environmental information missing

We played back both the C04–C06 evaluations of 126 new state models and 12 video estimators, as well as the two new datasets. We examined the state, video, and autonomous prediction aggregation of three independent data sets and verified the correlation of 135 figures, and prepared a report. The one-step joint rotation gain under general conditions was maintained for all three datasets, but the superiority of video correction varied depending on the data, and long-term divergence was significantly observed in the new dataset. The extreme sign difference in the aggregation was due to the cancellation of floating-point numbers, and the material was preserved, and separate verification was corrected to a precise integer sum. 81 environmental information missing cases are currently being re-trained, and we examined three cases where hidden information was changed to become the same input and different subsequent answers. The observation length experiment only prepared separate detailed plans but did not execute them. The entire C07 and subsequent targets are underway.


## Observation length data playback completed and execution status confirmed (2026-09-23T15:30:25+09:00)

6,720 trajectories, 360,000 states, and 53,760 RGB images, along with 1,536 counter-action trajectories, were verified by playing back at the common action time point t=7. The data is reused by matching the existing initial scenes and is not counted as new independent data. Input preparation by length has begun, and the learning of the 72 video estimator has not yet started. Environment missing learning was confirmed by survival in the execution session, and a new 62/81 items were completed based on snapshots, with a total of 89/108 conditions completed. The existing separate verification is at 49/108 conditions, and the next incremental verification is currently in progress. The overall research goal is ongoing.


## Environment information missing comparison completed and observation length learning·connection evaluation started

We played 9,815,040 predictions of transfer across 81 new missing conditions and 27 overall information conditions, totaling 108 conditions. We verified the differences between initial and scene states, such as 1,800 per data set, using separate codes, and confirmed the values of 48 tables and 36 figures. The hidden combined state of the joint rotation model increased the general test speed error by 30.8~51.3%, but the structure that only rotates in state improved when the combined state was obscured by external forces not learned. The 18 pairs of two rotation structures that hid the combined state had the same learning weights. The error limits of the three configuration examples are not expanded to the lower limit of the entire dataset.

The observation length data of 6,720 samples and 53,760 input/reference entries per length were passed through the non-interference of hidden inputs and model capacity verification. The 72-model estimator was started on MPS/CPU, and the C04–C06 evaluation began using the 6 validated models from the first data and one input. Evaluation, replication, and aggregation of the remaining lengths/data, as well as the final comprehensive assessment and external research scope, remain pending.

## 2026-09-23 16:15 KST Execution status updated

The observation length model for 72 samples has completed learning, development, playback, and verification. 11 steps and 11 long-term future models have been calculated out of a total of 12 combinations, and separate playback verification and final aggregation are currently underway. The aggregation code and independent verification pre-tests, which preserve the length differences between scenes similar to all indicators, have been completed. After verifying 500 pairs of SVIB public previews and 32 baseline conditions, we began learning 24 models. [Status by time point](STATUS_SNAPSHOT_2026-09-23_EN.md)

## 2026-09-23 16:34 KST External example/pre-research comparison completed

SVIB preview validated 24 models and 4,800 prediction/7,200 rotation input predictions, 1,536 aggregation/1,920 pair comparisons. C4 had a pixel error smaller than the general model, with an alpha average of 4/4 and initial pair of 11/12, but both models were worse than identity at the alpha average of 4/4. 48 table values, 92 table/figure connections, and 30 fixed example sources were completed, along with PNG visual verification. float32 separate placement formula validation failures were preserved and separated into float64 formula validation, and the actual storage prediction/rotation validation limits were unchanged. The pre-announced reduction external task scope for B06 was completed.

D01 concludes with a comparison of nine methods, combining the existing five methods with Aether, PEnGUiN, PE-RL v2, and Modeling What Changes v1. It confirms the Aether and sparse-residual formula code as fixed commits and explicitly states the implementation of unconfirmed/unpublished features. It does not claim the originality of general external force combination, partial symmetry, or object manipulation itself.

The observation length experiment has completed all prediction calculations for 12 combinations. Separate playback verification, aggregation, and summary are currently underway. The status is {'complete': 26, 'awaiting_human_review': 1, 'in_progress': 2, 'pending': 2}. The total goals remain C07, the entire drawing D02, the final notes D03, the replication bundle D04, and the actual human judgment A04.

## 2026-09-23 17:05 KST Observation length · Final document completed

All C04/C05/C06 playback and observation lengths, observation 52 and autonomous 468 aggregation blocks, and 126 data points, 127 curves, and 5 drawings in Table 126 have been verified. The C07 summary connected information deficiencies, the non-uniform effect of observation length, and long-term dispersion. In the completion review, it was found that the direct comparison of the same data for A05 at 3k/6k was missing, so 12 checkpoints, 216 validation candidates, and 344,064 reading results were examined. The cost of generation, values for 99 drawings, and the final notes, easy explanations, and 25 core visual material lists were also verified. Currently, 29 are completed, D04 ZIP/rearrangement verification is underway, and A04 human response is waiting.

## 2026-09-23T17:12:45+09:00 Jaehyun bundle·final audit completed

Completed 30 of 31. All 70,514 files in the 12,534,372,531-byte ZIP were completely rehashed, and the actual decompression was performed in another directory to replay 96 model and 2,688 CPU sample predictions. The entire file hashes before and after playback were identical. Build session 72212 and repartition session 10086 both exited with exit 0. This is not a full re-learning or new environment installation verification. The original 1,324 files were also preserved. Only 240 actual human judgment A04 images (response 0) remain, and the overall target is incomplete. The 29/31 records before ZIP sealing were unchanged and the final receipt was preserved outside the ZIP.
