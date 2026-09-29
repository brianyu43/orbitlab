# Current OrbitLab accuracy and implementation level of intention

2026-09-23. Verified the actual saved results and playback to update. Completed the execution of subsequent plans, including the A04 evaluation method changed by the user. It has not achieved full human verification. Generation and movement were compared with separate models, and image recognition, target selection, and command execution were linked into a single editing process for evaluation. It is not a single model that integrates all capabilities.

A basis has been established to implement the research plan and compare the causes. The ability to stably generate the intended scene or only change specific objects in the drawing has not yet been acquired. There is a significant gap between the high command success rate when providing the correct object state and the ability to read that state self-consciously in the image.

## Create a new single shape

This is the average of 3 new data seeds × 3 initializations from the C4 generator. It inputs 4 shapes and 6 colors as categories and did not evaluate natural language commands.

| Items | Combinations included in learning | Combinations excluded from learning |
| --- | ---: | ---: |
| Matches the requested color | 98.5% | 96.9% |
| Matches the requested shape | 56.6% | 53.9% |
| Both shape and color match | 55.9% | 52.0% |
| Satisfied with strict automatic shape standards and both shape and color matching | 15.6% | 11.4% |

If you create 100 images with this combination, about 56 images meet the shape and color classification criteria, and about 16 images pass even stricter geometric criteria. This is not a human quality guarantee. The original 240 review samples added 9 human and 231 AI auxiliary judgments. This is not full human verification or verification of this independent verification sample. The strict criteria pass rate for the same generative model with the same learning time was 5.3%, and the C4 side had both 9 pairs of matching comparisons higher. There are 3 independent data seeds.

We do not interpret the 52.2% based on the existing loose criteria as a high-quality success rate. The development results showing the existing diagnostic samples were drawn with a new decoder were 21.5%, but they do not replace the independent repeat 15.6%.

[Generation confirmation count](../../../../followup/reports/confirmation_v1/summary.json) · [Verification](../../../../followup/reports/confirmation_v1/verification.json)

## Know the object state and execute the command

The shared model for each object that provided the correct shape, color, coordinates, size, direction, and target was 99.93% of the two objects that satisfied both the target change and the preservation of the remaining properties and non-targets, and 100% of the three and four objects. It is a simple independent intervention for color change, 90-degree rotation, and movement, with the condition of allowing 1 pixel on each coordinate axis and a radius of 0.5 pixels. This is the result of three initializations and one data generation seed.

The obscured object also provided the correct state. Therefore, this number cannot be transferred to the success rate of image recognition or actual image editing.

[Result of intervention on the answer status and limitations](object_oracle_study_v1/RESULTS_EN.md)

## Reading the object state in the picture

A state reader was trained on the fixed RGB encoder. The train correct attributes were used as a supervisory signal, and only RGB is provided as input for evaluation. The average is the initialization of the reader for 128 validation images. The encoder is initialized once per method.

| Item | flat | Slot Attention |
| --- | ---: | ---: |
| Shape matching | 34.6% | 40.6% |
| Color matching | 91.5% | 88.8% |
| Direction matching | 26.3% | 22.4% |
| Coordinates are within 1 pixel of each axis | 13.0% | 25.7% |
| All properties of the scene and the number of objects match | 0/128, all three initializations | 0/128, all three initializations |

Color is relatively readable, but the ability to simultaneously identify shape, direction, and precise location is lacking. The total 0 successful cases are observations of this sample, and it does not mean the true success probability is exactly 0. This is a state-of-awareness score before editing. Subsequently, in separate new verification materials, the actual image editing was connected to the evaluation, and the results are as follows.

Previously, we corrected an evaluation error where incorrect predictions were made due to rounding errors in the direction vector. We preserved the model, learning, and initial results and created a separate new evaluation. The 6 readers, prediction 6,912 rows, and revised score replay verification passed. This table shows the revised results.

IoU applied RGB post-processing to the Slot Attention in individual object area segmentation was 0.628 for two objects, 0.742 for four objects, and 0.427 for occlusions. These overlap scores are not command accuracy and do not mean that 62.8% of the scenes were completely successful.

[Status reading results](object_readout_pose_corrected_v1/RESULTS_EN.md) · [Separated fixed evaluation](object_layers_heldout_v1/RESULTS_EN.md)

## Real image editing of new verification materials

We entered RGB·target point·explicit commands for 640 new scenes and 2,961 editing tasks. We connected the decoder's estimated state to the existing learning intervention model, and also evaluated the correct transformation rules and a control group that did not change anything. We removed duplication of existing data and rotation orbits, and re‑verified the predictions and actual output images for 115 comparison conditions, totaling 68,103.

In the two object tests, the target selection rates were 93.5% for flat and 92.2% for slot. The percentage of commands executed based on the state the model recognized itself in and the remaining percentage preserved was 100.0%/99.2%, but overall success, which included matching all properties and counts of the actual correct scene, was 0 for both methods. The overall success was also 0 for combinations of four objects with masking and non-learning.

When the same intervention model was given the correct state and correct target, both initializations were 100% correct in the new five conditions. Even applying explicit correct‑answer rules to the estimated state, since there were only 0 total successes, only the controller could be replaced to resolve the recognition error. In the masking case, even with a correct state, the rule of selecting the target around the point closest to the score only succeeded 92.2%. Natural language understanding or complete masking restoration were not evaluated.

[Real edit report](object_image_edit_v1/RESULTS_EN.md) · [Step-by-step GIF](../../../../followup/reports/object_image_edit_v1/edit_sequence.gif) · [Overall failure](OBJECT_FAILURE_SYNTHESIS_EN.md)

## One-step exercise prediction

3 environments × 7 methods × 3 initialization completed 63 model evaluations. 54 new learning and 9 previously verified 12k checkpoint reuses. 1,720,320 pending condition predictions were played back and verified. It provides the correct answer status and known external forces at every moment, and it is not an experiment that inputs video or predicts by itself.

In a test with variable external forces, the average absolute error in one-step speed was 0.0698 for the general interaction model and 0.0411 pixels/frame for the model that rotates both state and external forces together. In the external force conditions excluded from the learning process, the errors were 0.1292 and 0.0640 respectively. The lower the error, the better, and it does not convert it into a percentage of accuracy.

However, the error of the same rotation model in your object conditions increased to 0.1821. The reference to non-learning physics that calculates known forces and wall reflections was smaller, with general test 0.0129 and your object 0.0419. This reference assumes that you know the simulator's calculation rules and omits collisions between objects. The model cannot be claimed to have thoroughly learned physics or to be superior under all conditions.

[Model-by-model aggregation](../../../../followup/reports/dynamics_state_study_v1/aggregate.json) · [Prediction replay verification](../../../../followup/reports/dynamics_state_study_v1/verification.json) · [Physical reference verification](../../../../followup/reports/dynamics_state_baselines_v1/verification.json)

## Read the state from four past videos and predict the next moment

We completed video estimation and one-step connection evaluation for C04. We trained two models, RGB Direct CNN and a model that corrects the video measurements of circular objects by learning, respectively with three initializations. In the general test of variable external forces, the position error of RGB Direct CNN was 3.71 pixels on average, and the measurement+learning correction was 0.039 pixels. The latter uses a measuring instrument that knows the shape and color of the circle and the physical integration rules, so it cannot be interpreted as learning the same ability from only the video from the beginning.

When connected to the same fixed motion prediction engine, the speed error in the next moment was 0.0246 for the correct state·environment input, 0.8833 for the RGB direct estimation input, and 0.0638 pixels/frame for the measurement+learning correction input. The video measurement+non-learning physical estimation was 0.0529. The lower the error, the better, and it is not a percentage of accuracy. This evaluation only uses the initial one-moment data, so it does not directly compare the average performance improvement across the entire C03 period with the above.

In external forces not included in the learning process, the measurement+learning correction speed prediction error increased to 0.1421, and non-learning physical equation inputs were 0.0665. Therefore, it cannot be said that the improvement obtained under general conditions is maintained in new environments. We used 1,280 original trajectories from 13 hold groups, and 307,200 one-step prediction results from connection comparison were reproduced and verified. Long-term future evaluation was conducted separately in C05/C06 as follows.

[Video recognition and connection evaluation](dynamics_observation_eval_v1/RESULTS_EN.md) · [Replay verification](../../../../followup/reports/dynamics_observation_eval_v1/verification.json)

## Predicting the results of changing the long future and behavior

Completed the 1,404 condition evaluations for C05/C06. Saved response predictions that exhibited opposite behavior from the past, such as the 61-step prediction that only provided initial estimates, and all were reproduced from the initial input. The basic future transition count is 8,432,640 and the opposite behavior transition count is 3,373,056, and the independent sample consists of 1,280 original trajectories, of which 512 are responses. Repeated calculations of the same data are not counted as independent samples.

In the general test of variable external forces, the position error of the joint rotation model increased from 0.022 pixels at step 1 to 2.363 pixels at step 16 and 61 pixels at step 14.206 pixels in the video measurement + learning correction input. In the four objects, large finite dispersion was observed. Among the 138,240 basic comparison paths in total, 750 paths had non-finite values and were recorded as failures, and the failure period was not treated as a normal prediction.

In the same common rotation model + video measurement + learning correction input, the 16-step target location response error was 4.190 pixels, which was smaller than the 13.356 pixels of the no-response control. However, the non-target response error was 2.224 pixels, which was worse than the 1.510 pixels of the no-response control. In the 61st step, the superiority of the target response was not maintained either. It can be said that we have not sufficiently learned the stability of the long future and the behavioral transmission between objects.

Since the color and existence are maintained as they were in the initial state, they are not called tracking ability learned on their own. We independently verified it using the correct trajectory of the contact candidate obtained from the trajectory without collision output in the model. We confirmed 133,272 aggregates, 11,016 scene bootstrap segments, and 76,800 matches with the C04 first stage. The above values are the results of a single existing data generation seed. We rechecked them by adding two new data sets in the independent repetition below.

[Long-term/action change report](dynamics_autonomous_v1/RESULTS_EN.md) · [First case video of fixed](../../../../followup/reports/dynamics_autonomous_v1/autonomous_examples.gif)

## Re-confirmation in three independent data sets

The learning and validation of 4,480 trajectories for two new data sets, 17,920 videos, 126 new state models, and 12 video estimators has been completed. We played back 312 conditions for the C04 connection evaluation of the new data and 2,808 conditions for the long-term and behavioral changes in C05/C06, and validated a summary of three data units including the existing data. Nine initializations were not counted as nine independent data sets.

In the one-step speed error of general conditions, the joint rotation model performed better on both sets of data than the general interaction model, with a reduction in the error of 16.8–41.1%. However, the error of the physical reference that directly calculates the known force and wall was smaller on both sets of data. The data that outperformed the non-learning physical equations in the learning correction of video measurements were 2/3 of the general conditions and 1/3 of the non-learning external forces, so the superiority of one data set could not be generalized.

The instability of the long future became more evident. The 61st-stage general condition location error of video measurement+learning correction and joint rotation predictor was 14.52 per data, approximately 386 million, approximately 9.91×10²¹ pixels. It is the average including very large finite dispersion, and it is not just a number selected from successful cases. Under the same conditions, the 16th-stage target behavior response was better than both data for the non-response comparison, but the response transmitted to other objects was worse for both data. It cannot be said that a long-term world model or accurate object interaction has been achieved.

[Independent Repeat Results and Figures](dynamics_repeats_v1/RESULTS_EN.md) · [Full Data Verification](../../../../followup/reports/dynamics_repeats_v1/across_data/autonomous_verification.json)

## Does environmental information contribute to the model's benefits?

In a condition that covers both combined performance and resistance, it learned 81 new models from scratch and compared them with 27 overall information comparison conditions. It generated 9,815,040 transfer predictions across 108 overall conditions and validated three independent data sets, tables, and figures. The answer is a one-step experiment that provides current state and actions at every moment.

When combining forces in a joint rotation model was hidden, the speed error of the general test increased by 51.3%, 32.8%, and 30.8% per data set. The effect also deteriorated for non-learned external forces. In contrast, in models that do not rotate external forces together, the score for non-learned external forces was better when the information was deleted. The quantity of provided information and the model structure that appropriately uses it must be distinguished. The effect of hiding only resistance was inconsistent.

When the combined function is obscured, the two-rotation structure becomes the same function, and the weights were also completely identical in the 18 corresponding learning pairs. The three separate pairs of identical input and different answer examples illustrate specific cases of information deficiency, but they are not the optimal error bound for the entire dataset.

[Environmental information omission results](dynamics_context_omission_v1/RESULTS_EN.md) · [Number verification](../../../../followup/reports/dynamics_context_omission_v1/report_verification.json)

## Learn the rules from an external SVIB public example

We initially trained 24 small RGB predictors on 500 official public examples. Training per alpha was 70 pairs, and the external test shared by all models consisted of 100 pairs. This is not an experiment that transfers the existing OrbitLab checkpoint or replicates the entire official SVIB benchmark. We validated 4,800 prediction and 7,200 rotation input predictions, as well as all score and pair comparisons.

The test pixel MSE for C4 was all lower than the general model's 0.04402, 0.03786, 0.04387, and 0.03728 for alpha values of 0.04128, 0.02596, and 0.03124, respectively. However, both models performed worse than the identity model, which had an MSE of 0.02183 when nothing was changed. The learning MSE for C4 was extremely low, ranging from 0.00027 to 0.00041, resulting in a significant gap between the retained images and the original. The rotation output was consistent, but it failed to stably apply the transformation rules for small learning data to a new scene. Pixel error is not meaningfully converted into editing success rates.

There were 74 scenes that required actual changes, and 26 scenes where keeping them unchanged was the correct answer. In the area of changes, even if improvements are made, the areas that must be preserved and the damage to unchanged scenes can increase the overall error. We divided the two parts and also confirmed shape and color damage in the first fixed case. We did not perform human judgment or official LPIPS evaluation.

[External example results and diagrams](svib_preview_shape_swap_v1/RESULTS_EN.md) · [Total aggregation verification](../../../../followup/reports/svib_preview_shape_swap_v1/aggregate_v1/verification.json)

## Current judgment on the novelty of the research

The comparison of the nine closest methods has been completed. Aether already combines hidden external forces and equivariant dynamics, while PEnGUiN and PE-RL address partial symmetry. The very description of combining objects, symmetry, external forces, and operations cannot be claimed as a new contribution. Currently, OrbitLab's core is a diagnostic research that can reproduce the boundaries of symmetry benefits and failures from the same input information and observation length. It has even completed the comprehensive observation length, separating the benefits of symmetry from the limits of observation and long-term prediction.

[Text·Comparison of implementation and range of claims](NEAREST_METHODS_REVIEW_V2_EN.md)

## When the length of the past video is extended

At the same final time point t=7, video 1·2·4·8 frames, 72 estimated devices, and 520 total evaluation·playback·count block were verified. The speed estimation error between measurement and MLP was 0.7405→0.0471→0.0574→0.0611 pixels/frame. The gain from one frame to two frames was large, but it did not continue to improve as long as you watched longer afterward. The combined estimate was the smallest at 0.0049 for four frames and 0.0075 for eight frames. This is a condition using prior knowledge of the circular object and physical equations.

Even with 8 inputs, the position error for the general conditions at step 61 was unstable, with 15.3173 pixels per data point, approximately 2.555×10⁵, and approximately 8.651×10²³. This is a comparison of the same time point using the original initial scene, and it is not compared with the t=3 result or a simple performance increase. [Overall results for observation length](dynamics_observation_length_v1/RESULTS_EN.md) · [Information, symmetry, and time synthesis](C07_SYNTHESIS_EN.md)

## Expression check and final summary

The same input direct comparison of the 3,000/6,000 checkpoints that were missing in the final review was complemented. 12 models were read by CPU, the decoder was tuned only in training, and the settings were chosen for validation. The general test for C4 full-shape decoding was linear 41.67→45.54%, nonlinear 71.22→75.00%, and nonlinear untrained combinations were 32.81→49.22%. This is not an evidence of successful generation or the concept of independently editable editing. [Comparison and reproduction inspection](probe_step_comparison_v1/RESULTS_EN.md)

[Short Research Memo](RESEARCH_MEMO_V2_EN.md), [Easy Explanation](EASY_EXPLANATION_V2_EN.md), and [Key Figure & Video List](FIGURE_CATALOG_EN.md) have been completed. The execution of all 31 items was completed, reflecting the changes where the remaining judgments for A04 were delegated to AI. 9 human responses and 231 AI judgments were recorded separately. [A04 Changes](../PLAN_AMENDMENT_A04_EN.md) · [Judgment Results](mixed_review_v1/aggregate_v1/RESULTS_EN.md). The 70,514 files in the separate reproduction ZIP passed integrity checks, and 96 models and 2,688 sample predictions from other paths were also passed. This is not a test involving installing a new environment or performing a full retraining. [Final Thank You](scope_audit_final_v1.md) · [ZIP and Verification Records](../releases/orbitlab-followup-2026-09-23-v1/RELEASE_EN.md). The major bottleneck in current image editing is recognition, and movement is unstable even in initial answer information. The claims that natural language intent understanding, stable full image editing, and long-term world models have been achieved are not made.
