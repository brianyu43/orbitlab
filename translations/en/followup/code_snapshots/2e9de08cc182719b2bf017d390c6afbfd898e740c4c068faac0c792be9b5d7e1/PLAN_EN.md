# Segmented execution plan

2026-09-23 · The status and output are recorded together in `status.json`. Maintaining the full scope below, the termination of a single step does not mean the termination of the entire goal.

## A. The cause of the difference between current ability and generation

| ID | Work | Evidence to judge completion |
| --- | --- | --- |
| A01 | Original·43 times learning output protection, execution environment check | Full file hash check of the existing release manifest, MPS verification, separate output path |
| A02 | Pre-fixed new evaluation data and experimental conditions | Duplicate testing among train/val/test/OOD orbit, setting file·code hash |
| A03 | Evaluation device blurriness, distortion, false detection check | Shape and color judgment by distortion, valid detection rate, false detection of voice examples, sensitivity of judgment criteria |
| A03b | Voice comparison group for standard evaluation correction | Distinguish locked standard in new calibration, independent test and product re-evaluation, human verification |
| A04 | Preparation and collection of independent human judgment of the products | Model, seed, seen/OOD layer 240 pages, source screen private, response CSV, actual human response whether |
| A05 | Expression re-examination of the 6,000-step AE for generation | only train for probe fitting, validation selection, comparison with new test/OOD results and 3,000-step |
| A06 | Separation of restoration/generation errors, check of output by decoder direction | Comparison of restoration/generation of the same evaluator, inconsistent output by direction, error between average output and configuration output |
| A07 | Compare only flow restrictions on the same C4 AE | Each AE has general/enhanced/accurate C4 flow, weighted/sampling/evaluation noise matching, each 4,000 steps |
| A08 | Re-comparison controlled by calculation time and generation cost | Same time flow 3 types×3 seed, total AE+flow cost and inference time table |
| A09 | Check main effects with independent data generation seed | Total 3 data seed × 3 initialization from locked setting, uncertainty for new test and scene |
| A10 | decoder structure and correct latent comparison | alternative decoder that preserves equivariance, correct state oracle, comparison of the same input and learning budget |
| A11 | Verification of the restoration limit of invariant expressions | Derivation of general MSE inequalities, numerical verification of figures, separate interpretation of invariant-only results |
| A12 | Write the conclusion of cause analysis | Conditions for the effect to be maintained/disappeared, limitations of the evaluator, reasons for selecting the model to take to the next stage |

## B. Expression, combination, intervention of objects

| ID | Work | Evidence to judge completion |
| --- | --- | --- |
| B01 | Two object renderer and correct state·mask | Reproducible generation, individual object ID, non-colliding partition, global rotation verification |
| B02 | Correct answers for color, rotation, and position intervention pairs | Correct answers for preservation test and manipulation synthesis/order tests for variables other than changed variables |
| B03 | Diagnostic model using object segmentation with the correct answer | oracle display, comparison of equivalent conditions between flat scene representation and object representation |
| B04 | Expression model for finding objects in images | Evaluation results without entering the correct mask, slot allocation, masking, and reconstruction results |
| B05 | Generalization of unobserved combinations and object count | Sufficient slot capacity for all models, accuracy of target change and preservation of non-targets, results per object count |
| B06 | SVIB External Task Connection | Check official materials and split, reduce search/distinguish official protocol, data exposure record |
| B07 | Failure Analysis of Object Representation | Examples of curves with different shading, object count, and combination complexity, respectively |

## C. Future predictions including environmental conditions

| ID | Work | Evidence to judge completion |
| --- | --- | --- |
| C01 | Directionless motion and collision simulator | State and video match, dynamic verification, trajectory unit split |
| C02 | Fixed gravity, wind and condition transformation answer | T(Rs,Ra,Rc)=R T(s,a,c) exception when the test and condition are fixed |
| C03 | Status oracle's transition comparison | Constant speed/general/per object/symmetric/softening/conditions included model, same input information |
| C04 | State and environment inference in videos | Same past observation length, perception/condition inference error separated from oracle |
| C05 | Long future and unobserved environment·combination evaluation | Longer rollout than learning, object identity·position·speed·collision time |
| C06 | Prediction of counterfactuals that change behavior | Includes the correct trajectory of intervention from the same initial state, as well as the physical response of non-target objects |
| C07 | Independence of results repetition and boundary conditions | Differentiation of the identification impossibility of directional effects, observation length, and combination variance effects, and missing conditions |

## D. Research explanation and replication

| ID | Work | Evidence to judge completion |
| --- | --- | --- |
| D01 | Comparison of the closest prior research articles and codes | Table distinguishing actual originality candidates and previously solved parts |
| D02 | Key figures and intervention videos | Results of cause separation, calculation cost, condition changes, combination/long-term prediction |
| D03 | Short research notes and easy explanations | Hypothesis, methods, results, failures, limitations, connection between numbers and data |
| D04 | Complete all-range completion gratitude and separate distribution bundle | Actual evidence of all items, playback inspection, unchanged inspection of existing release, public of incomplete items |

## Common rules

- Each execution uses a new output directory and locked settings. Error correction retry is performed in a separate directory, leaving the existing failure log.
- Separate exploration and verification experiments. After observing the test of verification experiments, only select good seeds or do not lower the termination criteria.
- Data duplication is checked on the basis of the rotation orbit. Frames with the same trajectory as four rotations of the same scene are not counted as independent samples.
- Report them as different comparisons with the same step, the same calculation time, and the same number of parameters. They do not claim that they have met all the criteria at the same time.
- Even if human detection is not collected, code, control experiments, and new renderer work can be carried out. Only conclusions that require human verification are withheld.
- Action planning and inferring feasible rules were included as a selection extension in the top proposal. After confirming the results of the C stage, a separate specification will be made if a specific need arises.
