# Conditions for maintaining the benefit of symmetry and conditions for failure

2026-09-23. Independent data repetition, environmental information omission, and observation length comparison were completed. The conclusion supported by current evidence is that **correct symmetry helps with one-step predictions, but it does not address the absence of necessary information, image recognition errors, or the instability of long predictions**. These three issues cannot be combined into a single accuracy score.

## Does the benefit of the same structure appear in different data as well?

In an experiment that provides both the state and external forces at every instant, the model that rotates both the state and external forces reduced the speed error of the general test by reducing the number of independent data in all three cases compared to the general interaction model. The reduction was 16.8% to 41.1%. The hold condition, which varied the number of objects and the external force, was also evaluated. The reference case that directly calculated the known force and the wall and omitted only the collision between objects was more accurate in all three data sets.

Therefore, the benefit of the appropriate symmetry constraints has been repeatedly observed in this small synthetic world, but it is not evidence of obtaining a new optimal physical model. Even using three initializations, the independent data units are still three. [Repeat results and all comparison sets](dynamics_repeats_v1/RESULTS_EN.md)

## Do the information provided and the symmetrical structure affect each other?

We re-trained models that hide either the combined strength or the resistance, or both. Out of the total 108 conditions, 81 were new missing conditions learned, and 27 were comparisons of overall information. When combined strength was hidden in the joint rotation model, the general test speed error increased by 51.3%, 32.8%, and 30.8% in each of the three datasets. The effect of only hiding resistance was inconsistent.

In inappropriate constraints that only rotate the state while keeping external forces fixed, the side that hides the external forces performed better under the non-learned external force condition. Therefore, a good prediction is not guaranteed just because there is a lot of information. The information must match the expression structure. When the sum is hidden, the calculations of the two rotation structures become the same, and the weights were also the same in the 18-pair correspondence learning. [Environmental information omission and structure comparison](dynamics_context_omission_v1/RESULTS_EN.md)

## Does looking at the past longer at the same final point in time make it better?

We compared observation sessions 1, 2, 4, and 8, keeping the same action and future at the same final time point t=7. The capacity, initial weights, sample extraction order, and number of learning iterations of 72 estimators were adjusted to match the length of the data. Since the initial scenes from the three existing datasets were reused, the new independent data set bundle has not increased.

The general test average of the method of measuring a circular object and correcting it with MLP was as follows. First, the initialization average for each of the three data was obtained. This method utilizes prior knowledge of color, circular shape, and motion rules.

| Observation video | Speed estimation error (px/frame) | Combined estimation error (px/frame²) | Next speed error (px/frame) |
| --- | ---: | ---: | ---: |
| 1 page | 0.7405 | 0.0203 | 0.7478 |
| Page 2 | 0.0471 | 0.0198 | 0.0768 |
| 4 pages | 0.0574 | 0.0049 | 0.0809 |
| 8 pages | 0.0611 | 0.0075 | 0.0843 |

When the number of images increased from one to two, movement reading cues became available, resulting in a significant reduction in speed error. The combined accuracy estimate was the smallest among the four images. The combined accuracy error due to the same scene difference between eight and four images increased in all three data sets, and the 16-step position error also increased in all three data sets. Conversely, when comparing eight images to one image, speed, combined accuracy, next speed, and the 16-step position error all decreased in all three data sets.

This result means that **the benefit of observation length was not monotonic in this fixed estimator and learning budget**. It does not mean that more information is inherently harmful. A sufficiently flexible estimator may ignore additional past data. Contact, approximation of simple physical laws, input representation, and optimization can all act together in long observation periods, and this experiment alone did not disentangle the causal contributions of each. The contact interval was fixed at the 7th past time of L8 to calculate the length differences within the same group.

Even with eight RGB CNN inputs, the speed estimation error was large at 0.8155. It is separate from whether observational information exists and whether the current model extracts that information. [All tables, denominators, and 520 aggregation blocks](dynamics_observation_length_v1/RESULTS_EN.md)

## Will a stable future be achieved if the initial estimation improves?

That was not the case. The general condition 61 step position error of the joint rotation model with measurement+MLP was also 15.3173 pixels per data point, approximately 2.555×10⁵, approximately 8.651×10²³ pixels, even in L8. The large number does not be interpreted as a normal prediction that remains finite. Even if the relative failure rate is zero, a very large finite divergence occurred. Relative failure also appeared in the four objects.

If the comparison that provided the initial state and environment of the correct answer was extended for a long time, it failed, and we confirmed that the same result was obtained regardless of the observation length. This comparison is not a way of giving the correct answer again every moment afterward. Therefore, the long-term failure cannot be attributed solely to the initial video recognition. The cumulative error and the stability of the learned transfer must be treated separately. The known physical reference and the total prediction length 1·4·8·16·32·61 were preserved together.

In the previous independent repetition at the start of t=3, the behavioral response to the 16th target was better for both data sets than the non-response control, but the response transferred to another object was worse for both data sets. In the t=7 length comparison, the target and non-target responses and the failure rates for the two pathways were also separated. Since the response error for the finite case and the non-response average for the entire case may have different denominators, we do not interpret this as a conditional difference.

## Problems that cannot be solved because there is no information and problems with insufficient learning

These are four different video and action sequences with different forces and resistances, creating two environments where the future changes after intervention. Additionally, three pairs of situations were also created where the current state is the same, but only the hidden combined force and resistance differ. A deterministic predictor that outputs the same input for the same output cannot accurately match both correct answers simultaneously. The lower bound of the squared error for this case was confirmed using the standard identity that the squared distance between the two correct answers is 1/4.

These are the boundaries of the pairs we have constructed. They do not extend to the Bayes error limit for the entire random test, the identifiability of all long observations, or the performance limit of the model. Possible subsequent directions are active observations that demonstrate strength or predictions that indicate uncertainty. These two directions are proposals outside the completed range this time and are not execution results. [Same past · Different future](observation_identifiability_v1/EXPLANATION_EN.md) · [Three pairs of hidden conditions](dynamics_context_omission_v1/identifiability/EXPLANATION_EN.md)

Currently, C07 evaluates and connects both the boundaries of directional effect, information provision, observation length, object count, and prediction time. The conclusion that it has achieved high levels of visual understanding or completed a world model is not supported. The research contribution candidates should be presented as a comparable testable replication of this control and failure. The originality of the force estimation and the asymmetric combination itself overlaps with [nearby prior research](NEAREST_METHODS_REVIEW_V2_EN.md).
