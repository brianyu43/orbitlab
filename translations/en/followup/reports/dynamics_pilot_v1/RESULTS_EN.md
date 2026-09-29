# Correct answer status First diagnosis of future prediction

2026-09-23. Completed 3 environments × 7 models, 21 training iterations with sample extraction comparison, and **a total of 30 CPU training iterations**. All were initialized once, 1,000 steps, and validation diagnostics. The final comparison for C03 and C04~C07 have not yet been completed.

## Comparison conditions

We provided four-digit object states, object-specific behavior, and interaction/decay to all models. The actual number of objects is two in the training process, and each minibatch randomly mixes four-digit values including empty slots. We compared the general scene model (flat), object-specific independent model (object), interaction model (interaction), rotation augmentation, exact constraint that only rotates the state (wrong_exact), joint constraint that rotates the environment (joint_exact), and relaxed model that mixes incorrect constraints with the general model.

The independent model per object is a contrast group that cannot use neighboring information due to its structural nature. The four symmetric comparisons used the initial weights of 15,076 parameters of the same basic interaction model. The relaxation model has an additional mixing coefficient of 1. flat/object have 16,624/11,428 respectively. They do not claim that parameters and time are the same in all methods. The mixing of learning step·batch·loss·sample·position is correct. The initial output of all models is the same constant baseline, and the color·size·existence are preserved according to the task definition.

## Current results

Below is the speed error, which is the difference in the **correct answer state provided each moment and the next frame predicted**. Units are pixels/frames; the lower the value, the better. Each scene was first averaged, and then 64 validation scenes were averaged. Just because the model evaluated 65 frames does not mean it automatically rolled 61 frames.

| Conditions | Speed limit | General interaction | Rotation restriction only on state | Environment also rotation restriction |
| --- | ---: | ---: | ---: | ---: |
| No external force, collision sample emphasized | 0.0543 | 0.1771 | 0.1553 | 0.1553 |
| Fixed gravity, emphasis on collision samples | 0.0859 | 0.2786 | 0.1964 | 0.1961 |
| Changing external force, emphasizing collision sample | 0.0909 | 0.2307 | 0.1731 | 0.1706 |
| No external force, uniform sample | 0.0543 | 0.1246 | 0.1276 | 0.1276 |
| Constant gravity, uniform sample | 0.0859 | 0.1082 | 0.1421 | 0.1372 |
| Changing external force, uniform sample | 0.0909 | 0.1295 | 0.1522 | 0.1442 |

In the overall average speed error for all 30 learning models, they were worse than the constant velocity baseline. It cannot be said that they have shown excellent future prediction capabilities from the outset. We also confirmed that the calculations of the two accurate symmetric models become identical without external forces, as shown in the results. We do not interpret the small difference between the two constraints in the gravitational/change external force as the final advantage of a single initialization.

## Cause separation: Learning sample ratio

In the first pilot, half of the minibatch was selected from the transfer where collisions or wall reflections occurred, and the rest was randomly selected. The contact transfer of the raw material is 6.6–7.7%, but the expected ratio in actual learning is 53.3–53.9%. This was a choice made to ensure that we would see many difficult moments.

Then, the 9 control groups maintained the same initial weights, step, batch, and loss, and the second half was also randomly sampled from all transfers. This was verified by reproducing the results of the first half's samples and ensuring that all positions were mixed identically. **The overall average improved for all 9 groups, but the contact moment error worsened.** For example, in the general interaction model of fixed gravity:

| Evaluation section | Contact emphasis | Uniform sample |
| --- | ---: | ---: |
| Every moment | 0.2786 | 0.1082 |
| Contactless moment | 0.2641 | 0.0553 |
| Contact moment | 0.4368 | 0.7119 |

In other words, in this pilot, the sample ratio created a reciprocal relationship between non-contact/contact performance. Lack of training, structure, normalization, and long-term state distributions remain candidates, and the sample ratio alone does not explain all errors. Even when dividing the training period (t=0~18) into the time intervals (t=0~18) and the later answer states (t=19~63), the model still performed worse than the baseline. [Diagnosis by Time Interval](../../../../../followup/reports/dynamics_pilot_v1/horizon_diagnostics.json)

## Verification and next execution

The predictions of the first 21 models (86,016 predictions and 252 cumulative predictions) were recalculated. The predictions of the 9 sample-comparison models (36,864 predictions) were also run again. All checkpoints, data, and settings hashes, initial weights, and sample-position matching were checked. Rotation constraints, object order, empty slots, and gradient unit checks also passed. [pilot verification](../../../../../followup/reports/dynamics_pilot_v1/verification.json) · [sample-comparison verification](../../../../../followup/reports/dynamics_sampling_v1/verification.json)

C03 continues by dividing as follows.

1. **Simple physical reference line:** Only apply known resultant force and wall reflection in addition to constant velocity, and add a reference line that excludes collisions between objects. Clearly define what the model needs to learn to win.
2. **Learning curve:** Record longer learning periods with the same settings and the learning/validation error. Both contact/non-contact losses are reported, and only the overall average is selected.
3. **Input and loss diagnosis:** Compare by dividing the position/velocity errors in physical units from the normalized learning loss. If a structure including known movements is added, it aligns all information from the comparison groups with the basic motion.
4. **Final setting fixation and repetition:** After determining the settings based on the development results, repeat the initialization, and evaluate the test/property combination/object count/external force conditions that were postponed.
5. **Follow-up connections:** Video input for C04 and actual continuous prediction/behavior changes for C05/C06 are evaluated separately from this single oracle result step.
