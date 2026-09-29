# Next experiment to find objects in RGB

2026-09-23. This is the development pilot for B04. We have prepared the code and unit tests, but we do not complete the entire B04 project just by completing the pilot.

The circular structure consists of repetitive competitive attention of Slot Attention, shared GRU, and a spatial broadcast decoder for each object. The final image is reconstructed by combining the RGB and softmax masks for each slot. Refer to Section 2.1–2.2 of the paper and the official `model.py` code. [Original](https://arxiv.org/html/2006.15055v2), [Official implementation](https://github.com/google-research/google-research/tree/master/slot_attention).

| Item | This small experiment |
| --- | --- |
| Input | Use only 64×64 RGB |
| Object capacity | Both sides of the models have 5 slots, 32-dimensional per slot; maximum capacity to hold up to four objects and background |
| Comparison | Slot Attention / Flat-to-slots model using the same encoder and decoder |
| Training information | Only use RGB restoration; the foreground weighting is calculated from the input brightness and do not use the correct object mask and properties |
| Evaluation | ARI overview, total ARI, optimal one-to-one response mask IoU, and recovery errors recorded separately |
| First budget | Each 1,000 steps, batch 16, 1 initialization, only check validation |
| Recorded difference | Flat model has more parameters; does not claim it is a comparison of the same parameters/time |

This is a development experiment aimed at reducing or changing the encoder's width, resolution, decoder's loss, data scale, and training budget compared to the original. It is not referred to as a benchmark replication of the paper. The paper and code original are preserved in `references/slot_attention/` along with URLs and hashes.

Before execution, unit tests were performed to verify the invariance of the attention input order, consistency of the slot order, finite gradient, RGB mixing, mask aggregation, and perfect/merged examples of separation metrics. The pilot is used for checking computational time and whether the model collapses. Subsequently, the entire experiment is shifted to a fixed set of experiments using the learning budget, seed, and new evaluation data for verification.

In the entire B04–B05 section, object assignment, masking, and the three-four object, unobserved combination are evaluated without inputting masks. If a separate supervised probe is used to read properties, it is clearly distinguished from unsupervised expression learning. Interventions are scored together for goal change and preservation of non-target objects. Since the current data have different colors for each object, color segmentation may be easier, so in the B07 failure analysis, the same-color objects must be added.

The idea of handling geometric symmetry using object-centric coordinates is already present in Invariant Slot Attention. We do not claim that simply combining slots with geometric information constitutes a new contribution. The detailed comparison between the paper and the code is continued in D01. [Official Paper](https://proceedings.mlr.press/v202/biza23a.html).
