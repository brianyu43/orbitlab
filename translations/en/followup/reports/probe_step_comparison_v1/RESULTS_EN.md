# Comparison of the same material for 3,000 and 6,000 learning expressions

We did not change the original weights, but read 12 checkpoints from two training lengths using the CPU in the same train/validation/test/OOD environments. The standardization and reader were trained only on the train set, and C was selected based on validation accuracy. Each sample's four rotations are grouped into the same scene. We aligned the model, initialization, full representation space, and reader settings.

The table shows the three initialization mean accuracy (%) and difference (%p). The independent data generation seed is one, and the continuous training intermediate checkpoint was not saved and compared. The 6,000 values in this table are also the result of re‑tuning the decoder under the same CPU conditions, so they may differ slightly from the rounded value of the previous MPS‑encoding‑based diagnosis. It is a different indicator from the success rate of generation.

| Model / Reading items | Material | 3,000 times | 6,000 times | Difference (%p) |
| --- | --- | ---: | ---: | ---: |
| aug / shape/linear | test | 45.70 | 43.29 | -2.41 |
| aug / shape/linear | ood | 25.36 | 26.79 | +1.43 |
| aug / shape/rbf | test | 79.13 | 78.87 | -0.26 |
| aug / shape/rbf | ood | 44.34 | 52.21 | +7.88 |
| aug / color/linear | test | 99.48 | 98.99 | -0.49 |
| aug / color/linear | ood | 99.54 | 99.48 | -0.07 |
| aug / color/rbf | test | 98.40 | 98.08 | -0.33 |
| aug / color/rbf | ood | 99.48 | 99.51 | +0.03 |
| aug / rotation/linear | test | 35.32 | 33.07 | -2.25 |
| aug / rotation/linear | ood | 39.32 | 37.63 | -1.69 |
| equivariant / shape/linear | test | 41.67 | 45.54 | +3.87 |
| equivariant / shape/linear | ood | 18.75 | 21.48 | +2.73 |
| equivariant / shape/rbf | test | 71.22 | 75.00 | +3.78 |
| equivariant / shape/rbf | ood | 32.81 | 49.22 | +16.41 |
| equivariant / color/linear | test | 94.76 | 96.94 | +2.18 |
| equivariant / color/linear | ood | 97.01 | 99.48 | +2.47 |
| equivariant / color/rbf | test | 93.23 | 95.83 | +2.60 |
| equivariant / color/rbf | ood | 95.70 | 98.57 | +2.86 |
| equivariant / rotation/linear | test | 32.42 | 37.76 | +5.34 |
| equivariant / rotation/linear | ood | 38.80 | 42.84 | +4.04 |

The full shape reading of C4 was 41.67% linear and 75.00% nonlinear in the general test. In the non-learned combination, it was 32.81% nonlinear. Although there were more clues to read shape information in expressions that were learned for a longer period, linear reading or non-learned combinations remain weak. In aug, the shape reading in the general test did not increase.

The success of a simple reader is not evidence that it changes properties independently or accurately generates images. The previously completed 6,000 m0/m2/m13 diagnoses were preserved as is, and the range of this additional comparison is the full spatial domain. It is a search conducted on the same evaluation data and not a new independent verification experiment.

[Overall score](../../../../../followup/reports/probe_step_comparison_v1/summary.json) · [Input/selection rules](../../../../../followup/reports/probe_step_comparison_v1/protocol.json) · [Replay verification](../../../../../followup/reports/probe_step_comparison_v1/verification.json) · [Existing 6,000 partial space diagnosis](../../../../../followup/reports/probes_6000_v1/summary.json)
