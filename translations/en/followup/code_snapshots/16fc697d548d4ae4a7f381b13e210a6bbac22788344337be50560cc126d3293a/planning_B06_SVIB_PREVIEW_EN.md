# B06 Execution plan: Shape-Swap connection experiment of official SVIB preview

2026-09-23. Uses 500 pairs of official dSprites / Single Atomic public examples. This is not a replication of the official benchmark performance for the entire 12 tasks or each of the 64,000 learning and 8,000 evaluation tasks. It relates to the external task linking for the B06 reduction plan, and the full official benchmark execution status is specified separately.

## Material and division

Re-examine the existing fixed commits and 2,000 original file hashes of the official code and examples. Read the storage channel of the 128×128 PNG directly as PIL RGB, and do not arbitrarily modify the color channel according to the metadata. The learning input is only the source image, and the target image, scene properties, and modification area are not input.

Each of the 100 pairs of alpha=0.0/0.2/0.4/0.6 is divided into train 70%, validation 15%, and ID test 15% according to the sorting method and 70%/15%/15% ratio of the official data loader. The separate Test 100 pairs are a common unseen combination evaluation for all alphas. The accuracy of the learning and hold-out source/target images and the 90-degree rotation overlap are tested, and the exposure of the source combination and target combination is recorded separately. The same test reused across different alphas and initializations is not counted as independent data.

## Model and fixed learning budget

1. Includes a comparison of a constant image averaged with the identity that does not change anything and the corresponding train target.
2. The video predictor with a small global bottleneck CNN encoder/decoder is trained from the beginning by connecting the source residual. It does not input known transformation rules or the correct object properties.
3. Compare the C4 model, which applies the same basic predictor in four directions and averages after reverse propagation, by setting the parameter count, initial weights, and mini-batch order. The same 2,000 steps, batch size 8, Adam 0.0005, gradient clip 1, and seed 0/1/2 are fixed. The method is 2 × alpha 4 × initialization 3 = 24 training steps.
4. Evaluate the final checkpoint of all models. Do not select seed, number of training iterations, or checkpoint based on validation and common test results. The training loss is the total pixel MSE.
5. The C4 model uses the same parameters four times, so it is not a comparison of identical calculation costs. It records the actual training time and inference cost. It does not refer to the result of transferring the existing OrbitLab checkpoint or a new data-independent repetition.

## How to read scores and failures

- The `mse` of the official evaluation code is the error divided by the number of images, which is the sum of pixel and channel errors. Both the MSE for each element and the sum of squared errors for each image are recorded, and the 128×128×3 ratio is checked.
- Distinguish between the entire image, the union of the foreground and source/target, actual changed pixels, and the error of pixels that should not be changed. The foreground and the changed area are evaluation masks that use the target.
- In scenes with no actual changes, the change area score is set to zero and the denominator is recorded. Both the "All" and "Changed" and "No Changed" groups are reported. The proportion of scenes where the identity matches is not called Shape-Swap understanding accuracy.
- The difference in model and identity errors is calculated between the same scenes. All 100 external test examples are used, and the last mini-batch is not discarded. It is explicitly stated that it may differ from the default execution of the official evaluation code's batch/drop_last/max_images.
- Output consistency for rotation input is a separate indicator and does not replace external task accuracy. If LPIPS was not run, it is marked as unmeasured and does not mean that the official overall score has been reproduced.
- The table preserves all three fixed initializations per alpha. Using bootstrap reveals that the condition is scene-dependent. Since only the same public preview is used, it is not interpreted as an independent data seed verification experiment.

## Completion evidence

Original preservation, input/answer separation, division and exposure testing, identity/constant comparison, replay of 24 learning and all hold-out predictions, independent pixel metric recalculation, difference in the same scene and fixed example drawings, and a difference table with the official protocol. The report specifies over-fitting and uncertainty of small public examples and the failure of the official full benchmark. Failed results are also reported under the same rules.

Source: [Official project](https://systematic-visual-imagination.github.io/), the fixed commit `tasks/image_to_image/data.py` and `eval.py`, `data_creation/dsprites/create_data.py`. Official code commit `7eec57ef3c2ea7a2c7389cb6cd470eb3600fa57c`, public example commit `23586681bf0d79fba7e2f1e997964a3edd6d666e`.
