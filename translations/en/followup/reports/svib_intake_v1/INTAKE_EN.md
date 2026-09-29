# SVIB Representative Task Collection and Protocol Inspection

I only selected dSprites / Single Atomic (Shape-Swap) from the public example repository linked in the official project. The original image is 128×128 and this preview does not have a object mask. There is the correct metadata and image pair.

Official code commit: `7eec57ef3c2ea7a2c7389cb6cd470eb3600fa57c`. Public example commit: `23586681bf0d79fba7e2f1e997964a3edd6d666e`.

| Division | Example count | Input combination count | Target combination count | Input overlaps with test input combination | Target overlaps with test input combination | Input=Target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Test | 100 | 12 | 40 | 12 | 12 | 26 |
| Train/alpha-0.0 | 100 | 4 | 16 | 0 | 2 | 25 |
| Train/alpha-0.2 | 100 | 16 | 47 | 0 | 9 | 32 |
| Train/alpha-0.4 | 100 | 28 | 55 | 0 | 8 | 29 |
| Train/alpha-0.6 | 100 | 40 | 55 | 0 | 11 | 29 |

All 500 metadata pairs followed the rule of only changing the shapes of two objects while maintaining their color, position, and size. Some examples have identical shapes of the two objects, meaning the input and target images are the same. Therefore, a baseline that does not change anything must be included, and the performance of examples that actually involve changes must also be reported separately.

The learning/test separation of input combinations was confirmed in this preview. However, some combinations similar to test inputs appear in the learning target images. This is a recorded exposure condition that includes the scope of evaluating generalization of input combinations for this task. Cases where image-to-image learning and source-only representation learning are distinguished, and supervised pretraining is performed by mixing all input and target images, should be marked separately. This observation alone does not definitively indicate a leakage across the entire official benchmark.

The official generation code saves the RGB renderer array to cv2.imwrite and the official reading code uses PIL RGB. When creating an evaluator that compares metadata color and saved PNG color, the channel order must be checked. The original protocol that learns and evaluates the input and target PNGs unchanged is not arbitrarily changed.

The next task is to connect this form of data adapter with the identity baseline and a reduced prediction model. Each of the 100 examples in this preview is not reported as part of the full 64,000 training/8,000 test benchmark results.

[Official benchmark description](https://systematic-visual-imagination.github.io/) · [Official code](https://github.com/systematic-visual-imagination/svib) · [Official examples](https://github.com/systematic-visual-imagination/svib-samples)
