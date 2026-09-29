# Reading and replication of results

The experiments in this folder compare creation, object editing, and motion prediction separately. They do not combine the accuracy of a single integrated model. They view the entire scope of `PLAN_KO.md` and the current state of `status.json` together. Progress documents from past dates are records at the time and do not replace whether they are currently complete.

## The order to read the results

1. `reports/CURRENT_CAPABILITY_KO.md`: Actual abilities and limitations of each task.
2. `reports/STAGE_A_CONCLUSIONS_KO.md`: The cause of the generation difference and the limitations of the evaluation criteria.
3. `reports/OBJECT_FAILURE_SYNTHESIS_KO.md`: Failure location from object recognition to actual editing.
4. `reports/dynamics_repeats_v1/RESULTS_KO.md`, `reports/dynamics_context_omission_v1/RESULTS_KO.md`, `reports/dynamics_observation_length_v1/RESULTS_KO.md`: repetition, information omission, observation length.
5. `reports/svib_preview_shape_swap_v1/RESULTS_KO.md`: Comparison including negative results from external public examples.
6. `reports/NEAREST_METHODS_REVIEW_V2_KO.md`: Parts overlapping with existing research and still unproven originality.

## Verify the file integrity of ZIP

When you unpack the ZIP file, you will find existing research and `orbitlab/` files together under the top-level `followup/` directory. The original `RELEASE_MANIFEST.json` is the previous distribution list, and the new `BUNDLE_MANIFEST.json` is the complete list contained within the ZIP file. Run the command below using only the Python standard library. The output file must be specified with the new path, and it will not overwrite the existing file.

```sh
python3 orbitlab/followup/verify_bundle_integrity.py orbitlab --output bundle-integrity.json
```

This is a test that verifies that the bits have been preserved. Separate evidence that the numbers are scientifically correct or that the model is useful is found in the replication and aggregation verification of each result.

## CPU prediction playback on other paths

During the experiment, Python and package versions refer to environment records and `requirements.lock.txt`. Below is an example of creating a new virtual environment; package installation depends on the network of the user's environment and whether the version is distributed. The sealed files are not modified, and the virtual environment and verification output are kept outside.

```sh
python3.13 -m venv replay-env
replay-env/bin/python -m pip install -r orbitlab/requirements.lock.txt
PYTHONDONTWRITEBYTECODE=1 replay-env/bin/python orbitlab/followup/replay_bundle_sample.py --output replay-result.json
```

The range of reproduction is 32 first validation batches of 72 length-based estimators and 16 first external test batches of 24 SVIB estimators, respectively. The total is 96 models and predictions, comprising 2,688 instances, including all seeds and models. The same batch size and existing allowable error limits are used. Neither the test that was performed with a fresh overall training nor the test that was re-generated with all predictions are the same. The complete reproduction verification performed during the original work remains separately in each `verification.json` and execution log.

Some path strings in existing checkpoints and logs record the original working location. The above playback tool reads files from the relative location of the current bundle and does not require the original absolute path. It does not guarantee that all past execution scripts will work as-is from arbitrary paths.

## Whole experiment and change experiment

Each `planning_*.md` and `configs/` file fixes the learning count, seed, input information, comparison groups, and evaluation intervals. The `logs/` and execution session logs contain actual execution commands and whether the run was successful. The producer and verifier from previous experiments are tied to code hashes in the settings, so changing the code or checkpoint directly causes verification to fail.

When doing new learning, it does not overwrite the existing `runs/`. It re‑sets input, settings, and the original hash from a separate copy and the new experiment name. Some verifiers record the result files, so full re‑computation is also performed on the work copy. Integrity checks for the storage bundle itself are done using read‑only tools. The numerical consistency of CPU prediction replay and MPS retraining is another claim.

## Person judgment and external materials

`reports/human_review_v1/review.html` is a 240-page evaluation tool that masks the source. If there is no CSV submitted by a real person, human verification will not be completed. The internal answer connection table `private_key.csv` is not shown to the evaluators. The browser visual and interaction inspection on the public screen could not be performed due to the tool's local URL access restrictions. This restriction was not bypassed.

The SVIB example is a reduced experiment of the public preview. It does not replicate the full official data or LPIPS evaluation. External repositories and papers are included for source verification and personal storage, and the original license and copyright are retained. The fact that a ZIP file was created does not imply permission for the public redistribution of the entire material. Network uploads or laboratory transfers are not performed in this work.
