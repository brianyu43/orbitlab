# People·AI judgment added bundle

[Additional ZIP](../../../../../followup/releases/orbitlab-mixed-review-2026-09-23-v1/orbitlab-mixed-review-2026-09-23-v1.zip) is the evaluation data and revision report conducted after the original experiment ZIP. **9 people + 231 AI models**, and it is not a full human validation. The original 12.53GB experiment ZIP was not changed.

The ZIP file contains original human responses, AI judgments and reasons, hash before the release of the correct answer, review images, correct answer comparison and aggregation, verification codes, changes to the evaluation method by the user, and update reports. It also includes a correct answer table for the analysis stage, so it is not a bundle that is distributed as an unpublished source material to new judges.

[All file verification](../../../../../followup/releases/orbitlab-mixed-review-2026-09-23-v1/verification.json) · [Hash of each file](../../../../../followup/releases/orbitlab-mixed-review-2026-09-23-v1/manifest.json)

Independent aggregation reproduction example: Run the following command in the uncompressed `orbitlab`. Only the Python standard library is required. The output directory is named after the new name.

```sh
python3 followup/aggregate_mixed_review.py --review followup/reports/mixed_review_v1 --key followup/reports/human_review_v1/private_key.csv --out /tmp/orbitlab-mixed-review-recomputed
```

The `verify_mixed_review.py` script, which even verifies the basis of the existing 30 tasks, assumes that this appendix has been combined with the folder structure of the original experimental bundle. It has passed this verification in the current work folder. It is not a test to demonstrate the accuracy of AI judgments or the consistency with human responses.
