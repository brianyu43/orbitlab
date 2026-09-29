# A04 Change in evaluation method — Completion criteria based on user instructions

2026-09-23. User instruction: “There are too many 240, but you've already done up to 10, so do the rest yourself.”

The A04 in the existing `PLAN_KO.md` required actual human responses for 240 pages. Following this directive, the already submitted human responses have been preserved, and the rest have been changed to be visually judged by AI. This does not mean that the complete 240 independent human verification has been completed.

In the actual CSV, 9 of Q001–Q009 were stored. The remaining 231, including the unstored Q010, were determined by AI. From images with no model or correct answer to all reported images, AI fixed each AI decision with a hash and then published the correct answer.

The revised completion criteria are (1) preserving the original human responses, (2) AI visual assessment and individual reasons for the remaining all IDs, (3) no omissions or duplicates in 240 IDs, (4) separation of human/AI sources, (5) correct answer comparison and aggregation verification, (6) reflection of the report and replication annex. It does not replace existing automatic evaluation scores or claim the consistency between human and AI assessments.

The experimental scope and existing results for A01–A03b and A05–D04 will not be changed. The original plan, experiment ZIP, and previous completion review will be preserved as records at the time.

[Judgment result](reports/mixed_review_v1/aggregate_v1/RESULTS_EN.md) · [Data verification](../../../followup/reports/mixed_review_v1/verification.json)
