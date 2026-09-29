# OrbitLab follow-up research v3

This is the execution of `completion_v2/NEXT_RESEARCH_PLAN_KO.md`, authorized on 2026-09-29. The earlier release is immutable. A failed research gate is a result, not permission to declare the hypothesis successful. The full scope remains R1 → R3 → R2 → R4 → R5.

**Experiments stopped at the user's request on 2026-09-30 KST.** No automatic restart is authorized. Multiple Atomic plain/C4 checkpoints are retained at 5 epochs (9,000/36,000 updates); the full research scope remains incomplete. See [the stop record](STOPPED_KO.md).

- `SCOPE_KO.md`: requirements and evidence needed for closure.
- `status.json`: stage ledger; not itself proof of completion.
- `r1/`, `r3/`, `r2/`, `r4/`: completed development studies and failure analyses. None met its frozen expansion gate; conditional confirmation was not automatically expanded.
- `r5/disks/RESULTS_KO.md`: matched disk perception/dynamics/decoder integration, fully replayed; development gate failed.
- `r5/external_report/RESULTS_KO.md`: external SVIB report at the stopped boundary. Incomplete cells remain explicit.
- `r5/READER_DISTRIBUTION_AUDIT_KO.md`: post-hoc diagnosis of the imperfect measurement instrument, without changing frozen scores.
- `r5/GPU_SCHEDULING_AMENDMENT.md`: measured Mac MPS scheduling, with training budgets preserved.
- `NEXT_STUDY_DRAFT_KO.md`: conditional next-study design; not a completed or newly started experiment.
- `REPRODUCE_KO.md`: exact artifact, payload-restoration and fresh-forward replay commands, with untested clean-install/retraining boundaries.
- `review/RESOURCE_LEDGER_KO.md` and `review/SCOPE_AUDIT_KO.md`: measured costs and requirement-by-requirement evidence snapshots.
- `review_package.py plan`: provisional bundle inventory. Final packaging is gated on completed scientific scope and final closure evidence.

Scripts and comments are in English; explanatory reports are in Korean. No paid compute or messages to other people are part of this execution. The user separately authorized public GitHub publication; published snapshots and the live worktree are distinct. A genuinely independent human study cannot be replaced by AI ratings.

Generate an up-to-date external report with `.venv/bin/python research_v3/r5_external_report.py`. Add `--require-complete` only when all required external cells should be present; missing cells cause failure instead of a partial completion claim. The report consumes verified artifacts and does not train or change selection.
