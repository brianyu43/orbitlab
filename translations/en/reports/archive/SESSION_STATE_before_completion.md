# Restart execution memo

2026-09-23. All original ZIP/Word files in the user's iCloud were recovered into sources/originals. The blocked cause has been resolved. Original code audit and completion of original MPS doctor/smoke/pilot. Original 200-step quality was blurry, and after calibration1000, the main experimental 3000-step runs were fixed.

Previous exec session 54655 (termination code 0, 29/29 completed): 29 AE experiments are being sequentially executed in MPS using `.venv/bin/python scripts/run_matrix.py`. The corresponding session must be polled via tools.write_stdin. matrix_status.json is for reference and does not replace the live handle. Always set cwd=/Users/xavier/Documents/dev/orbitlab when running. exec require_escalated is required for MPS access. Preserve stdout status and report log. Do not rerun the same output.

Week 6 experiment ended. In reports/quality_gate.json, B1 seed2 failed with a gate of 90% at 88.3%. The generation AE 6000steps×two models×3 seed correction matrix was frozen in configs/generation_ae_repair.json without lowering the quality standards. The test/OOD results were not reviewed before this decision. The 3000step week experiment is preserved as is.

Current exec session 52212: 6th AE complement run on generation_ae_repair.json. Polling this handle. All 29th runs are completed.

Next order:
1. After confirming the session54655 termination, run MPS with 6 additional runs of `.venv/bin/python scripts/run_matrix.py --manifest configs/generation_ae_repair.json` (do not run with other learning and GPU simultaneously).
2. Run `.venv/bin/python scripts/quality_gate.py configs/generation_ae_repair.json` on CPU. Check the numerical gate in reports/quality_gate_repair.json and inspect each of the six repair runs’ validation/val_reconstruction.png images directly. If they pass, record the actual visual review in reports/repair_visual_review.json as {passed:true,...}. If they fail, investigate without changing the criteria.
3. `.venv/bin/python scripts/postprocess_matrix.py` MPS: Evaluation of 29 test/OOD/latents/interventions/probes. The first 8 (primary6+diagnostic2) have full/band probes & A90. Other conditions are reconstruction/latent metrics.
4. `.venv/bin/python scripts/run_generation.py` MPS: added flow4000steps×6 based on repairAE passed gate, Gaussian/16/32/64 sampling576 each, exactNN, coverage. equal-time exploratory flow s0×2. Partial failures require separate output recovery after log verification.
5. `.venv/bin/python scripts/nn_reference.py` MPS.
6. `.venv/bin/python scripts/final_metrics.py`, scripts/plot_results.py. 29AE CSV/summary and generation_results.csv. Read the actual results and generate reports/README/15 completed checklists/reproducibility commands.
7. Read the Presentations skill and successfully completed operation marker 1 time. I have created `.slides-build/build.mjs` but have not yet run it. The actual results must be written in reports/deck_data.json (cover,conclusions,flowSeries,notes[6]). Runtime Node=/Users/xavier/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node. After exporting the 6-slide PPTX artifact-tool to .slides-build/slide-1..6.png, perform visual inspection on each file. Fix the API/render errors in the build code. Deliverables/OrbitLab_6min.pptx scheduled.
8. All 5 unit tests have already passed, and any necessary modifications will be added to the final new tests. Verify source original hash, manifest, 30? Actual run count (29+ repair 6+ flow 8), numerical finite, all CSV counts, and source plan requirement 15 evidence. Clean reproduction smoke and codehash/package lock/final execution log. Do not update the update_goal complete before achieving the full target.

The tool's goal status remained blocked even after the user file notification, but the work within the user's original goal scope was actually restarted. Since there is no tool to change the goal status to active, the new goal is not replaced by create_goal. When the entire task is completed, try updating_goal complete.
