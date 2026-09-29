# Latest status — Original collection completed, experiment in progress

2026-09-23 ZIP and Word files were retrieved from iCloud Drive, and access restrictions were resolved. The original MPS doctor/smoke and pilot were passed. Currently, 29 experiments in `configs/experiment_matrix.json` are being executed sequentially, and the actual completion status is recorded in `reports/matrix_status.json`, each run's COMPLETE.json, and the logs. The following details are the history before retrieval. The status of each run will be updated in the final report.

---

# Progress status — 2026-09-23

## Complete

1. Check the recent conversation "Expressive Theory Experiment Plan" and retrieve the text to `sources/conversation_plan.md`.
2. In the original conversation, check the download buttons for `OrbitLab_starter.zip` and `OrbitLab_Mac_실험계획서.docx`. The attachments in this conversation are confirmed to be of two types. Do not claim to have collected all attachments from previous conversations.
3. Set up the work folder `/Users/xavier/Documents/dev/orbitlab` and the original preservation, execution, and reporting structure.
4. Specify the 15 plans mentioned in the conversation as work/completion criteria, experimental contracts, evaluation, termination/recovery items. See `EXECUTION_PLAN_KO.md`.
5. Check M5 Pro, arm64, 64GB. Install PyTorch 2.14.0, NumPy 2.5.3, Pillow 12.3.0 and dependencies in a Python 3.13.14-specific venv. Record the full version lock history.
6. CPU/MPS values and C4 arithmetic test completed without relying on the original data. Actual host MPS availability, forward maximum absolute difference 2.98e-7, gradient difference 3.35e-8, FFT round-trip error 1.49e-7. No NaN/Inf during testing. The code does not replace the doctor/smoke from the original data.
7. Check the official PyTorch documents, G-CNN, Flow Matching, DINOv3, V-JEPA 2.1, FloWM, RAE sources and distinguish the basis/scope of the plan.

## Execution evidence

| Execution | Results |
|---|---|
| python3.13 -m venv .venv; pip install torch numpy pillow | Installation successful, project internal environment |
| .venv/bin/python scripts/preflight.py reports/preflight_sandbox.json | CPU check passed, MPS unavailable in sandbox; script exit 1 |
| .venv/bin/python scripts/preflight.py reports/preflight_host.json | Both CPU/MPS checks passed on the actual host; script exit 0 |
| pip freeze | generate reports/environment.lock.txt |

Note: The first sandbox call resulted in the entire shell exit code being 0 after running pip freeze later. The preflight itself returned 1 due to MPS being unavailable. Consequently, a separate host re-check was performed. The 10-step time of the diagnostic CNN is not used for OrbitLab's learning speed prediction.

## Original collection limit

- The conversation reading tool returned the reference to the text file as `chatgpt-content-reference` and the downloadable attachment was empty.
- The ZIP button and Word attachment download button in the web conversation returned Chrome `ERR_BLOCKED_BY_CLIENT` after moving the content URL.
- Direct download of the verified ZIP content URL returned HTTP 403 due to sandbox DNS restrictions and when attempting to reestablish a secure network.
- The app's conversation page was opened, but the computer control tool refused access to the app for security reasons. No bypass of internal data or authentication information was performed.
- This is not a refusal of shell automatic approval review. The network retry itself was approved, but content access failed.

It is in a state where it has requested the user to save the two original files in `sources/originals/`. It does not mark the original download completed, code review completed, or learning execution completed.

## Status by episode

| Episode | Status |
|---|---|
| 1 | Partial completed: Independent environment check completed; ZIP/Word retrieval and original code verification waiting |
| 2–15 | Detailed execution plan writing, experiment not carried out |

## Next task after file arrival

1. Verify the exact file names in Downloads and `sources/originals/` and preserve the original. Verify the hash, CRC, and compression paths for `scripts/intake_originals.py`.
2. Read the entire original README/PLAN/IMPLEMENTATION_PROMPT/requirements/verification logs/model, split, learn, and evaluate code and write `reports/CODE_AUDIT_KO.md`. Compare the 15 detailed plans in Word with the current plan.
3. Organize a dedicated environment according to the original requirements and perform the original doctor/smoke. Record that actual MPS access requires execution outside the sandbox.
4. Apply modifications necessary to the work copy while preserving the original. Save the 32×32, n=256, batch=16, 200-step pilot file for separate runs. Check loss, foreground, and restoration PNG.
5. Based on the pilot's measurement results, confirm the experimental manifest and proceed in the order of 2–15 times. Update the status table with the actual results of all stages.

There are no learning tasks or scheduled automation currently running. The goal has not yet been achieved.

## Target blocking appreciation

The same original file access restriction continued in the original request turn and in two automatic follow-up turns. Even in the final re-verification, `sources/originals/` was empty and there were no OrbitLab files in Downloads. The previous follow-up turn was a turn that re-verified only the absence of files without any new experimental progress. Guessing the model and requirements of the original ZIP and replacing them does not satisfy the investigation and execution of the original code.

Therefore, the goal is switched to `blocked`. The resumption condition is to save the original ZIP·Word file in this folder or provide an accessible original file. Planning and independent device testing have been completed, but the original retrieval and the first round of original testing, as well as the 2–15 round experiments, are incomplete.
