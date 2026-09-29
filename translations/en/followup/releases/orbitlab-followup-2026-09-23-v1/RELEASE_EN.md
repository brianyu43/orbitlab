# OrbitLab Follow-up Research Replication Bundle — Final Verification Record

2026-09-23T17:12:45+09:00. The current plan is to complete **30/31 items**, and 240 actual human judgment sheets for A04 remain with 0 responses. No overall goal completion or human verification is claimed.

[Rehyun ZIP](../../../../../followup/releases/orbitlab-followup-2026-09-23-v1/orbitlab-followup-2026-09-23-v1.zip): **12,534,372,531 bytes (approximately 12.53 GB)**. Original and subsequent output files: 70,514 files, uncompressed total: 17,135,158,775 bytes. The file list is in [BUNDLE_MANIFEST.json](../../../../../followup/releases/orbitlab-followup-2026-09-23-v1/BUNDLE_MANIFEST.json).

ZIP SHA-256: `ec13c8fb120fdded3eced9708c6a02a23e44ca207d929f02a347bf04206b4ca0`

Verified items:

- Read all files inside the ZIP again and check the size and SHA-256: [Compression Verification](../../../../../followup/releases/orbitlab-followup-2026-09-23-v1/archive_verification.json).
- Uncompressed the actual files in `/private/tmp/orbitlab-followup-relocated-20260923-v1/orbitlab` and inspected all 70,514 files: [Uncompressed Integrity Check](../../../../../followup/releases/orbitlab-followup-2026-09-23-v1/extracted_integrity.json).
- Using the code and data of that path, it compared the results of storing 72 observation length models and 24 SVIB models, a total of 2,688 CPU sample predictions, with the results of prediction replay: [Predicted replay](../../../../../followup/releases/orbitlab-followup-2026-09-23-v1/relocated_replay.json).
- The hash of the entire file was the same even after playback: [Check after playback](../../../../../followup/releases/orbitlab-followup-2026-09-23-v1/extracted_integrity_after_replay.json).
- The two execution sessions 72212·10086 terminated with exit 0: [Relocated Execution Log](../../../../../followup/releases/orbitlab-followup-2026-09-23-v1/relocated_execution.json), [Execution Log](../../../../../followup/releases/orbitlab-followup-2026-09-23-v1/orbitlab_followup_relocated_verification_20260923_v1.log).

[Final audit of 31 items](scope_audit_final_v1.md) · [Final audit JSON](scope_audit_final_v1.json) · [Execution script](../../../../../followup/releases/orbitlab-followup-2026-09-23-v1/orbitlab_verify_relocated_bundle_v1.py)

This is a re-use of the currently installed Python environment for testing. It does not involve installing a new environment, running on another device, or re-running all learning or additional pre-prediction runs. The existing full prediction test basis is preserved within the ZIP file. 1,324 immutable tests from the original data are also included.

The status documents within ZIP are dated 29/31 before sealing. The re-placement inspection after sealing and the final 30/31 inspection were preserved as separate records in this folder, and the ZIP bytes were not changed. The status and information documents in the current work folder have been updated to reflect the completed status.

This bundle is for local personal storage. It includes reference papers and external codes, so it does not mean permission for public redistribution. When distributing publicly, you must review the license and distribution scope of the material separately.
