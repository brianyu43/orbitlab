# Final thanks and separate replication bundle

Create a **local repository reproduction ZIP** that includes the original 1,324 files of `RELEASE_MANIFEST.json` along with the subsequent research outputs. Do not modify the original distribution files or publish them online. The original research papers and external examples from prior research are preserved exactly as they are from their original sources and existing licenses, and this personal repository bundle itself does not imply distribution authorization.

1. Compare all IDs and completion evidence in `PLAN_KO.md`. Publicize the actual range, verification files, and remaining portions of each item in a table. If there is no human response, keep A04 as incomplete.
2. Connect the data report and the figures to the raw materials, and verify the range of reproduction. The reproduction of some samples is not called the overall learning reproduction.
3. Recheck the original list and the total SHA-256 and byte count. The subsequent code is stored separately.
4. Collect files from the original list, `planning/`, and subsequent code, configuration, data, checkpoint, original predictions, reports, failure records, and publicly available source materials. Exclude virtual environments, Python caches, macOS metadata, and temporary files in the process of generation. Record the exclusion list.
5. Read each file, calculate the hash, and put it into the ZIP file; if the file changes during writing, it will be stopped with an error. Read all items inside the ZIP file again and verify them separately. The automatic verification output is kept outside the sealed ZIP file.
6. Open the new temporary directory, verify the entire bytes and hashes again, and recalculate the CPU prediction using the stored input/checkpoint. It is checked at the location where the path has changed. Full retraining and separate hardware replication are only claimed when executed.

The acceptance conditions include ZIP path, size, SHA-256 hash, full file integrity verification, specified prediction playback results in other paths, original invariance verification, and incomplete public disclosure for each item. Actual human judgment is not replaced by code or automatic evaluation. In this case, even if a reproduction bundle is created, the entire goal is not marked as complete.
