# S9.5 repository handoff

Historical CV and validation results only. No holdout test evaluation or FINAL selection in this task. Team review is pending.

This upload bundle contains the script, notebook, 27 result files and QA evidence. The original `S9_5_QA_Evidence.zip` (including the verified input snapshot and its original manifest) is retained separately for internal team sharing. It is not part of this repository upload bundle. The external packaging report and `.sha256` refer to that original ZIP, not the GitHub upload ZIP.

For reproduction, obtain the original evidence ZIP from the team, verify its checksum and extract it. From the repository root, with the stated dependencies installed, run:

```bash
python code/c_aggregate_results.py --input-root /path/to/S9_5/inputs/S9_5_inputs --out-dir /path/to/new_results
```

Use a new or empty output directory outside the input directory. See `README_S9_5.md` for the original package layout and limitations.
