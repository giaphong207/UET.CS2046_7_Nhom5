# Review the local A4/A5 V3 integration

Repository: https://github.com/giaphong207/UET.CS2046_7_Nhom5

The authorized working branch is `integration/a4-a5-v3`, based on `origin/Hưng` at `19cf6d2a6256e3163443bc31360036a6c16ff5b5`. The user fetched, created the branch and ran `git mv research/lead_modeling/a4_a5/src src` locally after approval-service failures. The six source moves are staged; later edits/new files remain unstaged. No commit, push, merge or rebase is authorized for this implementation task.

These review commands are read-only and use the repository root:

```bash
cd /Users/humanormal/Projects/UET.CS2046_7_Nhom5
git status --short --branch
git diff --cached --stat
git diff --stat
git diff --check
git ls-files --others --exclude-standard
```

Review the source moves and unstaged/untracked content together. Production Python belongs only in root `src/diabetes_research/`; component configs, notebooks, ignored data/environments and historical outputs retain their paths. Root installation and execution commands are in [../README.md](../README.md).

Preserve these two pre-existing untracked documents exactly and leave them unstaged:

- `CROSS_BRANCH_PROTOCOL_AUDIT.md`
- `PROTOCOL_FREEZE_PROPOSAL.md`

Their contents have been reviewed as historical evidence; review does not grant permission to overwrite or stage them. Full fingerprints and final Git status are in [V3_INTEGRATION_REPORT.md](V3_INTEGRATION_REPORT.md). Do not use broad staging commands that would include them, datasets, model weights, credentials, caches or experiment outputs.

Use [PR_DESCRIPTION.md](PR_DESCRIPTION.md) as a review description after the user separately authorizes a publication workflow. This local integration has not been published to GitHub. Official experiment release is also separate: fold-local SelectFromModel is implemented, and software QA does not authorize official evaluation or establish research performance. Coordinate one verified shared metrics table with B/C after aligned runs; synthetic results do not count as research outcomes.
