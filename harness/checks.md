# checks.md

| step requirement | verdict | evidence |
|---|---|---|
| 1 Baseline gate on main-equivalent tree | pass | 475 passed, 99.64% coverage |
| 2 LICENSE (MIT) exists, ships in wheel | pass | `unzip -l` shows dist-info/licenses/LICENSE |
| 2 py.typed ships in wheel | pass | `tinylm/py.typed` in wheel, found after install |
| 2 sdist excludes node_modules/runs/data | pass | 71 entries, 0 matches (was 11,280) |
| 2 `uv lock` consistent with new bounds | pass | lock regenerated, 3 lines changed |
| 3 Community files exist | pass | CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, CHANGELOG, CITATION.cff, templates |
| 3 Relative markdown links resolve | pass | script: broken = [] |
| 3 CITATION.cff schema-validated | unresolved | only written, no cffconvert run |
| 4 ci.yml / dependabot.yml present | pass | files written; same script as local gate |
| 4 ci.yml executed on GitHub | unresolved | not pushed (needs approval) |
| 4 actionlint | unresolved | tool not installed |
| 4 Actions pinned by SHA | unresolved | tags used; SHAs not verifiable offline |
| 5 No personal paths/secrets in tracked files | pass | `git grep` empty |
| 5 No tracked file > 1 MB except lockfiles | pass | largest non-lock 32 KB |
| 6 README/docs say MIT, weights terms | pass | README, docs/models.md, docs/status.md edited |
| 6 Docs site builds | pass | 14 pages built |
| 7 Unsafe load | pass | `torch.load(weights_only=True)` made explicit; 475 tests pass |
| 7 Download checksums in fetch.py | unresolved | not audited in depth; manifests/ are checksummed per README |
| 8 CLAUDE.md written | pass | file in repo root |
| 9 Gate in clean dev-only env (CI-equivalent) | pass | 475 passed, 99.64% |
| 9 Wheel clean-venv import | pass | import ok on GIL 3.14 |
| Weights licence | unresolved | corpus terms unchecked; marked research-use only |
| GitHub settings, PyPI publish | unresolved | out of scope without approval |
