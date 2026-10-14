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
| 3 CITATION.cff schema-validated | pass | `cffconvert --validate`: valid, schema 1.2.0 |
| 4 ci.yml / dependabot.yml present | pass | files written; same script as local gate |
| 4 ci.yml executed on GitHub | pass | PR #1: ubuntu, macos gate and wheel smoke test all success after 3 fixes (python discovery, `+gil` unsupported by uv 0.13) |
| 4 actionlint | pass | actionlint 1.7.12 on both workflows, no findings |
| 4 Actions pinned by SHA | pass | SHAs looked up with `gh api`, tag kept as comment, both workflows |
| 5 No personal paths/secrets in tracked files | pass | `git grep` empty |
| 5 No tracked file > 1 MB except lockfiles | pass | largest non-lock 32 KB |
| 6 README/docs say MIT, weights terms | pass | README, docs/models.md, docs/status.md edited |
| 6 Docs site builds | pass | 14 pages built |
| 7 Unsafe load | pass | `torch.load(weights_only=True)` made explicit; 475 tests pass |
| 7 Download checksums in fetch.py | pass | read in full: 40-hex revision enforced, staged download, committed manifest is trust anchor and never overwritten, existing data only verified |
| 8 CLAUDE.md written | pass | file in repo root |
| 9 Gate in clean dev-only env (CI-equivalent) | pass | 475 passed, 99.64% |
| 9 Wheel clean-venv import | pass | import ok on GIL 3.14 |
| Weights licence positioned | pass | docs/weights-license.md: non-commercial research only; per-source terms from CHILDES ground rules, OTA/Lancaster BNC, OPUS pages |
| Switchboard, BNC edition, Wikipedia, Gutenberg terms | unresolved | low/medium confidence in the table; not read from primary licence texts |
| GitHub private vulnerability reporting | pass | API returns enabled=true |
| PyPI publish | unresolved | not done: irreversible name claim and needs credentials; awaiting explicit go-ahead |
