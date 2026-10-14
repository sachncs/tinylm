# Plan: make tinylm open-source ready and production ready

## Context
`tinylm` (BabyLM 2026 small LMs) already has strict typing, a 97%-coverage gate, pinned `uv.lock`, a docs site and a Pages workflow. Gaps found by inspection:
- No LICENSE (user chose **MIT**), CLAUDE.md, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, CHANGELOG, CITATION.cff, issue/PR templates, Dependabot.
- CI only builds the docs site; the quality gate (`scripts/gate.sh`) runs only in a local pre-commit hook.
- `pyproject.toml` lacks license, authors, classifiers, URLs, keywords; `torch`, `tokenizers`, `transformers` are unbounded; no `py.typed`.
- Root `plan.md` is the project status doc and collides with the harness `plan.md`; `.gitignore` misses `.hypothesis/`, `site/dist`, `.DS_Store`, build artefacts.
- 5 GB `runs/`, 706 MB `data/`, 142 MB `external/` are gitignored (good); need a check that no large/untracked-but-referenced files or personal paths leak into tracked files, and that data/weights licensing is documented.

## Scope (allowed changes)
Repo root files, `.github/`, `pyproject.toml`, `.gitignore`, `README.md`, `docs/` (docs only), `src/tinylm/py.typed`, small fixes the gate demands. **No** changes to model/training behaviour, experiment results, `runs/`, `data/`, `external/`. No publishing, tagging, pushing or PyPI upload without asking.

## Harness files
Written to `harness/` (`plan.md`, `checks.md`, `progress.md`) so the existing root `plan.md` (project status) is not clobbered. Root `plan.md` moves to `docs/status.md` via `git mv`, links updated. Recoverable state: work on branch `oss-ready`; no commits to `main`.

## Steps (each: output, check). P = parallelisable (disjoint files)
1. **Baseline** – run `./scripts/gate.sh`, record result and `git status`. Check: gate exit code. Reads all; writes `harness/progress.md`.
2. **Licence & metadata (P)** – `LICENSE` (MIT, holder from git author, year 2026), `pyproject.toml` metadata (license, authors, classifiers, urls, keywords, version bounds on torch/tokenizers/transformers consistent with `uv.lock`), `src/tinylm/py.typed`. Check: `uv build` succeeds; wheel contains LICENSE and py.typed; `uv lock --check`.
3. **Community docs (P)** – `CONTRIBUTING.md` (setup, gate, test-first rule), `SECURITY.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `CHANGELOG.md` (Keep a Changelog, from 0.2.0), `CITATION.cff`, `.github/ISSUE_TEMPLATE/*`, `PULL_REQUEST_TEMPLATE.md`. Check: `cffconvert`-style YAML parse; links resolve; markdown lint of relative links via script.
4. **CI (P)** – `.github/workflows/ci.yml`: ubuntu + macos, `astral-sh/setup-uv`, Python 3.14, runs the same four gate commands as `scripts/gate.sh` (single source: CI calls the script), least-privilege `permissions`, concurrency, actions pinned to SHAs; `dependabot.yml` (pip/uv, npm in `site/`, actions). Check: `actionlint` if available, else YAML parse + dry read; gate passes locally. Real CI run is unresolved until pushed (needs user approval).
5. **Hygiene** – extend `.gitignore`; `git mv plan.md docs/status.md`; scan tracked files for absolute personal paths, emails, tokens (`git grep`), large files (>1 MB) and untracked artefacts; ensure `.coverage`/`.hypothesis` untracked. Check: scan output empty or justified.
6. **README & docs polish** – badges (CI, licence), install, quickstart, citation, contributing, licence section, data/weights terms (BabyLM data and model-weight licence stated, flagged unresolved if unknown). Check: every relative link and path in README exists (script); site still builds (`npm ci && npm run build` in `site/`).
7. **Production hardening review** – read-only review of `src/` and `scripts/` for: unpinned downloads without checksum (`fetch.py`), unsafe `torch.load`/pickle, bare excepts, hard-coded paths, missing error messages at CLI entry points. Findings fixed test-first only if small and behaviour-preserving; otherwise listed in `checks.md` as unresolved.
8. **CLAUDE.md** – new file: commands, gate, hooks (`scripts/hooks/pre-commit`, `post-commit`, `install.sh`), CI, harness rules, mistakes-to-avoid. Check: every command listed runs.
9. **Full verification** – gate, `uv build`, site build, link check, secret/path scan, wheel install smoke test in a clean venv (`import tinylm`). Fill `harness/checks.md`.

Parallelism: steps 2, 3, 4 touch disjoint files and can run as subagents after step 1; steps 5–9 are sequential in the main session (5 and 6 both touch README/docs paths).

## Completion checklist (evidence)
- [ ] LICENSE (MIT) exists; wheel/sdist include it — `uv build` listing
- [ ] Gate passes on saved tree — `gate.sh` output
- [ ] CI workflow present and parses; local gate equals CI commands — file + diff
- [ ] All community files exist and open; links valid — link-check output
- [ ] Secret/personal-path/large-file scan clean — scan output
- [ ] Clean-venv install imports `tinylm` — smoke output
- [ ] Docs site builds — `npm run build` exit 0
- [ ] CLAUDE.md contains every rule learned; `checks.md`, `progress.md` saved
- Unresolved by design: first real GitHub CI run, PyPI publish, GitHub repo settings (branch protection, security advisories), weights/data licence if terms unknown.

## Limits
Three correction attempts per step, then report. Run limit: this session; stop at all-pass or a blocker. Ask before any push, tag, release, PyPI upload or remote setting.

## Verification (end to end)
`./scripts/gate.sh && uv build && (cd site && npm ci && npm run build)`, then the link/secret scans and clean-venv install from `harness/checks.md`.
