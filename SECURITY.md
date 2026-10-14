# Security policy

## Supported versions

Only the latest release on `main` receives fixes.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private reporting:
<https://github.com/sachncs/tinylm/security/advisories/new>. Expect an
acknowledgement within a week.

## Scope notes

- Training and evaluation download data and models from Hugging Face; downloads
  are checksummed against `manifests/`. Report any path that skips that check.
- Checkpoints are loaded with `safetensors`; do not load untrusted pickles.
