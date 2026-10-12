#!/bin/sh
# Points git at the versioned hooks (commit gate and external backup).
set -e
git config core.hooksPath scripts/hooks
echo "hooks installed: $(git config core.hooksPath)"
