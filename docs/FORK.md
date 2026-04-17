# Fork workflow (dspy-slim)

This directory is a **git repository** and the place where **upstream DSPy** is tracked. Parent folder `minimal_dspy/` is optional glue (Korbex, scripts); it may or may not be its own git repo.

**Public fork:** [github.com/Emre-C/dspy-slim](https://github.com/Emre-C/dspy-slim)

## Remotes (recommended layout)

| Remote     | URL | Role |
|-----------|-----|------|
| `origin`  | `https://github.com/Emre-C/dspy-slim.git` | Where you **push** your branch. |
| `upstream` | `https://github.com/stanfordnlp/dspy.git` | Official DSPy; **fetch / merge** to stay current. |

### One-time setup (remotes)

From this repo’s root on disk:

```bash
cd /path/to/minimal_dspy/dspy-slim

git remote set-url origin https://github.com/Emre-C/dspy-slim.git
git remote add upstream https://github.com/stanfordnlp/dspy.git 2>/dev/null || true
git fetch upstream
```

Verify:

```bash
git remote -v
```

## Staying current with upstream

From parent repo root `minimal_dspy/`:

```bash
./scripts/sync_dspy_upstream.sh
```

(`DSPY_SLIM` overrides the default path `…/dspy-slim` if your layout differs.)

Or manually:

```bash
cd dspy-slim
git fetch upstream
git merge upstream/main
# resolve conflicts, then:
cd .. && ./run_minimal_tests.sh
```

**Prefer frequent, small merges** over rare huge ones. After every merge, run `./run_minimal_tests.sh` from `minimal_dspy/`.

## After merging upstream

- Run **`./run_minimal_tests.sh`** (from `minimal_dspy/`).
- If GitHub restored upstream-only workflow files (e.g. full `run_tests.yml`), either remove or rename them again (this fork keeps the full matrix as `run_tests.yml.upstream-full-not-used` so it does not run). The workflow that runs in CI is `.github/workflows/minimal_fork_tests.yml`.
- Update the “last synced” note below (optional but useful).

## Publishing (PyPI)

Official PyPI package name `dspy` is reserved for upstream. If you publish this fork, use a **different distribution name** in `pyproject.toml` (`[project] name = "..."`) while keeping `import dspy` if the package layout is unchanged. Document the install line in `README.md`.

## Last synced (edit when you merge upstream)

- Upstream ref (optional): `main @ ____________`
- Notes: ____________
