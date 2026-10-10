# Contributing to Vyom

Thanks for helping. Vyom is a research platform: honest, reproducible results matter more
than impressive ones.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -e ".[dev]"
# exact versions used for the published results: requirements.lock.txt
```

Run from the repository root with `PYTHONPATH=src:.` (Windows PowerShell:
`$env:PYTHONPATH="src;."`) — the `environments/` package lives at the root.

## Before opening a pull request

```bash
ruff check src tests benchmarks environments examples
pytest -q
```

* Add tests for new behaviour (`tests/`). Tests must run offline with the `mock` backend; they
  must never download weights.
* Keep model weights out of git (except tiny, reproducible artefacts such as
  `models/world_model_gridworld.pt`, which `benchmarks/run_all.py` regenerates).
* No proprietary API dependencies; everything must run locally.

## Rules for claims and UI

* Never fabricate outputs. The UI may only display model text, measured telemetry, or content
  labelled with its source (`curated`, `image_metadata`).
* Any number in the docs must come from `benchmarks/run_all.py` (or another committed script)
  with its raw JSON in `benchmarks/results/`. Record seeds, configs, and failures.
* Negative results are welcome and should be reported as such.
* Do not describe the system as AGI, conscious, or a simulation of the brain.

## Benchmarks

```bash
python benchmarks/run_all.py --quick            # smoke (minutes)
python benchmarks/run_all.py                    # full (see docs/BENCHMARKS.md for runtime)
```
