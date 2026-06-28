# Phase 1 Notes — VOTS2025 toolkit + dummy tracker

## Environment

| Item | Value |
|------|-------|
| vot-toolkit version | 0.8.0 |
| vot-trax version | 4.0.2 |
| Python version | 3.12.13 (CPython, via uv) |
| Stack used | `vots2025/main` (not bare `vots2025` — toolkit requires subdirectory form) |
| venv path | `/home/alexx/Projects/VOTSTracker/.venv` |

## Dataset

| Item | Value |
|------|-------|
| Dataset name | VOTS2023 Challenge (reused by VOTS2025 main stack) |
| Sequences | 144 |
| Disk used | 27 GB (`workspace/sequences/`) |
| Results size | 355 MB (`workspace/results/`) |

## Deviations from plan

1. **Python 3.12 via uv, not system Python.** System Python is 3.14.4. Used `uv venv --python 3.12` to get 3.12.13 from `~/.local/share/uv/python/`. `uv pip install` used throughout (raw `pip` not on PATH in this setup).
2. **Stack identifier is `vots2025/main`, not `vots2025`.** Bare `vots2025` returned "stack not found"; toolkit requires the subdirectory form.
3. **Dataset downloads lazily on `vot evaluate`, not on `vot configure`.** The `--nodownload` flag and re-running configure are irrelevant; `Workspace.__init__` fetches the dataset automatically when `list.txt` is absent.
4. **Download stalled once at 41% (~13 GB).** A `CLOSE-WAIT` TCP socket indicated the server dropped the connection and the toolkit hung indefinitely. Killed and restarted manually; toolkit resumed from the 61 already-downloaded sequences.
5. **Dummy tracker uses `trax` directly, not a `vot` wrapper.** No `import vot` wrapper exists in vot-toolkit 0.8.x. Used `trax.Server([Region.MASK, Region.POLYGON, Region.RECTANGLE], [Image.PATH], multiobject=True)` from vot-trax directly.
6. **`trackers.ini` needs explicit `python =` field.** Without it, `sys.executable` would be used which is correct here, but being explicit avoids confusion if the venv is activated differently.

## Results

- `vot evaluate dummy` — completed successfully ("Evaluation concluded successfully")
- `vot analysis dummy` — completed successfully
- **Analysis report:** `workspace/analysis/2026-05-20T19-36-41.788700.json`

## Done criteria checklist

- [x] `vot --help` works in the project venv
- [x] `workspace/sequences/` populated with 144 VOTS2023 sequences (27 GB)
- [x] `dummy_tracker/tracker.py` exists and is registered in `trackers.ini`
- [x] `vot evaluate dummy` completed without unhandled errors
- [x] `vot analysis dummy` produced a report (`analysis/2026-05-20T19-36-41.788700.json`)
- [x] This file exists

## Notes for Phase 2

- The `traxpython` protocol with `command = tracker` and `paths = <dir>` is the correct pattern for registering a Python tracker. DAM4SAM will follow the same pattern.
- Re-download risk: if `vot evaluate` is killed mid-sequence, that sequence's partial data may cause issues. The toolkit checks `list.txt` at the dataset level, not per-sequence completeness.
- Keep-awake script at `keep_awake.ps1` must be running when doing long jobs (PowerShell: `powershell.exe -ExecutionPolicy Bypass -File keep_awake.ps1`).
