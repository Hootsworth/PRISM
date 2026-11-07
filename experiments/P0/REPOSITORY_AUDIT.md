# P0 repository and feasibility audit

## Repository state at audit
- The workspace initially contained ten build-pack documents and `PACK_MANIFEST.json`; no source, application, tests, data, model weights, or Git repository were present.
- The supplied `07_MASTER_AI_BUILD_PROMPT.md` requests an audit before implementation and restricts the first implementation task to P0. The user's request to begin implementation authorizes that P0 work.
- There is no `docs/` subdirectory; the authoritative documents are at the repository root. No conflicting implementation was found.

## Environment
- Python: 3.9.6; pip: 21.2.4.
- Available in the inspected interpreter: NumPy, SciPy, PyTorch, Pillow, PyYAML.
- Not available: rawpy, pytest.
- This environment is sufficient for the dependency-free P0 inventory and smoke path, but not for DNG decode. `pyproject.toml` declares optional data and learned dependencies; no heavyweight packages were installed.
- Workspace is not a Git repository, so commit and clean/dirty provenance are currently unavailable.

## Data and model assets
- No dataset files, model checkpoints, manifests with sample records, or dataset directories were present in the workspace.
- SIDD DNG availability, Raw-RGB MAT shapes, CFA/metadata values, target mapping, license, and capture counts: NOT INSPECTED because no SIDD files are available here.
- Modular Neural ISP code/data, exact layouts, licenses, pairing, and compatibility: NOT INSPECTED because no checkout or data is available here. The build pack's documented candidate layouts are planning information, not verified local assets.
- No Bayer-to-sRGB pairing is established. Training is blocked until a concrete dataset subset is inventoried and pairing validated.

## Traceability status
- P0: foundational package metadata, conservative inventory command, manifest schema, config, and project audit added. A wheel was built and installed into a fresh virtual environment; installed console entry points passed smoke and empty-inventory checks.
- P1–P8: NOT STARTED. There is no DNG decoder or ISP implementation yet.
- Requirements matrix rows remain unimplemented unless separately evidenced; design documents are not evidence of working functionality.

## Conflicts and decisions
- The master prompt refers to files under `docs/`, while this pack stores those same files at its root. This implementation preserves the supplied layout and records the mismatch.
- The prompt requires P0 smoke evidence; this is provided as a dependency-free command. No uncertain dataset interpretation was chosen.
- Human decision needed before P1: provide/mount an authorized SIDD and/or Modular ISP dataset location and the relevant source/license records. Team ID/USNs are also unknown and remain TBD for eventual submission artifacts.

## P0 validation record
- Command: `python3 -m venv --system-site-packages /tmp/prism-p0-clean-venv2`
- Install: `/tmp/prism-p0-clean-venv2/bin/python -m pip install --no-deps --no-build-isolation .`
- Smoke: installed `prism-smoke` passed.
- Inventory: installed `prism-inventory data --output /tmp/prism-p0-clean-inventory.csv` completed; 0 records and a readable CSV header.
- Limitation: this offline installation used system site packages for the preinstalled build backend, and did not resolve optional runtime dependencies. DNG decoding remains unavailable because rawpy is absent.

## Next gated work
1. Run `prism-inventory /path/to/authorized/dataset --output data/manifests/sidd_inventory.csv` (or use `python -m prism.data.inventory ...`).
2. Inspect MAT shapes and DNG metadata with format-aware tools, establish only verified pair mappings, and create capture-level splits.
3. Complete the P1 data feasibility report and review the pairing gate before implementing B0/B1/B2.

