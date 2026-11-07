"""Dependency-free P0 smoke check for package and manifest contract."""

from __future__ import annotations

import json
from pathlib import Path

from prism import __version__
from prism.data.inventory import FIELDS, classify


def main() -> None:
    schema = Path("data/manifests/dataset_sample.schema.json")
    payload = json.loads(schema.read_text(encoding="utf-8"))
    assert payload["title"] == "PRISM DataSample"
    assert "input_representation" in payload["properties"]
    assert "alignment_verified" in payload["properties"]
    assert len(FIELDS) >= 15
    assert classify(Path("frame.dng")).startswith("bayer_dng_candidate")
    assert classify(Path("raw.mat")) == "mat_contents_uninspected"
    print(f"PRISM {__version__}: P0 smoke check passed; dataset schema and conservative inventory are available.")


if __name__ == "__main__":
    main()

