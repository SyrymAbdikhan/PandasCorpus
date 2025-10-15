import json
import time
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd


# =========================
# CONFIG
# =========================
CODE_CELLS_PATH = Path("source") / "code_cells.parquet.gzip"
MARKDOWN_CELLS_PATH = Path("source") / "markdown_cells.parquet.gzip"
OUTPUT_DIR = Path("notebooks")
FLUSH_EVERY_SECS = 10


# =========================
# STATS
# =========================
class Stats:
    def __init__(self) -> None:
        self.start_monotonic = time.monotonic()
        self.notebooks_discovered = 0
        self.notebooks_written = 0
        self.code_cells_written = 0
        self.md_cells_written = 0

    def snapshot(self) -> Tuple[int, int, int, int, float]:
        elapsed = time.monotonic() - self.start_monotonic
        return (
            self.notebooks_discovered,
            self.notebooks_written,
            self.code_cells_written,
            self.md_cells_written,
            elapsed,
        )


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        # assume parquet
        df = pd.read_parquet(p)

    return df


def cell_to_nbformat(cell_type: str, source: str) -> dict:
    if cell_type not in {"code", "markdown"}:
        raise ValueError(f"Unsupported cell type: {cell_type}")
    if cell_type == "code":
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source,
        }
    else:
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source,
        }


def make_notebook(cells: List[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(nb: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2, allow_nan=False)


def reconstruct(
    code_path: Optional[str],
    markdown_path: Optional[str],
    out_dir: str,
) -> Tuple[int, int, int]:
    code_df = read_table(code_path) if code_path else pd.DataFrame()
    code_df = code_df.assign(cell_type="code")

    md_df = read_table(markdown_path) if markdown_path else pd.DataFrame()
    md_df = md_df.assign(cell_type="markdown")

    all_cells = pd.concat([code_df, md_df], ignore_index=True)
    all_cells = all_cells.sort_values(["notebook_hash", "cell_id"])

    out_base = Path(out_dir)
    notebooks_written = 0
    total_code_cells = int((all_cells["cell_type"] == "code").sum())
    total_md_cells = int((all_cells["cell_type"] == "markdown").sum())

    stats = Stats()
    stats.notebooks_discovered = all_cells["notebook_hash"].nunique()
    last_flush = time.monotonic()

    print(
        f"[Stats] discovered={stats.notebooks_discovered:_} code_cells={total_code_cells:_} md_cells={total_md_cells:_}"
    )

    for nb_hash, grp in all_cells.groupby("notebook_hash", sort=False):
        cells: List[dict] = []
        for _, row in grp.iterrows():
            cells.append(cell_to_nbformat(row["cell_type"], row["source"] or ""))

        nb = make_notebook(cells)
        out_path = out_base / f"{nb_hash}.ipynb"
        write_notebook(nb, out_path)

        notebooks_written += 1
        stats.notebooks_written += 1

        stats.code_cells_written += int((grp["cell_type"] == "code").sum())
        stats.md_cells_written += int((grp["cell_type"] == "markdown").sum())

        now = time.monotonic()
        if now - last_flush >= FLUSH_EVERY_SECS:
            disc, written_n, code_n, md_n, elapsed = stats.snapshot()
            remaining = max(0, disc - written_n)
            print(
                f"[flush] elapsed={elapsed:.1f}s written={written_n:_}/{remaining:_}",
                flush=True,
            )
            last_flush = now

    return notebooks_written, total_code_cells, total_md_cells


def main():
    print(
        f"Starting recreation of notebooks from '{CODE_CELLS_PATH}' and '{MARKDOWN_CELLS_PATH}'"
    )
    written, n_code, n_md = reconstruct(
        str(CODE_CELLS_PATH), str(MARKDOWN_CELLS_PATH), str(OUTPUT_DIR)
    )
    print(
        f"Done. notebooks={written} code_cells={n_code} markdown_cells={n_md} -> {OUTPUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
