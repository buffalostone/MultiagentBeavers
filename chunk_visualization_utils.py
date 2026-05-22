from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected file was not found: {path}")
    return pd.read_csv(path, low_memory=False)


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _build_chunk_year_outputs(
    *,
    run_root: Path,
    stem: str,
    last_year: int,
) -> list[tuple[str, str]]:
    outputs_chunk: list[tuple[str, str]] = []
    for year in range(1, int(last_year) + 1):
        counter_path = run_root / "year_cycles" / f"{stem}_site_counter_dem_year{year:02d}.csv"
        updated_path = run_root / "year_cycles" / f"{stem}_year{year}.csv"
        if not counter_path.exists() or not updated_path.exists():
            return []
        outputs_chunk.append((str(counter_path), str(updated_path)))
    return outputs_chunk


def _load_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _coerce_erosion_frame(df_erosion: pd.DataFrame, df_updated: pd.DataFrame) -> pd.DataFrame:
    if not df_erosion.empty:
        return df_erosion
    if {"x", "y", "elevation"}.issubset(df_updated.columns):
        return df_updated.loc[:, ["x", "y", "elevation"]].copy()
    return df_erosion


def _load_combined_outputs(
    *,
    combined_root: Path,
    stem: str,
    last_year: int,
    df_orig: pd.DataFrame,
    chunk_col: str | None,
) -> dict[str, Any] | None:
    counter_path = combined_root / f"{stem}_all_chunks_year{last_year:02d}_counter_dem.csv"
    volume_path = combined_root / f"{stem}_all_chunks_site_volume_year{last_year:02d}.csv"
    erosion_path = combined_root / f"{stem}_all_chunks_2d_erosion_year{last_year:02d}.csv"
    updated_path = combined_root / f"{stem}_all_chunks_year{last_year:02d}_updated_dem.csv"

    required = [counter_path, volume_path, updated_path]
    if not all(path.exists() for path in required):
        return None

    df_counter = _read_csv(counter_path)
    df_volume = _read_csv(volume_path)
    df_updated = _read_csv(updated_path)
    df_erosion = _coerce_erosion_frame(_load_or_empty(erosion_path), df_updated)

    suffix = " | combined saved outputs"
    if chunk_col and chunk_col in df_counter.columns:
        n_chunks = int(df_counter[chunk_col].dropna().nunique())
        suffix = f" | combined {n_chunks} chunks by {chunk_col}"

    return {
        "stem": stem,
        "run_root": combined_root,
        "last_year": last_year,
        "label_suffix": suffix,
        "df_orig": df_orig,
        "df_counter": df_counter,
        "df_volume": df_volume,
        "df_erosion": df_erosion,
        "df_updated": df_updated,
        "counter_path": counter_path,
        "volume_path": volume_path,
        "erosion_path": erosion_path,
        "updated_path": updated_path,
    }


def _rebuild_chunk_outputs_from_disk(
    *,
    runs_base_dir: str,
    stem: str,
    last_year: int,
    chunk_col: str | None,
) -> dict[Any, dict[str, Any]]:
    by_chunk_root = Path(runs_base_dir) / "by_chunk"
    if not by_chunk_root.exists():
        return {}

    chunk_outputs: dict[Any, dict[str, Any]] = {}
    for chunk_dir in sorted(p for p in by_chunk_root.iterdir() if p.is_dir()):
        run_dirs = sorted(
            p for p in chunk_dir.iterdir()
            if p.is_dir() and (p / "year_cycles").exists()
        )
        if not run_dirs:
            continue

        selected_run_root: Path | None = None
        outputs_chunk: list[tuple[str, str]] = []
        for run_root in sorted(run_dirs, key=lambda path: path.stat().st_mtime, reverse=True):
            outputs_chunk = _build_chunk_year_outputs(
                run_root=run_root,
                stem=stem,
                last_year=last_year,
            )
            if outputs_chunk:
                selected_run_root = run_root
                break
        if selected_run_root is None or not outputs_chunk:
            continue

        chunk_label = chunk_dir.name
        chunk_id: Any = chunk_label
        if chunk_col:
            prefix = f"{chunk_col}_"
            if chunk_label.startswith(prefix):
                chunk_id = chunk_label[len(prefix):]

        chunk_outputs[chunk_id] = {
            "chunk_label": chunk_label,
            "run_root": selected_run_root,
            "outputs": outputs_chunk,
            "df_final": None,
        }

    return chunk_outputs


def load_final_visualization_inputs(
    *,
    run_by_chunk: bool,
    chunk_outputs: dict[Any, dict[str, Any]] | None,
    outputs: list[tuple[str, str]],
    preview_df_world: pd.DataFrame | None,
    csv_path: str,
    n_years: int,
    runs_base_dir: str,
    chunk_col: str | None = None,
    preview_chunk_id: Any | None = None,
) -> dict[str, Any]:
    stem = Path(csv_path).stem
    last_year = int(n_years)
    df_orig_full = _read_csv(Path(csv_path))
    df_orig_preview = preview_df_world.copy() if preview_df_world is not None else df_orig_full.copy()

    if run_by_chunk:
        combined_root = Path(runs_base_dir) / "combined_final"
        if not chunk_outputs:
            chunk_outputs = _rebuild_chunk_outputs_from_disk(
                runs_base_dir=runs_base_dir,
                stem=stem,
                last_year=last_year,
                chunk_col=chunk_col,
            )
        if not chunk_outputs:
            combined_inputs = _load_combined_outputs(
                combined_root=combined_root,
                stem=stem,
                last_year=last_year,
                df_orig=df_orig_full,
                chunk_col=chunk_col,
            )
            if combined_inputs is not None:
                return combined_inputs
            raise ValueError(
                "RUN_BY_CHUNK is enabled, but no chunk outputs were found in memory or under runs/by_chunk, "
                "and no reusable combined_final outputs were available."
            )

        combined_root.mkdir(parents=True, exist_ok=True)
        counter_frames: list[pd.DataFrame] = []
        volume_frames: list[pd.DataFrame] = []
        erosion_frames: list[pd.DataFrame] = []
        updated_frames: list[pd.DataFrame] = []

        for chunk_id, meta in chunk_outputs.items():
            outputs_chunk = meta.get("outputs") or []
            if not outputs_chunk:
                raise ValueError(f"No yearly outputs were found for chunk {chunk_id!r}.")

            counter_path = Path(outputs_chunk[-1][0])
            updated_path = Path(outputs_chunk[-1][1])
            chunk_run_root = counter_path.parent.parent
            volume_path = chunk_run_root / "site_volume" / f"{stem}_site_volume_year{last_year:02d}.csv"
            erosion_path = chunk_run_root / "2d_erosion" / f"{stem}_2d_erosion_year{last_year:02d}.csv"

            counter_frames.append(_read_csv(counter_path))
            volume_frames.append(_read_csv(volume_path))
            erosion_frames.append(_read_csv(erosion_path))
            updated_frames.append(_read_csv(updated_path))

        df_counter = _concat_frames(counter_frames)
        df_volume = _concat_frames(volume_frames)
        df_updated = _concat_frames(updated_frames)
        df_erosion = _coerce_erosion_frame(_concat_frames(erosion_frames), df_updated)

        counter_path = combined_root / f"{stem}_all_chunks_year{last_year:02d}_counter_dem.csv"
        volume_path = combined_root / f"{stem}_all_chunks_site_volume_year{last_year:02d}.csv"
        erosion_path = combined_root / f"{stem}_all_chunks_2d_erosion_year{last_year:02d}.csv"
        updated_path = combined_root / f"{stem}_all_chunks_year{last_year:02d}_updated_dem.csv"

        df_counter.to_csv(counter_path, index=False)
        df_volume.to_csv(volume_path, index=False)
        df_erosion.to_csv(erosion_path, index=False)
        df_updated.to_csv(updated_path, index=False)

        suffix = f" | combined {len(chunk_outputs)} chunks"
        if chunk_col:
            suffix += f" by {chunk_col}"

        return {
            "stem": stem,
            "run_root": combined_root,
            "last_year": last_year,
            "label_suffix": suffix,
            "df_orig": df_orig_full,
            "df_counter": df_counter,
            "df_volume": df_volume,
            "df_erosion": df_erosion,
            "df_updated": df_updated,
            "counter_path": counter_path,
            "volume_path": volume_path,
            "erosion_path": erosion_path,
            "updated_path": updated_path,
        }

    if not outputs:
        raise ValueError("No run outputs were found for visualization.")

    last_counter_path, last_updated_path = outputs[-1]
    run_root = Path(last_counter_path).parent.parent
    counter_path = Path(last_counter_path)
    volume_path = run_root / "site_volume" / f"{stem}_site_volume_year{last_year:02d}.csv"
    erosion_path = run_root / "2d_erosion" / f"{stem}_2d_erosion_year{last_year:02d}.csv"
    updated_path = Path(last_updated_path)
    df_updated = _read_csv(updated_path)

    suffix = ""
    if preview_chunk_id is not None and chunk_col:
        suffix = f" | {chunk_col}={preview_chunk_id}"

    return {
        "stem": stem,
        "run_root": run_root,
        "last_year": last_year,
        "label_suffix": suffix,
        "df_orig": df_orig_preview,
        "df_counter": _read_csv(counter_path),
        "df_volume": _read_csv(volume_path),
        "df_erosion": _coerce_erosion_frame(_load_or_empty(erosion_path), df_updated),
        "df_updated": df_updated,
        "counter_path": counter_path,
        "volume_path": volume_path,
        "erosion_path": erosion_path,
        "updated_path": updated_path,
    }
