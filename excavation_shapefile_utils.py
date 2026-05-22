from __future__ import annotations

import argparse
import csv
import math
import os
import re
import struct
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from pyproj import CRS as PyprojCRS
except Exception:  # pragma: no cover - optional dependency
    PyprojCRS = None


POINT_SHAPE_TYPE = 1
POLYGON_SHAPE_TYPE = 5


@dataclass(frozen=True)
class ShapefileExportResult:
    geometry: str
    record_count: int
    shp_path: Path
    shx_path: Path
    dbf_path: Path
    prj_path: Path | None
    cpg_path: Path


@dataclass(frozen=True)
class CrsResolution:
    source_key: str | None
    raw_value: str | None
    epsg_code: int | None
    projected_wkt: str | None
    used_fallback_wkt: bool


SHAPEFILE_COMPONENT_SUFFIXES = (".shp", ".shx", ".dbf", ".prj", ".cpg")


def read_site_summary(summary_csv: str | Path) -> dict[str, str]:
    summary_path = Path(summary_csv)
    with summary_path.open("r", newline="", encoding="utf-8-sig") as handle:
        return {
            (row.get("Parameter") or "").strip(): (row.get("Value") or "").strip()
            for row in csv.DictReader(handle)
        }


def infer_site_summary_csv(counter_csv: str | Path) -> Path | None:
    counter_path = Path(counter_csv).resolve()
    for parent in (counter_path.parent, *counter_path.parents):
        matches = sorted(parent.glob("*_site_dem_summary.csv"))
        if matches:
            return matches[0]
    return None


def read_grid_resolution_from_site_summary(summary_csv: str | Path) -> tuple[float | None, float | None]:
    summary = read_site_summary(summary_csv)
    dx = _parse_float(summary.get("Resolution dx (map units)"))
    dy = _parse_float(summary.get("Resolution dy (map units)"))
    return dx, dy


def read_horizontal_crs_wkt(summary_csv: str | Path) -> str | None:
    resolution = resolve_horizontal_crs(summary_csv)
    return resolution.projected_wkt


def resolve_horizontal_crs(summary_csv: str | Path) -> CrsResolution:
    summary = read_site_summary(summary_csv)
    dem_raw_value = (summary.get("DEM CRS") or "").strip()
    dem_epsg = _extract_epsg_code(dem_raw_value)
    dem_wkt = _extract_projected_wkt(dem_raw_value)
    if dem_wkt:
        return CrsResolution(
            source_key="DEM CRS",
            raw_value=dem_raw_value,
            epsg_code=dem_epsg,
            projected_wkt=dem_wkt,
            used_fallback_wkt=False,
        )

    if dem_epsg is not None:
        epsg_wkt = _projected_wkt_from_epsg(dem_epsg)
        if epsg_wkt:
            return CrsResolution(
                source_key="DEM CRS",
                raw_value=dem_raw_value,
                epsg_code=dem_epsg,
                projected_wkt=epsg_wkt,
                used_fallback_wkt=False,
            )

    if dem_epsg is not None:
        for key in ("CRS", "Topographic Map CRS", "RGB imagery CRS"):
            raw_value = (summary.get(key) or "").strip()
            if not raw_value or raw_value.lower() == "none":
                continue
            projected_wkt = _extract_projected_wkt(raw_value)
            if not projected_wkt:
                continue
            if _wkt_contains_epsg(projected_wkt, dem_epsg):
                return CrsResolution(
                    source_key=key,
                    raw_value=dem_raw_value,
                    epsg_code=dem_epsg,
                    projected_wkt=projected_wkt,
                    used_fallback_wkt=True,
                )

    if dem_epsg is not None:
        for key in ("CRS", "Topographic Map CRS", "RGB imagery CRS"):
            raw_value = (summary.get(key) or "").strip()
            if not raw_value or raw_value.lower() == "none":
                continue
            projected_wkt = _extract_projected_wkt(raw_value)
            if projected_wkt:
                return CrsResolution(
                    source_key=key,
                    raw_value=dem_raw_value,
                    epsg_code=dem_epsg,
                    projected_wkt=projected_wkt,
                    used_fallback_wkt=True,
                )

    for key in ("CRS", "Topographic Map CRS", "RGB imagery CRS", "Projection Target EPSG"):
        raw_value = (summary.get(key) or "").strip()
        if not raw_value or raw_value.lower() == "none":
            continue
        projected_wkt = _extract_projected_wkt(raw_value)
        if projected_wkt:
            return CrsResolution(
                source_key=key,
                raw_value=dem_raw_value or raw_value,
                epsg_code=dem_epsg if dem_epsg is not None else _extract_epsg_code(raw_value),
                projected_wkt=projected_wkt,
                used_fallback_wkt=True,
            )

    return CrsResolution(
        source_key="DEM CRS" if dem_raw_value else None,
        raw_value=dem_raw_value or None,
        epsg_code=dem_epsg,
        projected_wkt=None,
        used_fallback_wkt=False,
    )


def load_excavation_cells(
    counter_csv: str | Path,
    *,
    x_col: str = "x",
    y_col: str = "y",
    value_col: str = "excavation_m",
    tile_col: str = "tile_id",
    positive_only: bool = True,
) -> pd.DataFrame:
    usecols = [x_col, y_col, value_col]
    if tile_col:
        usecols.append(tile_col)

    df = pd.read_csv(counter_csv, usecols=usecols, low_memory=False)
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col, value_col]).copy()
    if positive_only:
        df = df.loc[df[value_col] > 0].copy()
    if tile_col and tile_col not in df.columns:
        df[tile_col] = ""
    return df


def export_excavation_heatmap_shapefiles(
    counter_csv: str | Path,
    *,
    site_summary_csv: str | Path | None = None,
    out_dir: str | Path | None = None,
    out_stem: str | None = None,
    geometry: str = "both",
    x_col: str = "x",
    y_col: str = "y",
    value_col: str = "excavation_m",
    tile_col: str = "tile_id",
    cell_width: float | None = None,
    cell_height: float | None = None,
    positive_only: bool = True,
    write_prj: bool = True,
) -> list[ShapefileExportResult]:
    counter_path = Path(counter_csv)
    summary_path = Path(site_summary_csv) if site_summary_csv else infer_site_summary_csv(counter_path)
    prj_wkt = None
    crs_resolution: CrsResolution | None = None

    if summary_path and summary_path.exists():
        if cell_width is None or cell_height is None:
            dx, dy = read_grid_resolution_from_site_summary(summary_path)
            cell_width = cell_width if cell_width is not None else dx
            cell_height = cell_height if cell_height is not None else dy
        if write_prj:
            crs_resolution = resolve_horizontal_crs(summary_path)
            prj_wkt = crs_resolution.projected_wkt

    out_root = Path(out_dir) if out_dir else counter_path.parent
    out_root.mkdir(parents=True, exist_ok=True)
    base_stem = out_stem or _default_output_stem(counter_path.stem)
    export_df = load_excavation_cells(
        counter_path,
        x_col=x_col,
        y_col=y_col,
        value_col=value_col,
        tile_col=tile_col,
        positive_only=positive_only,
    )
    if export_df.empty:
        raise ValueError(f"No excavation cells were found in {counter_path}")
    if crs_resolution is not None:
        validate_coordinate_crs_match(export_df, x_col=x_col, y_col=y_col, crs=crs_resolution, summary_path=summary_path)

    if cell_width is None:
        cell_width = infer_axis_step(export_df[x_col])
    if cell_height is None:
        cell_height = infer_axis_step(export_df[y_col])
    if not cell_width or cell_width <= 0 or not math.isfinite(cell_width):
        raise ValueError(f"Could not infer a valid cell width from {counter_path}")
    if not cell_height or cell_height <= 0 or not math.isfinite(cell_height):
        raise ValueError(f"Could not infer a valid cell height from {counter_path}")

    year_value = _parse_year_from_name(counter_path.stem)
    requested = geometry.lower()
    if requested not in {"polygon", "point", "both"}:
        raise ValueError("geometry must be 'polygon', 'point', or 'both'")

    results: list[ShapefileExportResult] = []
    if requested in {"polygon", "both"}:
        polygon_base = out_root / f"{base_stem}_cells"
        results.append(
            write_excavation_shapefile(
                export_df,
                out_base=polygon_base,
                geometry="polygon",
                x_col=x_col,
                y_col=y_col,
                value_col=value_col,
                tile_col=tile_col,
                year_value=year_value,
                cell_width=cell_width,
                cell_height=cell_height,
                prj_wkt=prj_wkt,
            )
        )
    if requested in {"point", "both"}:
        point_base = out_root / f"{base_stem}_points"
        results.append(
            write_excavation_shapefile(
                export_df,
                out_base=point_base,
                geometry="point",
                x_col=x_col,
                y_col=y_col,
                value_col=value_col,
                tile_col=tile_col,
                year_value=year_value,
                cell_width=cell_width,
                cell_height=cell_height,
                prj_wkt=prj_wkt,
            )
        )

    return results


def zip_shapefile_results(
    results: Iterable[ShapefileExportResult],
    *,
    archive_path: str | Path | None = None,
    root_dir: str | Path | None = None,
) -> Path:
    result_list = list(results)
    if not result_list:
        raise ValueError("No shapefile results were provided for zipping.")

    component_paths = _collect_shapefile_component_paths(result_list)
    if not component_paths:
        raise ValueError("No shapefile component files were found to archive.")

    if root_dir is None:
        root_path = _common_parent(component_paths)
    else:
        root_path = Path(root_dir)
    root_path = root_path.resolve()

    if archive_path is None:
        archive_path = root_path.with_suffix(".zip")
    archive = Path(archive_path)
    archive.parent.mkdir(parents=True, exist_ok=True)

    seen_arc_names: set[str] = set()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in component_paths:
            resolved_path = path.resolve()
            try:
                arcname = resolved_path.relative_to(root_path)
            except ValueError:
                arcname = Path(resolved_path.name)
            arcname_str = str(arcname)
            if arcname_str in seen_arc_names:
                continue
            zf.write(resolved_path, arcname_str)
            seen_arc_names.add(arcname_str)

    return archive


def write_excavation_shapefile(
    df: pd.DataFrame,
    *,
    out_base: str | Path,
    geometry: str,
    x_col: str = "x",
    y_col: str = "y",
    value_col: str = "excavation_m",
    tile_col: str = "tile_id",
    year_value: int | None = None,
    cell_width: float | None = None,
    cell_height: float | None = None,
    prj_wkt: str | None = None,
) -> ShapefileExportResult:
    normalized_geometry = geometry.lower()
    if normalized_geometry not in {"polygon", "point"}:
        raise ValueError("geometry must be 'polygon' or 'point'")
    if normalized_geometry == "polygon" and (cell_width is None or cell_height is None):
        raise ValueError("cell_width and cell_height are required for polygon exports")

    export_df = df.copy()
    export_df[x_col] = pd.to_numeric(export_df[x_col], errors="coerce")
    export_df[y_col] = pd.to_numeric(export_df[y_col], errors="coerce")
    export_df[value_col] = pd.to_numeric(export_df[value_col], errors="coerce")
    export_df = export_df.dropna(subset=[x_col, y_col, value_col]).copy()
    export_df = export_df.loc[export_df[value_col] > 0].copy()
    if export_df.empty:
        raise ValueError("No positive excavation cells were available for export")

    if tile_col and tile_col not in export_df.columns:
        export_df[tile_col] = ""
    if year_value is None:
        export_df["year"] = None
    else:
        export_df["year"] = int(year_value)

    out_base_path = Path(out_base)
    out_base_path.parent.mkdir(parents=True, exist_ok=True)
    shp_path = out_base_path.with_suffix(".shp")
    shx_path = out_base_path.with_suffix(".shx")
    dbf_path = out_base_path.with_suffix(".dbf")
    prj_path = out_base_path.with_suffix(".prj") if prj_wkt else None
    cpg_path = out_base_path.with_suffix(".cpg")

    fields = [
        ("excav_m", "N", 18, 6),
        ("x_center", "N", 18, 6),
        ("y_center", "N", 18, 6),
    ]
    if year_value is not None:
        fields.append(("year", "N", 6, 0))
    if tile_col:
        fields.append(("tile_id", "C", 40, 0))

    shape_type = POLYGON_SHAPE_TYPE if normalized_geometry == "polygon" else POINT_SHAPE_TYPE
    record_count = len(export_df)
    bbox = [math.inf, math.inf, -math.inf, -math.inf]
    offset_words = 50

    with (
        shp_path.open("wb") as shp_handle,
        shx_path.open("wb") as shx_handle,
        dbf_path.open("wb") as dbf_handle,
    ):
        shp_handle.write(b"\x00" * 100)
        shx_handle.write(b"\x00" * 100)
        _write_dbf_header(dbf_handle, record_count, fields)

        for record_number, row in enumerate(export_df.itertuples(index=False), start=1):
            x_value = float(getattr(row, x_col))
            y_value = float(getattr(row, y_col))
            exc_value = float(getattr(row, value_col))
            tile_value = getattr(row, tile_col) if tile_col else None

            if normalized_geometry == "polygon":
                half_w = float(cell_width) / 2.0
                half_h = float(cell_height) / 2.0
                ring = _build_cell_ring(x_value, y_value, half_w, half_h)
                record_bbox = (
                    x_value - half_w,
                    y_value - half_h,
                    x_value + half_w,
                    y_value + half_h,
                )
                content = _build_polygon_content(ring, record_bbox)
            else:
                record_bbox = (x_value, y_value, x_value, y_value)
                content = _build_point_content(x_value, y_value)

            bbox[0] = min(bbox[0], record_bbox[0])
            bbox[1] = min(bbox[1], record_bbox[1])
            bbox[2] = max(bbox[2], record_bbox[2])
            bbox[3] = max(bbox[3], record_bbox[3])

            content_length_words = len(content) // 2
            shp_handle.write(struct.pack(">2i", record_number, content_length_words))
            shp_handle.write(content)
            shx_handle.write(struct.pack(">2i", offset_words, content_length_words))
            offset_words += 4 + content_length_words

            dbf_record = {
                "excav_m": exc_value,
                "x_center": x_value,
                "y_center": y_value,
            }
            if year_value is not None:
                dbf_record["year"] = year_value
            if tile_col:
                dbf_record["tile_id"] = "" if tile_value is None else str(tile_value)
            _write_dbf_record(dbf_handle, dbf_record, fields)

        shp_length_words = offset_words
        shx_length_words = 50 + (record_count * 4)
        _write_shapefile_header(shp_handle, shp_length_words, shape_type, bbox)
        _write_shapefile_header(shx_handle, shx_length_words, shape_type, bbox)

    with cpg_path.open("w", encoding="ascii") as cpg_handle:
        cpg_handle.write("UTF-8")
    if prj_path and prj_wkt:
        with prj_path.open("w", encoding="utf-8") as prj_handle:
            prj_handle.write(prj_wkt)

    return ShapefileExportResult(
        geometry=normalized_geometry,
        record_count=record_count,
        shp_path=shp_path,
        shx_path=shx_path,
        dbf_path=dbf_path,
        prj_path=prj_path,
        cpg_path=cpg_path,
    )


def infer_axis_step(values: Iterable[Any]) -> float | None:
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    unique_values = sorted(set(float(value) for value in numeric.tolist()))
    if len(unique_values) < 2:
        return None
    steps = [
        curr - prev
        for prev, curr in zip(unique_values[:-1], unique_values[1:])
        if curr > prev and math.isfinite(curr - prev)
    ]
    if not steps:
        return None
    return min(steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export cumulative excavation cells to ArcGIS-friendly shapefiles."
    )
    parser.add_argument("--counter-csv", required=True, help="Path to the site_counter_dem CSV.")
    parser.add_argument(
        "--site-summary-csv",
        help="Optional path to the *_site_dem_summary.csv used to populate the .prj and cell size.",
    )
    parser.add_argument(
        "--out-dir",
        help="Optional output directory. Defaults to the counter CSV directory.",
    )
    parser.add_argument(
        "--out-stem",
        help="Optional base filename stem. Defaults to '<counter name>_excavation_heatmap'.",
    )
    parser.add_argument(
        "--geometry",
        choices=("polygon", "point", "both"),
        default="both",
        help="Which shapefile geometry to export.",
    )
    parser.add_argument("--cell-width", type=float, help="Override the polygon cell width.")
    parser.add_argument("--cell-height", type=float, help="Override the polygon cell height.")
    parser.add_argument(
        "--include-zero",
        action="store_true",
        help="Include zero-value cells instead of exporting only excavated cells.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = export_excavation_heatmap_shapefiles(
        args.counter_csv,
        site_summary_csv=args.site_summary_csv,
        out_dir=args.out_dir,
        out_stem=args.out_stem,
        geometry=args.geometry,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
        positive_only=not args.include_zero,
    )

    for result in results:
        print(
            f"{result.geometry}: {result.record_count} features -> {result.shp_path}"
        )
        if result.prj_path:
            print(f"  prj: {result.prj_path}")


def _default_output_stem(counter_stem: str) -> str:
    stem = counter_stem
    if stem.endswith("_counter_dem"):
        stem = stem[: -len("_counter_dem")]
    return f"{stem}_excavation_heatmap"


def _parse_year_from_name(name: str) -> int | None:
    match = re.search(r"year(\d+)", name, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _parse_float(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    value = raw_value.strip()
    if not value or value.lower() == "none":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _extract_epsg_code(raw_value: str | None) -> int | None:
    if raw_value is None:
        return None
    value = str(raw_value)
    match = re.search(r"EPSG\s*[:\"]\s*(\d+)", value, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    projected_wkt = _extract_projected_wkt(value)
    if projected_wkt:
        matches = re.findall(r'AUTHORITY\["EPSG","(\d+)"\]', projected_wkt, flags=re.IGNORECASE)
        if matches:
            return int(matches[-1])
    matches = re.findall(r'AUTHORITY\["EPSG","(\d+)"\]', value, flags=re.IGNORECASE)
    if matches:
        return int(matches[-1])
    return None


def _extract_projected_wkt(raw_value: str) -> str | None:
    for token in ("PROJCRS[", "PROJCS["):
        extracted = _extract_balanced_wkt(raw_value, token)
        if extracted:
            return extracted
    return None


def _wkt_contains_epsg(projected_wkt: str, epsg_code: int) -> bool:
    return re.search(rf'AUTHORITY\["EPSG","{int(epsg_code)}"\]', projected_wkt, flags=re.IGNORECASE) is not None


def _projected_wkt_from_epsg(epsg_code: int) -> str | None:
    if PyprojCRS is None:
        return None
    try:
        crs = PyprojCRS.from_epsg(int(epsg_code))
        if not crs.is_projected:
            return None
        return crs.to_wkt(version="WKT1_ESRI")
    except Exception:
        return None


def _looks_like_lon_lat(values: pd.DataFrame, x_col: str, y_col: str) -> bool:
    x_min = float(values[x_col].min())
    x_max = float(values[x_col].max())
    y_min = float(values[y_col].min())
    y_max = float(values[y_col].max())
    return abs(x_min) <= 180 and abs(x_max) <= 180 and abs(y_min) <= 90 and abs(y_max) <= 90


def validate_coordinate_crs_match(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    crs: CrsResolution,
    summary_path: Path | None,
) -> None:
    if df.empty or crs.raw_value is None:
        return

    raw_upper = crs.raw_value.upper()
    has_projected_wkt = crs.projected_wkt is not None
    looks_lon_lat = _looks_like_lon_lat(df, x_col, y_col)
    location = str(summary_path) if summary_path is not None else "site summary"

    if has_projected_wkt and looks_lon_lat:
        raise ValueError(
            f"Coordinate sanity check failed for {location}: the shapefile export resolved a projected DEM CRS "
            f"from '{crs.source_key}', but the x/y values still look like longitude/latitude."
        )

    if ("GEOGCS[" in raw_upper or "GEOGCRS[" in raw_upper) and not has_projected_wkt and not looks_lon_lat:
        raise ValueError(
            f"Coordinate sanity check failed for {location}: DEM CRS looks geographic, but x/y values look projected."
        )


def _collect_shapefile_component_paths(results: Iterable[ShapefileExportResult]) -> list[Path]:
    component_paths: list[Path] = []
    for result in results:
        for suffix in SHAPEFILE_COMPONENT_SUFFIXES:
            path = result.shp_path.with_suffix(suffix)
            if path.exists() and path.is_file():
                component_paths.append(path)
    return component_paths


def _common_parent(paths: Iterable[Path]) -> Path:
    resolved_parts = [str(path.resolve()) for path in paths]
    common_path = Path(os.path.commonpath(resolved_parts))
    return common_path


def _extract_balanced_wkt(raw_value: str, token: str) -> str | None:
    start = raw_value.find(token)
    if start < 0:
        return None

    depth = 0
    for index in range(start, len(raw_value)):
        char = raw_value[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return raw_value[start : index + 1]
    return None


def _build_cell_ring(x_value: float, y_value: float, half_w: float, half_h: float) -> list[tuple[float, float]]:
    return [
        (x_value - half_w, y_value - half_h),
        (x_value - half_w, y_value + half_h),
        (x_value + half_w, y_value + half_h),
        (x_value + half_w, y_value - half_h),
        (x_value - half_w, y_value - half_h),
    ]


def _build_point_content(x_value: float, y_value: float) -> bytes:
    return struct.pack("<idd", POINT_SHAPE_TYPE, x_value, y_value)


def _build_polygon_content(
    ring: list[tuple[float, float]],
    bbox: tuple[float, float, float, float],
) -> bytes:
    content = bytearray()
    content.extend(struct.pack("<i", POLYGON_SHAPE_TYPE))
    content.extend(struct.pack("<4d", *bbox))
    content.extend(struct.pack("<2i", 1, len(ring)))
    content.extend(struct.pack("<i", 0))
    for x_value, y_value in ring:
        content.extend(struct.pack("<2d", x_value, y_value))
    return bytes(content)


def _write_shapefile_header(
    handle,
    file_length_words: int,
    shape_type: int,
    bbox: list[float],
) -> None:
    handle.seek(0)
    handle.write(struct.pack(">i", 9994))
    handle.write(struct.pack(">5i", 0, 0, 0, 0, 0))
    handle.write(struct.pack(">i", file_length_words))
    handle.write(struct.pack("<i", 1000))
    handle.write(struct.pack("<i", shape_type))
    handle.write(struct.pack("<4d", *bbox))
    handle.write(struct.pack("<4d", 0.0, 0.0, 0.0, 0.0))


def _write_dbf_header(handle, record_count: int, fields: list[tuple[str, str, int, int]]) -> None:
    today = date.today()
    header_length = 32 + (32 * len(fields)) + 1
    record_length = 1 + sum(field_length for _, _, field_length, _ in fields)
    handle.write(
        struct.pack(
            "<BBBBLHH20x",
            0x03,
            today.year - 1900,
            today.month,
            today.day,
            record_count,
            header_length,
            record_length,
        )
    )

    for name, field_type, field_length, decimal_count in fields:
        encoded_name = name.encode("ascii", errors="ignore")[:10]
        descriptor = encoded_name + (b"\x00" * (11 - len(encoded_name)))
        descriptor += field_type.encode("ascii")
        descriptor += struct.pack("<LBB14x", 0, field_length, decimal_count)
        handle.write(descriptor)

    handle.write(b"\r")


def _write_dbf_record(
    handle,
    record: dict[str, Any],
    fields: list[tuple[str, str, int, int]],
) -> None:
    handle.write(b" ")
    for name, field_type, field_length, decimal_count in fields:
        value = record.get(name)
        handle.write(_format_dbf_value(value, field_type, field_length, decimal_count))


def _format_dbf_value(value: Any, field_type: str, field_length: int, decimal_count: int) -> bytes:
    if field_type == "C":
        if value is None:
            text = ""
        else:
            text = str(value)
        encoded = text.encode("utf-8", errors="ignore")[:field_length]
        return encoded.ljust(field_length, b" ")

    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return b" " * field_length

    numeric = float(value)
    if decimal_count == 0:
        text = f"{int(round(numeric))}"
    else:
        text = f"{numeric:.{decimal_count}f}"

    if len(text) > field_length:
        text = f"{numeric:.{max(decimal_count - 1, 0)}f}"
    if len(text) > field_length:
        raise ValueError(f"DBF field overflow for value {value!r} with width {field_length}")
    return text.rjust(field_length, " ").encode("ascii")


if __name__ == "__main__":
    main()
