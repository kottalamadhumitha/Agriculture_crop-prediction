"""
Phase 2 yield-series loader.

Your project currently supports two possible “APY-like” formats:

1) CSV (long format)
   - You place an official district/crop/year table as CSV.
   - Loader detects columns like Year/State/District/Crop/Yield.

2) HTML-table exports saved with .xls extension (DES/APY-style)
   - Your `Rabi_prod.xls` and `Kharif_prod.xls` are such exports.
   - They are “wide” tables: each row is (State, District, Year), and each crop
     has multiple columns including "Yield (...)".
   - Loader parses them with `pandas.read_html` and extracts the relevant yield column.

No invented values: all chart points come only from your local files.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "apy" / "district_apy.csv"
DEFAULT_RABI_XLS = ROOT / "data" / "apy" / "Rabi_prod.xls"
DEFAULT_KHARIF_XLS = ROOT / "data" / "apy" / "Kharif_prod.xls"
DEFAULT_RABI_CSV = ROOT / "data" / "apy" / "Rabi_prod.csv"
DEFAULT_KHARIF_CSV = ROOT / "data" / "apy" / "Kharif_prod.csv"


def _norm_key(s: str) -> str:
    s = str(s or "").strip().lower()
    # remove leading numeric prefixes like "1. Telangana"
    s = re.sub(r"^\s*\d+\.\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


# -------- CSV (long format) loader --------

# Map model class label -> substrings to match in official "Crop" column (lowercase).
CROP_LOOKUP_TOKENS: dict[str, list[str]] = {
    "rice": ["rice", "paddy"],
    "cotton": ["cotton"],
    "maize": ["maize"],
    "mungbean": ["mung", "green gram"],
    "blackgram": ["black gram", "blackgram", "urd bean", "urd"],
    "pigeonpeas": ["pigeon pea", "pigeonpea", "red gram", "arhar", "tur dal"],
}


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    mapping = {_norm_key(c): c for c in df.columns}
    for cand in candidates:
        nc = _norm_key(cand)
        if nc in mapping:
            return mapping[nc]
    for col in df.columns:
        n = _norm_key(str(col))
        for cand in candidates:
            if _norm_key(cand) in n:
                return col
    return None


def _resolve_columns(df: pd.DataFrame) -> dict[str, str] | None:
    year = _find_column(df, ("year", "crop_year", "yr"))
    state = _find_column(df, ("state name", "state_name", "state"))
    district = _find_column(df, ("district name", "district_name", "district"))
    crop = _find_column(df, ("crop name", "crop_name", "crop"))
    yield_col = _find_column(df, ("yield", "productivity", "yield (kg/ha)", "yield(kg/ha)"))
    prod = _find_column(df, ("production", "prod", "production (tonnes)"))
    area = _find_column(df, ("area", "area (in hectare)", "area (hectare)"))

    if not all([year, state, district, crop]):
        return None
    return {
        "year": year,
        "state": state,
        "district": district,
        "crop": crop,
        "yield": yield_col or "",
        "production": prod or "",
        "area": area or "",
    }


class APYRepository:
    def __init__(self, csv_path: Path | None) -> None:
        self.csv_path = csv_path
        self._df: pd.DataFrame | None = None
        self._cols: dict[str, str] | None = None
        self.load_error: str | None = None

        if csv_path is None or not csv_path.is_file():
            self.load_error = (
                f"No APY CSV at {csv_path or DEFAULT_CSV}. "
                "If you downloaded HTML-table exports, ensure Rabi_prod.xls and Kharif_prod.xls exist."
            )
            return

        try:
            raw = pd.read_csv(csv_path, low_memory=False)
            raw.columns = [str(c).strip() for c in raw.columns]
            cols = _resolve_columns(raw)
            if not cols:
                self.load_error = (
                    "Could not detect Year / State / District / Crop columns in APY CSV. "
                    "Use headers similar to the Area–Production–Yield open-data exports."
                )
            else:
                self._df = raw
                self._cols = cols
        except Exception as e:
            self.load_error = f"Failed to read APY CSV: {e}"

    @property
    def is_loaded(self) -> bool:
        return self._df is not None and self._cols is not None

    def series_for(
        self, state_hint: str, district_hint: str, model_crop: str
    ) -> tuple[list[dict], str | None]:
        if not self.is_loaded or self._df is None or self._cols is None:
            return [], self.load_error

        df = self._df
        c = self._cols
        tokens = CROP_LOOKUP_TOKENS.get(model_crop.lower(), [model_crop.lower()])

        st = _norm_key(state_hint)
        dist = _norm_key(district_hint)

        def score_row(row: pd.Series) -> int:
            rs = _norm_key(str(row[c["state"]]))
            rd = _norm_key(str(row[c["district"]]))
            sc = 0
            if rs == st:
                sc += 2
            elif st in rs or rs in st:
                sc += 1
            if rd == dist:
                sc += 2
            elif dist in rd or rd in dist:
                sc += 1
            return sc

        mask_crop = df[c["crop"]].astype(str).str.lower().apply(
            lambda s: any(tok in s.lower() for tok in tokens)
        )
        sub = df.loc[mask_crop].copy()
        if sub.empty:
            return [], (
                f"No rows in APY CSV for crop tokens {tokens!r} (model label: {model_crop})."
            )

        scores = sub.apply(score_row, axis=1)
        max_sc = int(scores.max())
        if max_sc < 2:
            return [], (
                "Could not reliably match geocoded state/district to this APY CSV. "
                "Check spelling in your CSV versus Nominatim output."
            )
        sub = sub.loc[scores == max_sc].copy()

        sub["_year"] = pd.to_numeric(sub[c["year"]], errors="coerce")
        sub = sub.dropna(subset=["_year"])

        if c["yield"]:
            sub["_yield"] = pd.to_numeric(sub[c["yield"]], errors="coerce")
        elif c["production"] and c["area"]:
            prod = pd.to_numeric(sub[c["production"]], errors="coerce")
            area = pd.to_numeric(sub[c["area"]], errors="coerce")
            sub["_yield"] = prod / area.replace(0, float("nan"))
        else:
            sub["_yield"] = float("nan")

        out: list[dict[str, Any]] = []
        for yr, g in sub.groupby("_year"):
            y_mean = pd.to_numeric(g["_yield"], errors="coerce").mean()
            if pd.isna(y_mean):
                continue
            out.append({"year": int(float(yr)), "yield": round(float(y_mean), 2)})
        out.sort(key=lambda x: x["year"])
        note = (
            "Values come only from your local APY CSV (see data/apy/README.txt). "
            "Units are as in the source file."
        )
        return out, note


# -------- Wide HTML-table export loader --------


MODEL_CROP_TO_WIDE_CROP: dict[str, str] = {
    "rice": "Rice",
    "cotton": "Cotton(lint)",
    "maize": "Maize",
    "mungbean": "Moong(Green Gram)",
    "blackgram": "Urad",
    "pigeonpeas": "Arhar/Tur",
}


class WideAPYRepository:
    def __init__(self, rabi_xls: Path, kharif_xls: Path) -> None:
        self.rabi_xls = rabi_xls
        self.kharif_xls = kharif_xls
        self._rabi_df: pd.DataFrame | None = None
        self._kharif_df: pd.DataFrame | None = None
        self.load_error: str | None = None
        self._units_by_crop: dict[str, str] = {}

        if not rabi_xls.is_file() and not kharif_xls.is_file():
            self.load_error = (
                "No APY wide-table files found. Expected data/apy/Rabi_prod.xls and/or "
                "data/apy/Kharif_prod.xls."
            )
            return

        try:
            if rabi_xls.is_file():
                self._rabi_df = pd.read_html(rabi_xls)[0]
            if kharif_xls.is_file():
                self._kharif_df = pd.read_html(kharif_xls)[0]
        except Exception as e:
            msg = str(e)
            if "lxml" in msg.lower() or "import lxml" in msg.lower():
                self.load_error = (
                    "Failed to parse APY wide exports (Rabi_prod.xls / Kharif_prod.xls) because `lxml` "
                    "is not available in this Python environment. "
                    "Install dependencies with `pip install -r requirements.txt` and restart the server."
                )
            else:
                self.load_error = f"Failed to parse wide APY exports (read_html): {e}"

    @property
    def is_loaded(self) -> bool:
        return self.load_error is None and (self._rabi_df is not None or self._kharif_df is not None)

    def _extract_columns(self, df: pd.DataFrame) -> dict[str, Any]:
        # df.columns is MultiIndex with 3 levels: (crop, season, metric) for crop cols
        # and (State, State, State), etc for meta cols.
        def find_level0(label: str) -> Any | None:
            for c in df.columns:
                if isinstance(c, tuple) and str(c[0]).strip().lower() == label:
                    return c
            return None

        state_col = find_level0("state")
        district_col = find_level0("district")
        year_col = find_level0("year")
        if state_col is None or district_col is None or year_col is None:
            raise ValueError("Could not find State/District/Year columns in APY wide export.")
        return {"state_col": state_col, "district_col": district_col, "year_col": year_col}

    def _yield_col_for_crop(self, df: pd.DataFrame, crop_header: str) -> tuple[Any, str] | None:
        # pick the column where level0 == crop_header and last level contains 'Yield'
        for c in df.columns:
            if not isinstance(c, tuple):
                continue
            if str(c[0]).strip() != crop_header:
                continue
            last = str(c[-1])
            if "Yield" in last:
                return c, last
        return None

    def _match_state_district(
        self, df: pd.DataFrame, state_hint: str, district_hint: str, cols: dict[str, Any]
    ) -> pd.DataFrame:
        st = _norm_key(state_hint)
        dist = _norm_key(district_hint)

        state_series = df[cols["state_col"]].astype(str).apply(_norm_key)
        district_series = df[cols["district_col"]].astype(str).apply(_norm_key)

        # substring match (robust to formatting differences like "Adilabad" vs "Adilabad district")
        mask = (state_series == st) & (district_series == dist)
        if mask.any():
            return df.loc[mask].copy()

        mask2 = (state_series.str.contains(re.escape(st), na=False)) & (
            district_series.str.contains(re.escape(dist), na=False)
        )
        return df.loc[mask2].copy() if mask2.any() else df.iloc[0:0].copy()

    def series_for(
        self, state_hint: str, district_hint: str, model_crop: str
    ) -> tuple[list[dict], str | None]:
        if not self.is_loaded:
            return [], self.load_error

        crop_header = MODEL_CROP_TO_WIDE_CROP.get(model_crop.lower(), None)
        if not crop_header:
            return [], f"Unsupported crop label for wide APY exports: {model_crop}"

        series_by_year: dict[int, list[float]] = defaultdict(list)
        unit_note: str | None = None

        for season_name, df in [("Rabi", self._rabi_df), ("Kharif", self._kharif_df)]:
            if df is None:
                continue

            cols = self._extract_columns(df)
            matched = self._match_state_district(df, state_hint, district_hint, cols)
            if matched.empty:
                continue

            yinfo = self._yield_col_for_crop(df, crop_header)
            if not yinfo:
                continue
            ycol, unit = yinfo
            unit_note = unit_note or unit

            def parse_year_range(val: Any) -> int | None:
                # Example values: "2018 - 2019"
                s = str(val or "")
                years = re.findall(r"\d{4}", s)
                if not years:
                    return None
                return int(years[-1])  # use end-year for the chart

            years = matched[cols["year_col"]].apply(parse_year_range)
            yields = pd.to_numeric(matched[ycol], errors="coerce")

            tmp = pd.DataFrame({"year": years, "yield": yields}).dropna()
            for _, r in tmp.iterrows():
                y = int(r["year"])
                v = float(r["yield"])
                if pd.isna(v):
                    continue
                series_by_year[y].append(v)

        if not series_by_year:
            return [], (
                "No yield values found for the matched district/crop in your wide APY exports. "
                "Try a different district name (geocoder spelling matters) or confirm the crop exists."
            )

        out: list[dict[str, Any]] = []
        for year in sorted(series_by_year.keys()):
            vals = series_by_year[year]
            if not vals:
                continue
            out.append({"year": year, "yield": round(sum(vals) / len(vals), 2)})

        unit_txt = f"Unit from source: {unit_note}. " if unit_note else ""
        note = (
            "Values are extracted from your local wide APY exports (Rabi/Kharif_prod.xls). "
            "These are official district-level tables; units come from the selected yield column. "
            + unit_txt
        )
        return out, note

    def districts_for_state(self, state_hint: str, limit: int = 25) -> list[str]:
        """
        Return up to `limit` unique district strings that exist in the wide APY
        exports for the provided state. This is extracted directly from the
        local files (no guessing/mapping).
        """
        if not self.is_loaded:
            return []

        st = _norm_key(state_hint)

        districts: set[str] = set()
        state_col = ("State", "State", "State")
        district_col = ("District", "District", "District")

        def collect_from(df: pd.DataFrame | None) -> None:
            if df is None:
                return
            state_series = df[state_col].astype(str).apply(_norm_key)
            dist_series = df[district_col].astype(str)
            matched = df.loc[state_series == st, district_col]
            for v in matched.astype(str).tolist():
                districts.add(v)

        collect_from(self._rabi_df)
        collect_from(self._kharif_df)

        # Stable sort by normalized key
        def sort_key(x: str) -> str:
            return _norm_key(x)

        return sorted(districts, key=sort_key)[:limit]


def get_apy_repository() -> Any:
    """
    Choose loader based on what files exist locally.
    Priority:
      1) APY_CSV_PATH (if set and exists)
      2) Rabi_prod.xls / Kharif_prod.xls wide exports
      3) default CSV error
    """
    csv_path = os.environ.get("APY_CSV_PATH", "").strip()
    if csv_path:
        p = Path(csv_path)
        return APYRepository(p)

    # If no CSV is provided, prefer the wide exports your project already contains.
    # Your project may have converted DES/APY wide exports from .xls to .csv.
    if DEFAULT_RABI_CSV.is_file() or DEFAULT_KHARIF_CSV.is_file():
        return WideCSVRepository(DEFAULT_RABI_CSV, DEFAULT_KHARIF_CSV)

    if DEFAULT_RABI_XLS.is_file() or DEFAULT_KHARIF_XLS.is_file():
        return WideAPYRepository(DEFAULT_RABI_XLS, DEFAULT_KHARIF_XLS)

    return APYRepository(DEFAULT_CSV)


class WideCSVRepository:
    """
    Loader for DES/APY wide exports saved as CSV.

    The file header is multi-row flattened:
      Row 0: meta columns (State, District, Year) then crop names with blanks
      Row 1: season names (e.g., Rabi/Kharif) repeated for each crop block
      Row 2: metric names in each 3-column block (Area, Production, Yield ...)
      Rows >= 3: actual data, with columns grouped as (Area, Production, Yield)
    """

    def __init__(self, rabi_csv: Path, kharif_csv: Path) -> None:
        self.rabi_csv = rabi_csv
        self.kharif_csv = kharif_csv
        self._rabi_df: pd.DataFrame | None = None
        self._kharif_df: pd.DataFrame | None = None
        self.load_error: str | None = None
        self._yield_col_by_crop: dict[str, int] = {}
        self._yield_col_note_by_crop: dict[str, str] = {}

        if not rabi_csv.is_file() and not kharif_csv.is_file():
            self.load_error = (
                "No APY wide-table files found. Expected data/apy/Rabi_prod.csv and/or "
                "data/apy/Kharif_prod.csv."
            )
            return

        try:
            if rabi_csv.is_file():
                self._rabi_raw, self._rabi_meta, self._rabi_data = self._parse_wide_csv(rabi_csv)
                self._yield_col_by_crop, self._yield_col_note_by_crop = self._build_yield_map_from_meta(
                    self._rabi_meta
                )
            if kharif_csv.is_file():
                self._kharif_raw, self._kharif_meta, self._kharif_data = self._parse_wide_csv(kharif_csv)
                # If rabi is missing a crop block, enrich yield map from kharif
                ymap, notes = self._build_yield_map_from_meta(self._kharif_meta)
                for k, v in ymap.items():
                    if k not in self._yield_col_by_crop:
                        self._yield_col_by_crop[k] = v
                for k, v in notes.items():
                    if k not in self._yield_col_note_by_crop:
                        self._yield_col_note_by_crop[k] = v
        except Exception as e:
            self.load_error = f"Failed to parse wide APY CSV exports: {e}"

    def _parse_wide_csv(self, p: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        raw = pd.read_csv(p, header=None, dtype=str, keep_default_na=False)
        if raw.shape[0] < 4 or raw.shape[1] < 6:
            raise ValueError(f"Unexpected CSV shape for {p.name}: {raw.shape}")
        meta0 = raw.iloc[0].copy()
        # meta1 = raw.iloc[1]  # season, not needed for yield extraction
        meta2 = raw.iloc[2].copy()
        data = raw.iloc[3:].copy()
        return raw, (meta0, meta2), data

    def _build_yield_map_from_meta(
        self, meta: tuple[pd.Series, pd.Series]
    ) -> tuple[dict[str, int], dict[str, str]]:
        meta0, meta2 = meta
        ncols = int(meta0.shape[0])
        # Each crop block is 3 columns wide: (Area, Production, Yield)
        # State/District/Year occupy indices 0,1,2.
        yield_col_by_crop: dict[str, int] = {}
        notes_by_crop: dict[str, str] = {}

        for start in range(3, ncols, 3):
            crop_name = str(meta0.iloc[start]).strip()
            if not crop_name or crop_name.lower() == "nan":
                continue
            # In the original table ordering: Area, Production, Yield
            candidate_yield = start + 2
            metric_label = str(meta2.iloc[candidate_yield]).strip()
            if "Yield" not in metric_label:
                # fallback: search within the 3-column block
                yield_col = None
                for j in (start, start + 1, start + 2):
                    if "Yield" in str(meta2.iloc[j]):
                        yield_col = j
                        metric_label = str(meta2.iloc[j]).strip()
                        break
                if yield_col is None:
                    continue
            else:
                yield_col = candidate_yield

            yield_col_by_crop[crop_name] = yield_col
            notes_by_crop[crop_name] = metric_label

        return yield_col_by_crop, notes_by_crop

    @property
    def is_loaded(self) -> bool:
        return self.load_error is None and (
            getattr(self, "_rabi_data", None) is not None or getattr(self, "_kharif_data", None) is not None
        )

    def _match_rows(self, data: pd.DataFrame, state_hint: str, district_hint: str) -> pd.DataFrame:
        # In the exported wide tables, State/District are typically repeated only
        # for the first year row; later year rows may contain blanks.
        st = _norm_key(state_hint)
        dist = _norm_key(district_hint)

        state_raw = data.iloc[:, 0].replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
        district_raw = data.iloc[:, 1].replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})

        state_filled = state_raw.ffill()
        district_filled = district_raw.ffill()

        state_norm = state_filled.astype(str).apply(_norm_key)
        district_norm = district_filled.astype(str).apply(_norm_key)

        mask = (state_norm == st) & (district_norm == dist)
        return data.loc[mask].copy()

    def _parse_year_end(self, v: Any) -> int | None:
        s = str(v or "")
        years = re.findall(r"\d{4}", s)
        if not years:
            return None
        return int(years[-1])

    def _series_from_data(self, data: pd.DataFrame, state_hint: str, district_hint: str, crop_header: str) -> list[dict]:
        if data is None:
            return []
        matched = self._match_rows(data, state_hint, district_hint)
        if matched.empty:
            return []

        # Find yield column index for this crop header
        if crop_header not in self._yield_col_by_crop:
            return []
        ycol = self._yield_col_by_crop[crop_header]

        years = matched.iloc[:, 2].apply(self._parse_year_end)
        raw_yields = matched.iloc[:, ycol].astype(str).str.replace(",", "", regex=False)
        yields = pd.to_numeric(raw_yields, errors="coerce")

        out_by_year: dict[int, list[float]] = defaultdict(list)
        for yr, val in zip(years.tolist(), yields.tolist()):
            if yr is None:
                continue
            if pd.isna(val):
                continue
            out_by_year[yr].append(float(val))

        out: list[dict] = []
        for yr in sorted(out_by_year.keys()):
            vals = out_by_year[yr]
            if not vals:
                continue
            out.append({"year": yr, "yield": round(sum(vals) / len(vals), 2)})
        return out

    def series_for(
        self, state_hint: str, district_hint: str, model_crop: str
    ) -> tuple[list[dict], str | None]:
        if not self.is_loaded:
            return [], self.load_error

        crop_header = MODEL_CROP_TO_WIDE_CSV.get(model_crop.lower())
        if not crop_header:
            return [], f"Unsupported crop label for wide APY CSV exports: {model_crop}"

        series_points: dict[int, list[float]] = defaultdict(list)
        unit_note: str | None = None

        # Rabi and Kharif are separate files; merge by end-year.
        for data in [getattr(self, "_rabi_data", None), getattr(self, "_kharif_data", None)]:
            if data is None:
                continue
            pts = self._series_from_data(data, state_hint, district_hint, crop_header)
            for p in pts:
                series_points[p["year"]].append(p["yield"])
            unit_note = unit_note or self._yield_col_note_by_crop.get(crop_header)

        if not series_points:
            return [], (
                "No yield values found for the matched district/crop in your wide APY CSV exports. "
                "Try a different district name (spelling must match the APY file rows)."
            )

        out: list[dict] = []
        for year in sorted(series_points.keys()):
            vals = series_points[year]
            out.append({"year": year, "yield": round(sum(vals) / len(vals), 2)})

        note = (
            "Values are extracted from your local wide APY CSV exports (Rabi_prod.csv / Kharif_prod.csv). "
            "These are official district-level tables; units come from the selected yield column. "
            + (f"Unit from source: {unit_note}. " if unit_note else "")
        )
        return out, note

    def districts_for_state(self, state_hint: str, limit: int = 25) -> list[str]:
        if not self.is_loaded:
            return []
        st = _norm_key(state_hint)
        districts: set[str] = set()

        for data in [getattr(self, "_rabi_data", None), getattr(self, "_kharif_data", None)]:
            if data is None:
                continue
            state_norm = data.iloc[:, 0].astype(str).apply(_norm_key)
            matched = data.loc[state_norm == st, :]
            for v in matched.iloc[:, 1].astype(str).tolist():
                districts.add(v)

        return sorted(districts, key=_norm_key)[:limit]


# Model label -> crop header used inside Rabi_prod.csv / Kharif_prod.csv wide tables
MODEL_CROP_TO_WIDE_CSV: dict[str, str] = {
    "rice": "Rice",
    "cotton": "Cotton(lint)",
    "maize": "Maize",
    "mungbean": "Moong(Green Gram)",
    "blackgram": "Urad",
    "pigeonpeas": "Arhar/Tur",
}
