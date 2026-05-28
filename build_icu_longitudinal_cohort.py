from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Cấu hình cho việc xây dựng cohort ICU longitudinal
INPUT_DIR = Path("mimic-iv-3.1")
OUTPUT_FILE = Path("cohort_icu_longitudinal_90d.csv")
OUTPUT_DIR = Path("outputs")
hours = 24
HORIZON_DAYS = 90
CHUNKSIZE = 1_000_000

# Các itemid MIMIC-IV của vital signs ICU.
VITAL_ITEMIDS = {
    "heart_rate": [220045],
    "sbp": [220179, 220050],
    "dbp": [220180, 220051],
    "mbp": [220181, 220052],
    "resp_rate": [220210],
    "temperature": [223761, 223762],
    "spo2": [220277],
    "glucose": [225664, 220621, 226537],
}

# Các itemid MIMIC-IV của labs.
LAB_ITEMIDS = {
    "creatinine": [50912],
    "bun": [51006],
    "wbc": [51300, 51301],
    "hemoglobin": [51222],
    "platelet": [51265],
    "sodium": [50983],
    "potassium": [50971],
    "chloride": [50902],
    "bicarbonate": [50882],
    "lactate": [50813],
    "ph": [50820, 50831],
    "po2": [50821],
    "pco2": [50818],
}


def log(message: str) -> None:
    # In log ngay để theo dõi tiến trình khi quét các bảng MIMIC lớn.
    print(message, flush=True)


def find_table(root: Path, relative: str) -> Path:
    # Tìm file CSV cho một bảng MIMIC.
    plain = root / relative
    gz = root / f"{relative}.gz"
    if plain.exists():
        return plain
    if gz.exists():
        return gz
    raise FileNotFoundError(f"Missing required MIMIC-IV table: {plain} or {gz}")


def read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    # Đọc một bảng CSV của MIMIC.
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def itemid_to_name(mapping: dict[str, list[int]]) -> dict[int, str]:
    # Chuyển mapping từ tên biến sang itemid thành mapping ngược để gán tên biến khi load chartevents/labevents.
    out = {}
    for name, itemids in mapping.items():
        for itemid in itemids:
            out[itemid] = name
    return out


def add_outcome(stays: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    # Thêm cột outcome tử vong trong horizon_days và thời gian đến event cho mỗi ICU stay.
    out = stays.copy()
    intime = pd.to_datetime(out["intime"], errors="coerce")
    death_time = pd.to_datetime(out["deathtime"], errors="coerce")
    dod = pd.to_datetime(out["dod"], errors="coerce")
    death_datetime = death_time.fillna(dod)
    days_to_death = (death_datetime - intime).dt.total_seconds() / 86400
    event = days_to_death.notna() & (days_to_death >= 0) & (days_to_death <= horizon_days)
    out["death_datetime"] = death_datetime
    out["days_to_death"] = days_to_death
    out[f"death_{horizon_days}d"] = event.astype(int)
    out[f"time_to_event_{horizon_days}d"] = np.where(
        event, days_to_death, float(horizon_days)
    )
    return out


def add_standard_outcomes(stays: pd.DataFrame) -> pd.DataFrame:
    # Thêm outcome tử vong 7 ngày, 28 ngày và 90 ngày cho bảng ICU stay.
    out = stays.copy()
    for horizon_days in [7, 28, 90]:
        out = add_outcome(out, horizon_days=horizon_days)
    return out


def prepare_icu_base(mimic_dir: Path, horizon_days: int) -> pd.DataFrame:
    # Tạo cohort ICU nền từ các bảng admissions, patients và icustays.
    hosp = mimic_dir / "hosp"
    icu = mimic_dir / "icu"

    admissions = read_csv(
        find_table(hosp, "admissions.csv"), usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "admission_type", "insurance", "race", "marital_status"])
    patients = read_csv(find_table(hosp, "patients.csv"), usecols=["subject_id", "gender", "anchor_age", "dod"])
    icustays = read_csv(
        find_table(icu, "icustays.csv"),
        usecols=["subject_id", "hadm_id", "stay_id", "first_careunit", "last_careunit", "intime", "outtime", "los"])
    
    # Chuyển đổi các cột thời gian sang datetime để tính toán.
    for col in ["admittime", "dischtime", "deathtime"]:
        admissions[col] = pd.to_datetime(admissions[col], errors="coerce")
    patients["dod"] = pd.to_datetime(patients["dod"], errors="coerce")
    
    for col in ["intime", "outtime"]:
        icustays[col] = pd.to_datetime(icustays[col], errors="coerce")

    # Ghép thông tin ICU stay với thông tin nhập viện và nhân khẩu học bệnh nhân.
    stays = (icustays.merge(admissions, on=["subject_id", "hadm_id"], how="left").merge(patients, on="subject_id", how="left").sort_values(["subject_id", "intime", "stay_id"]))
    
    # Giữ ICU stay đầu tiên của mỗi bệnh nhân để giảm phụ thuộc trong cùng bệnh nhân.
    stays = stays.drop_duplicates("subject_id", keep="first")
    stays["icu_los_hours"] = (stays["outtime"] - stays["intime"]).dt.total_seconds() / 3600
    stays["icu_los_days"] = stays["icu_los_hours"] / 24
    stays = add_standard_outcomes(stays)
    
    return stays


def summarize_longitudinal_rows(rows: pd.DataFrame, hours: int) -> pd.DataFrame:
    # Tóm tắt dữ liệu dọc đã load thành một dòng wide cho mỗi ICU stay.
    if rows.empty:
        return pd.DataFrame()
    rows = rows.dropna(subset=["valuenum", "charttime", "intime"]).copy()
    rows["hours_from_icu_intime"] = (rows["charttime"] - rows["intime"]).dt.total_seconds() / 3600
    rows = rows[(rows["hours_from_icu_intime"] >= 0) & (rows["hours_from_icu_intime"] <= hours)]
    if rows.empty:
        return pd.DataFrame()

    rows = rows.sort_values(["stay_id", "variable", "charttime"])
    grouped = rows.groupby(["stay_id", "variable"], observed=False)
    summary = grouped["valuenum"].agg(["min", "max", "mean", "median", "std", "count"])
    first = grouped.first()[["valuenum", "hours_from_icu_intime"]].rename(columns={"valuenum": "first", "hours_from_icu_intime": "first_hour"})
    last = grouped.last()[["valuenum", "hours_from_icu_intime"]].rename(columns={"valuenum": "last", "hours_from_icu_intime": "last_hour"})
    
    summary = summary.join(first).join(last).reset_index()
    elapsed_days = (summary["last_hour"] - summary["first_hour"]) / 24
    summary["slope"] = np.where(elapsed_days.abs() > 1e-9, (summary["last"] - summary["first"]) / elapsed_days, np.nan)
    
    value_cols = ["first", "last", "min", "max", "mean", "median", "std", "count", "slope"]
    wide_parts = []
    
    # Tạo các cột wide cho mỗi biến và mỗi thống kê.
    for stat in value_cols:
        wide = summary.pivot(index="stay_id", columns="variable", values=stat)
        wide.columns = [f"{col}_{stat}_{hours}h" for col in wide.columns]
        wide_parts.append(wide)
    return pd.concat(wide_parts, axis=1).reset_index()


def summarize_chunk(rows: pd.DataFrame) -> pd.DataFrame:
    # Tính thống kê theo stay/biến cho một chunk, có thể gộp với các chunk khác.
    if rows.empty:
        return pd.DataFrame()
    rows = rows.sort_values(["stay_id", "variable", "charttime"]).copy()
    rows["valuesq"] = rows["valuenum"] ** 2
    keys = ["stay_id", "variable"]
    grouped = rows.groupby(keys, observed=False)
    summary = grouped["valuenum"].agg(["min", "max", "sum", "count"])
    summary["sumsq"] = grouped["valuesq"].sum()
    first = rows.drop_duplicates(keys, keep="first")[keys + ["valuenum", "hours_from_icu_intime"]].rename(columns={"valuenum": "first", "hours_from_icu_intime": "first_hour"})
    last = rows.drop_duplicates(keys, keep="last")[keys + ["valuenum", "hours_from_icu_intime"]].rename(columns={"valuenum": "last", "hours_from_icu_intime": "last_hour"})
    
    return summary.join(first.set_index(keys)).join(last.set_index(keys))


def merge_group_summaries(current: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    # Gộp summary đã tích luỹ với summary mới từ chunk tiếp theo, cập nhật các thống kê tổng hợp.
    if new.empty:
        return current if current is not None else pd.DataFrame()
    if current is None or current.empty:
        return new

    keys = ["stay_id", "variable"]
    combined = pd.concat([current.reset_index(), new.reset_index()], ignore_index=True)
    
    grouped = combined.groupby(keys, observed=False)
    numeric = grouped.agg(
        min=("min", "min"),
        max=("max", "max"),
        sum=("sum", "sum"),
        count=("count", "sum"),
        sumsq=("sumsq", "sum"),
    )
    
    first_idx = grouped["first_hour"].idxmin()
    last_idx = grouped["last_hour"].idxmax()
    first = combined.loc[first_idx, keys + ["first", "first_hour"]].set_index(keys)
    last = combined.loc[last_idx, keys + ["last", "last_hour"]].set_index(keys)
    
    return numeric.join(first).join(last)


def finalize_group_summary(summary: pd.DataFrame, hours: int) -> pd.DataFrame:
    # Tính mean, std, slope và pivot sang format wide sau khi đã gộp tất cả các chunk.
    if summary is None or summary.empty:
        return pd.DataFrame()
    summary = summary.reset_index()
    summary["mean"] = summary["sum"] / summary["count"]
    var = (summary["sumsq"] - (summary["sum"] ** 2 / summary["count"])) / (summary["count"] - 1)
    summary["std"] = np.sqrt(var.clip(lower=0))
    summary.loc[summary["count"] <= 1, "std"] = np.nan
    elapsed_days = (summary["last_hour"] - summary["first_hour"]) / 24
    summary["slope"] = np.where(elapsed_days.abs() > 1e-9, (summary["last"] - summary["first"]) / elapsed_days, np.nan)

    value_cols = ["first", "last", "min", "max", "mean", "std", "count", "slope"]
    wide_parts = []
    # Tạo các cột wide cho mỗi biến và mỗi thống kê.
    for stat in value_cols:
        wide = summary.pivot(index="stay_id", columns="variable", values=stat)
        wide.columns = [f"{col}_{stat}_{hours}h" for col in wide.columns]
        wide_parts.append(wide)
    return pd.concat(wide_parts, axis=1).reset_index()


def keep_rows_in_icu_window(rows: pd.DataFrame, hours: int) -> pd.DataFrame:
    # Giữ các dòng có charttime trong hours đầu tiên kể từ intime của ICU stay.
    if rows.empty:
        return rows
    rows = rows.dropna(subset=["valuenum", "charttime", "intime"]).copy()
    rows["hours_from_icu_intime"] = (rows["charttime"] - rows["intime"]).dt.total_seconds() / 3600
    rows = rows[(rows["hours_from_icu_intime"] >= 0) & (rows["hours_from_icu_intime"] <= hours)]
    
    return rows[["stay_id", "variable", "charttime", "valuenum", "intime", "hours_from_icu_intime"]]

def collect_chartevents(mimic_dir: Path, stays: pd.DataFrame, hours: int, chunksize: int) -> pd.DataFrame:
    # Trích xuất và tóm tắt vital signs từ bảng chartevents rất lớn, chỉ giữ các dòng trong hours đầu tiên của ICU stay.
    OUTPUT_DIR.mkdir(exist_ok=True)
    cache_path = OUTPUT_DIR / f"cache_icu_vitals_{hours}h.csv"
    if cache_path.exists():
        log(f"\nLoading cached ICU vital summaries: {cache_path}")
        return pd.read_csv(cache_path)

    path = find_table(mimic_dir / "icu", "chartevents.csv")
    item_map = itemid_to_name(VITAL_ITEMIDS)
    itemids = set(item_map)
    stay_lookup = stays[["stay_id", "intime"]].copy()
    # collected = []
    summary_accumulator = None
    total = 0
    kept = 0
    chunk_no = 0
    log("\nExtracting ICU vital signs from chartevents:")
    
    # Xử lý chartevents theo chunk.
    for chunk in pd.read_csv(path, usecols=["subject_id", "stay_id", "charttime", "itemid", "valuenum"], chunksize=chunksize, low_memory=False):
        chunk_no += 1
        total += len(chunk)
        chunk = chunk[chunk["itemid"].isin(itemids) & chunk["valuenum"].notna()]
        if chunk.empty:
            continue
        chunk["variable"] = chunk["itemid"].map(item_map)
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.merge(stay_lookup, on="stay_id", how="inner")
        chunk = keep_rows_in_icu_window(chunk, hours)
        kept += len(chunk)
        if not chunk.empty:
            summary_accumulator = merge_group_summaries(summary_accumulator, summarize_chunk(chunk))
    if summary_accumulator is None or summary_accumulator.empty:
        log("  No matching vital sign rows found")
        return pd.DataFrame()
    log(f"Vitals rows kept in first {hours}h: {kept:,}")
    summary = finalize_group_summary(summary_accumulator, hours)
    summary.to_csv(cache_path, index=False)
    log(f"Cached vital summaries: {cache_path}")
    
    return summary


def collect_labevents(mimic_dir: Path, stays: pd.DataFrame, hours: int, chunksize: int) -> pd.DataFrame:
    # Trích xuất và tóm tắt labs từ bảng labevents rất lớn, chỉ giữ các dòng trong hours đầu tiên của ICU stay.
    OUTPUT_DIR.mkdir(exist_ok=True)
    cache_path = OUTPUT_DIR / f"cache_icu_labs_{hours}h.csv"
    if cache_path.exists():
        log(f"\nLoading cached ICU lab summaries: {cache_path}")
        return pd.read_csv(cache_path)

    path = find_table(mimic_dir / "hosp", "labevents.csv")
    item_map = itemid_to_name(LAB_ITEMIDS)
    itemids = set(item_map)
    stay_lookup = stays[["stay_id", "subject_id", "hadm_id", "intime"]].copy()
    # collected = []
    summary_accumulator = None
    total = 0
    kept = 0
    chunk_no = 0
    log("\nExtracting labs from labevents:")
    
    # Xử lý labevents theo chunk và chỉ giữ các itemid cần thiết gần thời điểm vào ICU.
    for chunk in pd.read_csv(path, usecols=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"], chunksize=chunksize, low_memory=False):
        chunk_no += 1
        total += len(chunk)
        chunk = chunk[chunk["itemid"].isin(itemids) & chunk["valuenum"].notna()]
        if chunk.empty:
            continue
        chunk["variable"] = chunk["itemid"].map(item_map)
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.merge(stay_lookup, on=["subject_id", "hadm_id"], how="inner")
        chunk = keep_rows_in_icu_window(chunk, hours)
        kept += len(chunk)
        if not chunk.empty:
            summary_accumulator = merge_group_summaries(
                summary_accumulator,
                summarize_chunk(chunk),
            )
    if summary_accumulator is None or summary_accumulator.empty:
        log("  No matching lab rows found")
        return pd.DataFrame()
    log(f"Lab rows kept in first {hours}h: {kept:,}")
    summary = finalize_group_summary(summary_accumulator, hours)
    summary.to_csv(cache_path, index=False)
    log(f"Cached lab summaries: {cache_path}")
    return summary


def build_cohort() -> pd.DataFrame:
    # Xây dựng cohort ICU longitudinal bằng cách trích xuất stays, vitals và labs, sau đó gộp lại và lưu ra file CSV.
    log("[icu_longitudinal_builder] Starting")
    log(f"  MIMIC-IV directory: {INPUT_DIR.resolve()}")
    log(f"  Window: first {hours} ICU hours")
    log(f"  Outcome horizon: {HORIZON_DAYS} days")

    # Bắt đầu với một dòng cho ICU stay đầu tiên của mỗi bệnh nhân, sau đó thêm summary vital/lab.
    stays = prepare_icu_base(mimic_dir=INPUT_DIR, horizon_days=HORIZON_DAYS)
    log(f"\nBase ICU cohort:")
    log(f"  ✓ ICU rows: {len(stays):,}")
    log(f"  ✓ Subjects: {stays['subject_id'].nunique():,}")
    log(f"  ✓ Events {HORIZON_DAYS}d: {int(stays[f'death_{HORIZON_DAYS}d'].sum()):,}")
    
    vitals = collect_chartevents(mimic_dir=INPUT_DIR, stays=stays, hours=hours, chunksize=CHUNKSIZE)
    
    labs = collect_labevents(mimic_dir=INPUT_DIR, stays=stays, hours=hours, chunksize=CHUNKSIZE)

    # Ghép các bảng feature wide trở lại cohort nền theo stay_id.
    cohort = stays.copy()
    for features in [vitals, labs]:
        if not features.empty:
            cohort = cohort.merge(features, on="stay_id", how="left")

    cohort["source_cohort"] = "mimiciv_icu_longitudinal"
    cohort.to_csv(OUTPUT_FILE, index=False)
    log(f"\nEXPORT COMPLETE")
    log(f"  Saved {len(cohort):,} ICU rows x {cohort.shape[1]:,} columns")
    log(f"  Output: {OUTPUT_FILE.resolve()}")
    
    return cohort


if __name__ == "__main__":
    build_cohort()
