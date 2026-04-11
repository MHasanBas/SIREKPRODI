import pandas as pd

def _norm_series(series: pd.Series) -> pd.Series:
    # Normalisasi min-max yang aman untuk series konstan.
    s_max = series.max()
    s_min = series.min()
    if s_max == s_min:
        return pd.Series(0, index=series.index, dtype="float64")
    return (series - s_min) / (s_max - s_min)

def _norm_in_group(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    # Normalisasi min-max per group, aman untuk group dengan nilai konstan.
    g = df.groupby(group_col)[value_col]
    min_v = g.transform("min")
    max_v = g.transform("max")
    denom = (max_v - min_v)
    norm = (df[value_col] - min_v) / denom.where(denom != 0, 1)
    return norm.where(denom != 0, 0).fillna(0)

def hitung_poin_prestasi(teks):
    # Return: (poin_akademik, poin_non_akademik, teks_akademik, teks_non_akademik)
    if pd.isna(teks) or not isinstance(teks, str):
        return 0, 0, "", ""

    t_clean = str(teks).strip()
    t = t_clean.lower()

    # Filter karakter sampah
    sampah = t.replace("-", "").replace("tingkat", "").strip()
    if not sampah or len(sampah) < 2 or "tidak ada" in t or "kosong" in t:
        return 0, 0, "", ""

    # 1) Peringkat Poin (Rank)
    rank_poin = 1
    if "juara 1" in t or "1 /" in t or "pertama" in t or "emas" in t:
        rank_poin = 3
    elif "juara 2" in t or "2 /" in t or "kedua" in t or "perak" in t:
        rank_poin = 2

    # 2) Tingkat Poin (Level)
    level_poin = 1
    if "internasional" in t or "international" in t or "dunia" in t or "nasional" in t or "national" in t:
        level_poin = 3
    elif "provinsi" in t or "regional" in t or "jatim" in t:
        level_poin = 2

    base_skor = rank_poin * level_poin

    # 3) Kategori (Multiplier: Akad x2, Non x1)
    akademik_words = [
        "olimpiade", "sains", "cerdas cermat", "matematika", "komputer", "karya ilmiah", "essay",
        "debat", "akademik", "bahasa", "fisika", "kimia", "biologi", "karya tulis", "osn",
        "ilmiah", "robot",
    ]

    if any(w in t for w in akademik_words):
        return base_skor * 2, 0, t_clean, ""
    return 0, base_skor * 1, "", t_clean

def _ensure_prestasi_columns(df: pd.DataFrame) -> None:
    # Hitung sekali untuk df model aktif; hasilnya dicache bersama df.
    required = {"POIN_AKADEMIK", "POIN_NON_AKADEMIK", "TEKS_AKADEMIK", "TEKS_NON_AKADEMIK"}
    if required.issubset(set(df.columns)):
        return

    p1 = df["PRESTASI 1"] if "PRESTASI 1" in df.columns else pd.Series([""] * len(df), index=df.index)
    p2 = df["PRESTASI 2"] if "PRESTASI 2" in df.columns else pd.Series([""] * len(df), index=df.index)

    res1 = [hitung_poin_prestasi(v) for v in p1.tolist()]
    res2 = [hitung_poin_prestasi(v) for v in p2.tolist()]

    p1_a = [r[0] for r in res1]
    p1_na = [r[1] for r in res1]
    t1_a = [r[2] for r in res1]
    t1_na = [r[3] for r in res1]

    p2_a = [r[0] for r in res2]
    p2_na = [r[1] for r in res2]
    t2_a = [r[2] for r in res2]
    t2_na = [r[3] for r in res2]

    df["POIN_AKADEMIK"] = pd.Series([a + b for a, b in zip(p1_a, p2_a)], index=df.index)
    df["POIN_NON_AKADEMIK"] = pd.Series([a + b for a, b in zip(p1_na, p2_na)], index=df.index)

    df["TEKS_AKADEMIK"] = pd.Series(
        [(" | ".join([x for x in (a, b) if x])) for a, b in zip(t1_a, t2_a)],
        index=df.index,
    )
    df["TEKS_NON_AKADEMIK"] = pd.Series(
        [(" | ".join([x for x in (a, b) if x])) for a, b in zip(t1_na, t2_na)],
        index=df.index,
    )
