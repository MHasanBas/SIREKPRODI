import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import random
import os
import pickle
import json
import math

DEFAULT_RANDOM_SEED = 42
DEFAULT_GA_POP_SIZE = 20
DEFAULT_GA_GENERATIONS = 25
DEFAULT_GA_MUTATION_RATE = 0.05
DEFAULT_GA_MAX_STAGNANT = 8
DEFAULT_TRAINING_RUNS = 1
DEFAULT_GA_MUTATION_STRATEGY = "adaptive_early_high_mutation"
DEFAULT_GA_FULL_EVAL_TOP_N = 2



DERIVED_FEATURE_PREFIXES = ("Distance_to_", "Membership_")
DERIVED_FEATURE_COLUMNS = {"CLUSTER", "DBI_SCORE", "SILHOUETTE_SCORE"}
SCHOOL_NAME_COLUMNS = {"ASAL SEKOLAH", "SEKOLAH", "NAMA SEKOLAH", "SCHOOL NAME"}
ID_FEATURE_COLUMNS = {"NPSN", "NPSN SEKOLAH"}
CATEGORICAL_FEATURE_COLUMNS = [
    "JURUSAN SEKOLAH",
    "PROVINSI SEKOLAH",
]
NUMERIC_FEATURE_COLUMNS = [
    "NILAI KESELURUHAN",
    "IPK",
]
REQUIRED_CATEGORICAL_COLUMNS = []


def _is_excluded_training_column(column):
    column_upper = str(column).upper().strip()
    return (
        column_upper in SCHOOL_NAME_COLUMNS
        or column_upper in ID_FEATURE_COLUMNS
        or "NPSN" in column_upper
        or column_upper in DERIVED_FEATURE_COLUMNS
        or any(column_upper.startswith(prefix.upper()) for prefix in DERIVED_FEATURE_PREFIXES)
    )


def _build_feature_matrix(df):
    for col in REQUIRED_CATEGORICAL_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Kolom kategorikal '{col}' tidak ditemukan di DataFrame!")

    categorical_cols = [
        col for col in CATEGORICAL_FEATURE_COLUMNS
        if col in df.columns and not _is_excluded_training_column(col)
    ]
    numeric_cols = [
        col for col in NUMERIC_FEATURE_COLUMNS
        if col in df.columns and not _is_excluded_training_column(col)
    ]
    numeric_cols = [
        col for col in numeric_cols
        if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype == object
    ]
    numeric_source = df[numeric_cols].copy()
    numeric_cols = [
        col for col in numeric_source.columns
        if not _is_excluded_training_column(col)
    ]

    numeric_features = numeric_source[numeric_cols].copy()
    numeric_features = numeric_features.apply(pd.to_numeric, errors="coerce")
    numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)

    all_nan_cols = [col for col in numeric_features.columns if numeric_features[col].isna().all()]
    if all_nan_cols:
        print(f"Peringatan: menghapus kolom numerik kosong: {all_nan_cols}")
        numeric_features = numeric_features.drop(columns=all_nan_cols)

    if not numeric_features.empty:
        numeric_features = numeric_features.fillna(numeric_features.median()).fillna(0)

    categorical_features = pd.DataFrame(index=df.index)
    if categorical_cols:
        categorical_source = df[categorical_cols].copy()
        categorical_source = categorical_source.fillna("UNKNOWN").astype(str)
        categorical_source = categorical_source.apply(lambda s: s.str.strip().str.upper().replace("", "UNKNOWN"))
        categorical_features = pd.get_dummies(categorical_source, columns=categorical_cols, dtype=float)

    feature_df = pd.concat([numeric_features, categorical_features], axis=1)
    if feature_df.empty:
        raise ValueError("Tidak ada fitur numerik atau kategorikal yang tersedia untuk scaling!")

    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return feature_df


def prepare_data(df):
    df = df.copy()
    if df.empty:
        raise ValueError("DataFrame kosong! Tidak bisa diproses.")

    feature_df = _build_feature_matrix(df)

    # Standard scale ONLY the numeric columns (e.g., NILAI KESELURUHAN, IPK)
    numeric_cols = [col for col in ["NILAI KESELURUHAN", "IPK"] if col in feature_df.columns]
    categorical_cols = [col for col in feature_df.columns if col not in numeric_cols]

    df_scaled_parts = []
    if numeric_cols:
        scaler = StandardScaler()
        scaled_numeric = scaler.fit_transform(feature_df[numeric_cols])
        df_scaled_parts.append(pd.DataFrame(scaled_numeric, index=feature_df.index, columns=numeric_cols))
    else:
        # Fallback if no numeric columns are found (should not happen in practice)
        scaler = StandardScaler()

    if categorical_cols:
        # Weight categorical features by 0.1 to avoid Standard Scaling amplification artifacts
        weighted_categorical = feature_df[categorical_cols] * 0.1
        df_scaled_parts.append(weighted_categorical)

    if df_scaled_parts:
        feature_df_scaled = pd.concat(df_scaled_parts, axis=1)
    else:
        feature_df_scaled = feature_df

    df_scaled = feature_df_scaled.to_numpy()
    df_scaled = np.nan_to_num(df_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return df_scaled, list(feature_df.columns), scaler


def find_optimal_k(data, k_range=range(2, 8)):
    """
    [Saran #2] Metode Elbow untuk mencari jumlah cluster optimal.
    Menggunakan inertia dan mendeteksi "siku" (elbow point) secara otomatis.
    Mengembalikan k optimal dan dictionary data elbow untuk divisualisasikan.
    """
    inertias = []
    k_values = list(k_range)

    for k in k_values:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km.fit(data)
        inertias.append(km.inertia_)

    # Deteksi elbow menggunakan metode "jarak terbesar dari garis lurus"
    # (perpendicular distance / knee detection)
    if len(k_values) < 2:
        return k_values[0], {"k_values": k_values, "inertias": inertias, "optimal_k": k_values[0]}

    # Vektor dari titik pertama ke titik terakhir
    p1 = np.array([k_values[0], inertias[0]])
    p2 = np.array([k_values[-1], inertias[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    max_dist = -1
    optimal_k = k_values[1]  # default fallback ke k=3

    for i, (k, inertia) in enumerate(zip(k_values, inertias)):
        point = np.array([k, inertia])
        # Normalisasi agar skala k dan inertia sebanding
        point_norm = np.array([
            (k - k_values[0]) / (k_values[-1] - k_values[0]),
            (inertia - inertias[-1]) / (inertias[0] - inertias[-1] + 1e-10)
        ])
        p1_norm = np.array([0.0, 1.0])
        p2_norm = np.array([1.0, 0.0])
        line_norm = p2_norm - p1_norm
        # Jarak tegak lurus dari titik ke garis
        dist = abs(np.cross(line_norm, point_norm - p1_norm)) / np.linalg.norm(line_norm)
        if dist > max_dist:
            max_dist = dist
            optimal_k = k

    elbow_data = {
        "k_values": k_values,
        "inertias": [round(v, 4) for v in inertias],
        "optimal_k": int(optimal_k)
    }
    print(f"Elbow analysis: k_values={k_values}, inertias={[round(v,2) for v in inertias]}")
    print(f"K optimal terdeteksi: {optimal_k}")

    return int(optimal_k), elbow_data


def evaluate_scores_kmeans(data, centroids, n_clusters, use_silhouette=True):
    """
    Menghitung fitness multi-objective untuk GA.
    Fitness lebih besar = lebih baik, dengan kombinasi:
      (0.6 * (1 / DBI)) + (0.4 * silhouette_score)
    """
    kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    unique = np.unique(labels)

    # Jika ada cluster kosong (< 2 cluster terisi), beri penalti minimal
    if len(unique) < 2:
        return float('-inf')

    try:
        dbi = davies_bouldin_score(data, labels)
        if dbi <= 0 or not np.isfinite(dbi):
            return float('-inf')
        if not use_silhouette:
            return 1.0 / dbi
        silhouette = silhouette_score(data, labels, random_state=42)
        if not np.isfinite(silhouette):
            return float('-inf')
        return (0.6 * (1.0 / dbi)) + (0.4 * silhouette)
    except Exception:
        return float('-inf')


def crossover(parent1, parent2):
    cp = random.randint(1, parent1.shape[1] - 1)
    child1 = np.concatenate((parent1[:, :cp], parent2[:, cp:]), axis=1)
    child2 = np.concatenate((parent2[:, :cp], parent1[:, cp:]), axis=1)
    return child1, child2


def genetic_algorithm_kmeans(
    data,
    n_clusters=3,
    pop_size=DEFAULT_GA_POP_SIZE,
    mutation_rate=DEFAULT_GA_MUTATION_RATE,
    generations=DEFAULT_GA_GENERATIONS,
    max_stagnant=DEFAULT_GA_MAX_STAGNANT,
):
    """
    Genetic Algorithm untuk optimasi centroid KMeans.
    Fungsi fitness: kombinasi DBI dan Silhouette Score - maximize.
    """
    population = np.random.normal(0, 1, size=(pop_size, n_clusters, data.shape[1]))
    best_centroids = None
    best_score = float('-inf')  # Fitness: semakin besar semakin baik
    stagnant = 0

    for generation in range(generations):
        quick_scores = [evaluate_scores_kmeans(data, ind, n_clusters, use_silhouette=False) for ind in population]
        ranked_quick = sorted(zip(quick_scores, population), key=lambda x: x[0], reverse=True)

        top_n = min(DEFAULT_GA_FULL_EVAL_TOP_N, len(ranked_quick))
        ranked = []
        for idx, (quick_score, individual) in enumerate(ranked_quick):
            if idx < top_n:
                full_score = evaluate_scores_kmeans(data, individual, n_clusters, use_silhouette=True)
                ranked.append((full_score, individual))
            else:
                ranked.append((quick_score * 0.6, individual))
        ranked.sort(key=lambda x: x[0], reverse=True)
        current_score, current_best_centroids = ranked[0]

        if current_score > best_score:
            best_score = current_score
            best_centroids = current_best_centroids
            stagnant = 0
        else:
            stagnant += 1

        if stagnant >= max_stagnant:
            break

        parents = [x[1] for x in ranked[:pop_size // 2]]
        children = []
        while len(children) < pop_size:
            p1, p2 = random.sample(parents, 2)
            c1, c2 = crossover(p1, p2)
            children += [c1, c2]
        population = np.array(children)

        # Mutasi lebih agresif di awal agar GA lebih mudah keluar dari local optima.
        progress_ratio = generation / max(generations - 1, 1)
        if progress_ratio < 0.33:
            current_mutation_rate = max(mutation_rate, 0.28)
            mutation_noise = 0.9
        elif progress_ratio < 0.66:
            current_mutation_rate = max(mutation_rate, 0.14)
            mutation_noise = 0.6
        else:
            current_mutation_rate = mutation_rate
            mutation_noise = 0.35

        for i in range(pop_size):
            if random.random() < current_mutation_rate:
                population[i] += np.random.normal(0, mutation_noise, size=population[i].shape)

    return best_centroids, best_score


def _safe_ga_kmeans(
    data,
    n_clusters,
    pop_size=DEFAULT_GA_POP_SIZE,
    mutation_rate=DEFAULT_GA_MUTATION_RATE,
    generations=DEFAULT_GA_GENERATIONS,
    max_stagnant=DEFAULT_GA_MAX_STAGNANT,
):
    """
    Wrapper GA yang aman - jika GA menghasilkan centroids=None
    (semua individu menghasilkan cluster tunggal / DBI=inf),
    fallback ke KMeans k-means++ standar.
    """
    best_centroids, best_score = genetic_algorithm_kmeans(
        data, n_clusters=n_clusters,
        pop_size=pop_size, mutation_rate=mutation_rate, generations=generations,
        max_stagnant=max_stagnant
    )
    if best_centroids is None or not math.isfinite(best_score):
        print("  Peringatan: GA tidak menemukan solusi valid, fallback ke k-means++")
        km_fallback = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        km_fallback.fit(data)
        best_centroids = km_fallback.cluster_centers_
        try:
            dbi = davies_bouldin_score(data, km_fallback.labels_)
            silhouette = silhouette_score(data, km_fallback.labels_, random_state=42)
            if dbi <= 0 or not np.isfinite(dbi) or not np.isfinite(silhouette):
                best_score = float('-inf')
            else:
                best_score = (0.6 * (1.0 / dbi)) + (0.4 * silhouette)
        except Exception:
            best_score = float('-inf')
    return best_centroids, best_score


def _compute_cluster_metrics(df_scaled, labels, n_clusters):
    """
    [Saran #1] Menghitung Silhouette Score, Davies-Bouldin Index, dan Calinski-Harabasz Score.
    Mengembalikan dict berisi metrik-metrik tersebut.
    """
    metrics = {"silhouette_score": None, "dbi_score": None, "ch_score": None}
    unique_labels = np.unique(labels)

    # Minimal 2 cluster terisi dan minimal 2 sampel agar metrik valid
    if len(unique_labels) < 2 or len(df_scaled) < 2:
        print("Peringatan: metrik tidak dapat dihitung: cluster tidak cukup.")
        return metrics

    try:
        sil = silhouette_score(df_scaled, labels, random_state=42)
        metrics["silhouette_score"] = round(float(sil), 4)
        print(f"   Silhouette score: {metrics['silhouette_score']:.4f} (mendekati 1 = sangat baik)")
    except Exception as e:
        print(f"Peringatan: gagal hitung silhouette score: {e}")

    try:
        dbi = davies_bouldin_score(df_scaled, labels)
        metrics["dbi_score"] = round(float(dbi), 4)
        print(f"   Davies-Bouldin index: {metrics['dbi_score']:.4f} (mendekati 0 = sangat baik)")
    except Exception as e:
        print(f"Peringatan: gagal hitung DBI score: {e}")

    try:
        ch = calinski_harabasz_score(df_scaled, labels)
        metrics["ch_score"] = round(float(ch), 4)
        print(f"   Calinski-Harabasz index: {metrics['ch_score']:.4f} (semakin besar = sangat baik)")
    except Exception as e:
        print(f"Peringatan: gagal hitung CH score: {e}")

    return metrics


def _set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def _run_kmeans_single(
    df,
    n_clusters=None,
    random_seed=DEFAULT_RANDOM_SEED,
    ga_pop_size=DEFAULT_GA_POP_SIZE,
    ga_generations=DEFAULT_GA_GENERATIONS,
    ga_mutation_rate=DEFAULT_GA_MUTATION_RATE,
    ga_max_stagnant=DEFAULT_GA_MAX_STAGNANT,
):
    _set_random_seed(random_seed)
    df = df.copy()

    total_inertia = 0
    all_silhouette = []
    all_dbi = []
    all_ch = []
    all_dbi_baseline = []
    dbi_comparison_per_prodi = []
    elbow_data_global = None
    _n_clusters_fallback = 3
    used_k_values = set()

    df['Cluster'] = 0
    df['Distance_to_Assigned_Centroid'] = 0.0

    if 'PROGRAM STUDI' not in df.columns:
        print("Peringatan: kolom PROGRAM STUDI tidak ada. Menjalankan mode global fallback.")
        df_scaled, _, _ = prepare_data(df)

        print("Menjalankan elbow method untuk mencari k optimal...")
        optimal_k, elbow_data_global = find_optimal_k(df_scaled)
        active_k = optimal_k if n_clusters is None else n_clusters
        used_k_values.add(int(active_k))
        if n_clusters is None:
            print(f"k otomatis dari elbow: {active_k}")
        elif optimal_k != n_clusters:
            print(f"Info: elbow menyarankan k={optimal_k}, menggunakan k={active_k} (manual)")

        for i in range(active_k):
            df[f'Distance_to_Centroid_{i}'] = 0.0

        print("Menghitung baseline KMeans (k-means++) untuk perbandingan...")
        baseline_km = KMeans(n_clusters=active_k, init='k-means++', n_init=10, random_state=DEFAULT_RANDOM_SEED).fit(df_scaled)
        baseline_metrics = _compute_cluster_metrics(df_scaled, baseline_km.labels_, active_k)
        if baseline_metrics["dbi_score"] is not None:
            all_dbi_baseline.append(baseline_metrics["dbi_score"])
            print(f"   DBI baseline (KMeans): {baseline_metrics['dbi_score']:.4f}")

        best_centroids, _ = _safe_ga_kmeans(
            df_scaled,
            n_clusters=active_k,
            pop_size=ga_pop_size,
            generations=ga_generations,
            mutation_rate=ga_mutation_rate,
            max_stagnant=ga_max_stagnant,
        )
        kmeans = KMeans(n_clusters=active_k, init=best_centroids, n_init=1, random_state=DEFAULT_RANDOM_SEED).fit(df_scaled)
        total_inertia = float(kmeans.inertia_)
        df['Cluster'] = kmeans.labels_
        centroid_dist = pairwise_distances(df_scaled, kmeans.cluster_centers_)
        for i in range(active_k):
            df[f'Distance_to_Centroid_{i}'] = centroid_dist[:, i]
        df['Distance_to_Assigned_Centroid'] = centroid_dist[np.arange(len(df)), kmeans.labels_]

        print("Menghitung metrik evaluasi global (setelah GA)...")
        global_metrics = _compute_cluster_metrics(df_scaled, kmeans.labels_, active_k)
        all_silhouette.append(global_metrics["silhouette_score"])
        all_dbi.append(global_metrics["dbi_score"])
        if global_metrics["ch_score"] is not None:
            all_ch.append(global_metrics["ch_score"])
        _n_clusters_fallback = active_k

    else:
        prodi_groups = df.groupby('PROGRAM STUDI')
        print(f"Memulai clustering terisolasi untuk {len(prodi_groups)} Program Studi...")

        for prodi_name, subset in prodi_groups:
            idx = subset.index
            min_k = n_clusters if n_clusters is not None else 2

            if len(subset) < min_k:
                print(f"  -> [Abaikan] {prodi_name}: {len(subset)} baris (<{min_k})")
                continue

            try:
                df_scaled, _, _ = prepare_data(subset)
            except Exception as e:
                print(f"  -> [Error] {prodi_name}: {e}")
                continue

            max_k = min(6, len(subset) - 1)
            if max_k >= 2:
                print(f"  -> Elbow method untuk {prodi_name} (k=2..{max_k})...")
                local_optimal_k, local_elbow = find_optimal_k(df_scaled, k_range=range(2, max_k + 1))
                local_k = n_clusters if n_clusters is not None else local_optimal_k
                if elbow_data_global is None:
                    elbow_data_global = local_elbow
            else:
                local_k = _n_clusters_fallback

            used_k_values.add(int(local_k))
            print(f"  -> Memproses {prodi_name}: {len(subset)} baris. GA {local_k} Kluster...")

            baseline_km = KMeans(n_clusters=local_k, init='k-means++', n_init=10, random_state=DEFAULT_RANDOM_SEED).fit(df_scaled)
            baseline_m = _compute_cluster_metrics(df_scaled, baseline_km.labels_, local_k)
            if baseline_m["dbi_score"] is not None:
                all_dbi_baseline.append(baseline_m["dbi_score"])

            best_centroids, _ = _safe_ga_kmeans(
                df_scaled,
                n_clusters=local_k,
                pop_size=ga_pop_size,
                generations=ga_generations,
                mutation_rate=ga_mutation_rate,
                max_stagnant=ga_max_stagnant,
            )

            kmeans = KMeans(n_clusters=local_k, init=best_centroids, n_init=1, random_state=DEFAULT_RANDOM_SEED).fit(df_scaled)
            total_inertia += float(kmeans.inertia_)
            df.loc[idx, 'Cluster'] = kmeans.labels_

            centroid_dist = pairwise_distances(df_scaled, kmeans.cluster_centers_)
            for i in range(local_k):
                col = f'Distance_to_Centroid_{i}'
                if col not in df.columns:
                    df[col] = 0.0
                df.loc[idx, col] = centroid_dist[:, i]
            df.loc[idx, 'Distance_to_Assigned_Centroid'] = centroid_dist[np.arange(len(subset)), kmeans.labels_]

            _n_clusters_fallback = local_k

            print(f"     Menghitung metrik evaluasi untuk {prodi_name}...")
            metrics = _compute_cluster_metrics(df_scaled, kmeans.labels_, local_k)
            if baseline_m["dbi_score"] is not None and metrics["dbi_score"] is not None:
                imprv = ((baseline_m["dbi_score"] - metrics["dbi_score"]) / baseline_m["dbi_score"]) * 100
                print(f"     DBI: {baseline_m['dbi_score']:.4f} -> {metrics['dbi_score']:.4f} ({imprv:+.1f}%)")
                dbi_comparison_per_prodi.append({
                    "prodi": str(prodi_name),
                    "n": int(len(subset)),
                    "k": int(local_k),
                    "dbi_before": round(float(baseline_m["dbi_score"]), 4),
                    "dbi_after": round(float(metrics["dbi_score"]), 4),
                    "improvement_pct": round(float(imprv), 2),
                    "ch_score": round(float(metrics["ch_score"]), 2) if metrics["ch_score"] is not None else None
                })
            if metrics["silhouette_score"] is not None:
                all_silhouette.append(metrics["silhouette_score"])
            if metrics["dbi_score"] is not None:
                all_dbi.append(metrics["dbi_score"])
            if metrics["ch_score"] is not None:
                all_ch.append(metrics["ch_score"])

    final_k = _n_clusters_fallback
    sorted_k_values = sorted(used_k_values)
    if len(sorted_k_values) > 1:
        clusters_used_meta = f"{sorted_k_values[0]}-{sorted_k_values[-1]}"
    elif len(sorted_k_values) == 1:
        clusters_used_meta = str(sorted_k_values[0])
    else:
        clusters_used_meta = str(final_k)

    print("\nMenghitung PCA 2D untuk visualisasi...")
    pca_meta = None
    try:
        pca_features = _build_feature_matrix(df)
        if pca_features.shape[1] < 2:
            raise ValueError(f"Hanya {pca_features.shape[1]} fitur tersedia setelah pembersihan - PCA butuh minimal 2.")

        n_components = min(2, pca_features.shape[1])
        scaled_for_pca, _, _ = prepare_data(df)
        pca = PCA(n_components=n_components, random_state=DEFAULT_RANDOM_SEED)
        coords = pca.fit_transform(scaled_for_pca)

        if coords.shape[1] == 1:
            coords = np.hstack([coords, np.zeros((len(coords), 1))])

        n_sample = min(2000, len(df))
        sample_idx = np.random.choice(len(df), n_sample, replace=False)

        pca_points = []
        for i in sample_idx:
            row = df.iloc[i]
            pca_points.append({
                "x": round(float(coords[i, 0]), 4),
                "y": round(float(coords[i, 1]), 4),
                "cluster": int(row['Cluster']),
                "prodi": str(row.get('PROGRAM STUDI', '')) if 'PROGRAM STUDI' in df.columns else '',
                "sekolah": str(row.get('ASAL SEKOLAH', '')) if 'ASAL SEKOLAH' in df.columns else '',
            })

        pca_meta = {
            "variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
            "total_variance_explained": round(float(sum(pca.explained_variance_ratio_)), 4),
            "n_sample": n_sample,
            "n_total": len(df),
            "points": pca_points,
        }
    except Exception as e:
        print(f"Peringatan: PCA gagal dihitung: {e}")

    avg_silhouette = round(float(np.mean([v for v in all_silhouette if v is not None])), 4) if any(v is not None for v in all_silhouette) else None
    avg_dbi_ga = round(float(np.mean([v for v in all_dbi if v is not None])), 4) if any(v is not None for v in all_dbi) else None
    avg_dbi_base = round(float(np.mean([v for v in all_dbi_baseline if v is not None])), 4) if any(v is not None for v in all_dbi_baseline) else None
    avg_ch = round(float(np.mean([v for v in all_ch if v is not None])), 4) if any(v is not None for v in all_ch) else None

    dbi_improvement = None
    if avg_dbi_base and avg_dbi_ga:
        dbi_improvement = round(((avg_dbi_base - avg_dbi_ga) / avg_dbi_base) * 100, 2)

    return {
        "data": df,
        "random_seed": random_seed,
        "final_k": final_k,
        "n_clusters_used": clusters_used_meta,
        "k_values_used": sorted_k_values,
        "pca_meta": pca_meta,
        "silhouette_score": avg_silhouette,
        "dbi_score": avg_dbi_ga,
        "dbi_before_ga": avg_dbi_base,
        "dbi_after_ga": avg_dbi_ga,
        "dbi_improvement_pct": dbi_improvement,
        "ch_score": avg_ch,
        "total_inertia": round(float(total_inertia), 4),
        "dbi_comparison_per_prodi": sorted(dbi_comparison_per_prodi, key=lambda x: x["improvement_pct"], reverse=True) if dbi_comparison_per_prodi else [],
        "elbow_data": elbow_data_global,
    }


def jalankan_kmeans(df, n_clusters=3, save_path="models/model_utama/", progress_callback=None):
    """
    Menjalankan pipeline KMeans + Genetic Algorithm.
    Default memakai K=3. Jika n_clusters=None, k optimal ditentukan otomatis via Elbow Method.
    """
    os.makedirs(save_path, exist_ok=True)
    ga_params = {
        "random_seed": DEFAULT_RANDOM_SEED,
        "population_size": DEFAULT_GA_POP_SIZE,
        "generations": DEFAULT_GA_GENERATIONS,
        "mutation_rate": DEFAULT_GA_MUTATION_RATE,
        "max_stagnant": DEFAULT_GA_MAX_STAGNANT,
    }
    run_results = []
    for run_idx in range(DEFAULT_TRAINING_RUNS):
        run_seed = DEFAULT_RANDOM_SEED + run_idx
        print(f"\nTraining run {run_idx + 1}/{DEFAULT_TRAINING_RUNS} dengan seed={run_seed}")
        if progress_callback:
            progress_callback(run_idx + 1, DEFAULT_TRAINING_RUNS, "running")
        run_results.append(_run_kmeans_single(
            df,
            n_clusters=n_clusters,
            random_seed=run_seed,
            ga_pop_size=DEFAULT_GA_POP_SIZE,
            ga_generations=DEFAULT_GA_GENERATIONS,
            ga_mutation_rate=DEFAULT_GA_MUTATION_RATE,
            ga_max_stagnant=DEFAULT_GA_MAX_STAGNANT,
        ))
        if progress_callback:
            progress_callback(run_idx + 1, DEFAULT_TRAINING_RUNS, "completed")

    valid_results = [r for r in run_results if r.get("dbi_after_ga") is not None]
    best_result = min(valid_results, key=lambda r: r["dbi_after_ga"]) if valid_results else run_results[0]

    output_path = os.path.join(save_path, f"hasil_kmeans_{best_result['final_k']}_cluster.xlsx")
    best_result["data"].to_excel(output_path, index=False)
    with open(os.path.join(save_path, "hasil_kmeans_3cluster.pkl"), "wb") as f:
        pickle.dump({"data": best_result["data"]}, f)

    print("\nKMeans + GA selesai, hasil terbaik disimpan:", output_path)
    if best_result["pca_meta"] is not None:
        pca_path = os.path.join(save_path, "pca_data.json")
        with open(pca_path, "w") as f:
            json.dump(best_result["pca_meta"], f)
        print(f"   PCA data disimpan: {best_result['pca_meta']['n_sample']} titik, variance explained={best_result['pca_meta']['total_variance_explained']:.1%}")

    avg_run_dbi = round(float(np.mean([r["dbi_after_ga"] for r in valid_results])), 4) if valid_results else None
    avg_run_silhouette = round(float(np.mean([r["silhouette_score"] for r in run_results if r.get("silhouette_score") is not None])), 4) if any(r.get("silhouette_score") is not None for r in run_results) else None
    avg_run_ch = round(float(np.mean([r["ch_score"] for r in run_results if r.get("ch_score") is not None])), 4) if any(r.get("ch_score") is not None for r in run_results) else None

    print("\nRingkasan Metrik Evaluasi (Run Terbaik):")
    print(f"   Rata-rata Silhouette Score    : {best_result['silhouette_score']}")
    print(f"   DBI Sebelum GA (KMeans)       : {best_result['dbi_before_ga']}  (mendekati 0 = baik)")
    print(f"   DBI Setelah  GA (KMeans+GA)   : {best_result['dbi_after_ga']}  (mendekati 0 = baik)")
    if best_result["dbi_improvement_pct"] is not None:
        arrow = "membaik" if best_result["dbi_improvement_pct"] > 0 else "memburuk"
        print(f"   Peningkatan DBI oleh GA       : {best_result['dbi_improvement_pct']:+.2f}% ({arrow})")
    print(f"   Calinski-Harabasz Score (GA)  : {best_result['ch_score']}  (semakin besar = baik)")
    print(f"   Total Inertia (GA)            : {best_result['total_inertia']:.4f}")

    meta_path = os.path.join(save_path, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    meta["silhouette_score"] = best_result["silhouette_score"]
    meta["dbi_score"] = best_result["dbi_after_ga"]
    meta["dbi_before_ga"] = best_result["dbi_before_ga"]
    meta["dbi_after_ga"] = best_result["dbi_after_ga"]
    meta["dbi_improvement_pct"] = best_result["dbi_improvement_pct"]
    meta["ch_score"] = best_result["ch_score"]
    meta["total_inertia"] = best_result["total_inertia"]
    meta["n_clusters_used"] = best_result["n_clusters_used"]
    meta["k_values_used"] = best_result["k_values_used"]
    meta["pca_variance_explained"] = best_result["pca_meta"].get("total_variance_explained") if best_result["pca_meta"] else None
    meta["dbi_comparison_per_prodi"] = best_result["dbi_comparison_per_prodi"]
    meta["feature_policy"] = "NILAI KESELURUHAN/IPK + JURUSAN SEKOLAH + PROVINSI SEKOLAH; STATUS hanya untuk cleaning; PROGRAM STUDI hanya untuk grouping per prodi"
    meta["k_selection"] = "elbow" if n_clusters is None else f"manual_k_{n_clusters}"
    meta["training_numeric_features"] = NUMERIC_FEATURE_COLUMNS
    meta["training_categorical_features"] = CATEGORICAL_FEATURE_COLUMNS
    if best_result["elbow_data"]:
        meta["elbow_data"] = best_result["elbow_data"]
    meta["random_seed"] = best_result["random_seed"]
    meta["best_run_seed"] = best_result["random_seed"]
    meta["population_size"] = ga_params["population_size"]
    meta["generations"] = ga_params["generations"]
    meta["mutation_rate"] = ga_params["mutation_rate"]
    meta["mutation_strategy"] = DEFAULT_GA_MUTATION_STRATEGY
    meta["mutation_schedule"] = {
        "early_phase": {"mutation_rate_min": max(DEFAULT_GA_MUTATION_RATE, 0.28), "noise_std": 0.9},
        "mid_phase": {"mutation_rate_min": max(DEFAULT_GA_MUTATION_RATE, 0.14), "noise_std": 0.6},
        "late_phase": {"mutation_rate_base": DEFAULT_GA_MUTATION_RATE, "noise_std": 0.35},
    }
    meta["training_runs"] = DEFAULT_TRAINING_RUNS
    meta["run_seeds"] = [DEFAULT_RANDOM_SEED + i for i in range(DEFAULT_TRAINING_RUNS)]
    meta["avg_dbi_after_ga_runs"] = avg_run_dbi
    meta["avg_silhouette_runs"] = avg_run_silhouette
    meta["avg_ch_runs"] = avg_run_ch
    # Backward-compatible keys used by existing dashboard templates/routes.
    meta["avg_dbi_after_ga_3_runs"] = avg_run_dbi
    meta["avg_silhouette_3_runs"] = avg_run_silhouette

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"   Metrik disimpan ke {meta_path}")

    print("\nStatistik Singkat KMeans + GA")
    print(f"Total Inertia Gabungan: {best_result['total_inertia']:.4f}")
    if best_result["total_inertia"] > 0:
        print("\nDistribusi Global Cluster (%) :")
        print(best_result["data"]['Cluster'].value_counts(normalize=True).mul(100).round(1))

    unique_clusters = sorted(best_result["data"]['Cluster'].unique())
    for i in unique_clusters:
        df_cluster = best_result["data"][best_result["data"]['Cluster'] == i].copy()
        if df_cluster.empty:
            continue

        df_cluster_group = df_cluster.groupby(['ASAL SEKOLAH', 'KOTA SEKOLAH', 'PROVINSI SEKOLAH']).agg(
            Cluster=('Cluster', 'first'),
            **{
                'Jumlah Mahasiswa': ('NILAI KESELURUHAN', 'count'),
                'Nilai Mean': ('NILAI KESELURUHAN', 'mean'),
                'Nilai Range': ('NILAI KESELURUHAN', lambda x: x.max() - x.min()),
            }
        ).reset_index()

        cluster_dir = os.path.join(save_path, f'cluster_{i}')
        os.makedirs(cluster_dir, exist_ok=True)
        df_cluster_group.to_excel(os.path.join(cluster_dir, f'detail_cluster_{i}.xlsx'), index=False)
