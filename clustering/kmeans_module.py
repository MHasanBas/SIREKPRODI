import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import pairwise_distances
import random
import os
import pickle


def prepare_data(df):
    random.seed(42)
    np.random.seed(42)
    df = df.copy()
    if df.empty:
        raise ValueError("DataFrame kosong! Tidak bisa diproses.")

    # Buang kolom turunan dari model sebelumnya agar tidak memicu NaN (Cluster, Distance, Membership, dsb.)
    derived_cols = [c for c in df.columns if c.startswith("Distance_to_") or c.startswith("Membership_")]
    derived_cols += ['Cluster', 'DBI_SCORE', 'SILHOUETTE_SCORE']
    df = df.drop(columns=[c for c in derived_cols if c in df.columns], errors='ignore')

    label_encoders = {}
    categorical_cols = ['ASAL SEKOLAH', 'KOTA SEKOLAH', 'PROVINSI SEKOLAH']
    optional_categorical = ['PROGRAM STUDI']
    for col in categorical_cols + optional_categorical:
        if col in df.columns:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        elif col in categorical_cols:
            raise ValueError(f"Kolom kategorikal '{col}' tidak ditemukan di DataFrame!")

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

    all_nan_cols = [c for c in numeric_df.columns if numeric_df[c].isna().all()]
    if all_nan_cols:
        print(f"⚠️ Menghapus kolom numerik kosong: {all_nan_cols}")
        numeric_df = numeric_df.drop(columns=all_nan_cols)
        df = df.drop(columns=all_nan_cols, errors='ignore')

    if numeric_df.empty:
        raise ValueError("Tidak ada kolom numerik yang tersedia untuk scaling!")

    medians = numeric_df.median()
    numeric_df = numeric_df.fillna(medians)

    if numeric_df.isna().any().any():
        remaining = int(numeric_df.isna().sum().sum())
        print(f"⚠️ Masih ada {remaining} nilai NaN setelah isi median, mengisi dengan 0 (fallback).")
        numeric_df = numeric_df.fillna(0)

    if numeric_df.isna().any().any():
        na_rows = numeric_df[numeric_df.isna().any(axis=1)].index
        print(f"⚠️ Menghapus {len(na_rows)} baris yang tetap mengandung NaN setelah imputasi.")
        numeric_df = numeric_df.drop(index=na_rows).reset_index(drop=True)
        df = df.drop(index=na_rows).reset_index(drop=True)

    # Sinkronkan kembali ke df sumber
    df.loc[:, numeric_df.columns] = numeric_df

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(numeric_df)
    df_scaled = np.nan_to_num(df_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return df_scaled, label_encoders, scaler


def evaluate_scores_kmeans(data, centroids, n_clusters):
    """
    Gunakan inertia (sum of squared distances) sebagai skor GA.
    Lebih cepat dibanding DBI/Silhouette karena tidak perlu jarak antar-label.
    """
    kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1)
    kmeans.fit(data)
    return kmeans.inertia_


def crossover(parent1, parent2):
    cp = random.randint(1, parent1.shape[1] - 1)
    child1 = np.concatenate((parent1[:, :cp], parent2[:, cp:]), axis=1)
    child2 = np.concatenate((parent2[:, :cp], parent1[:, cp:]), axis=1)
    return child1, child2


def genetic_algorithm_kmeans(data, n_clusters=3, pop_size=30, mutation_rate=0.05, max_stagnant=30):
    # Menggunakan distribusi normal (mean 0, std 1) selaras dengan hasil StandardScaler
    # Menghindarkan penempatan titik centroid awal yang bias di area [0,1]
    population = np.random.normal(0, 1, size=(pop_size, n_clusters, data.shape[1]))
    best_centroids = None
    best_inertia = np.inf
    stagnant = 0
    gen = 0

    while stagnant < max_stagnant:
        scores = [evaluate_scores_kmeans(data, ind, n_clusters) for ind in population]
        ranked = sorted(zip(scores, population), key=lambda x: x[0])  # sort by inertia (lebih kecil lebih baik)
        current_inertia, current_best_centroids = ranked[0]

        if current_inertia < best_inertia:
            best_inertia = current_inertia
            best_centroids = current_best_centroids
            stagnant = 0
        else:
            stagnant += 1

        parents = [x[1] for x in ranked[:pop_size // 2]]
        children = []
        while len(children) < pop_size:
            p1, p2 = random.sample(parents, 2)
            c1, c2 = crossover(p1, p2)
            children += [c1, c2]
        population = np.array(children)

        for i in range(pop_size):
            if random.random() < mutation_rate:
                # Memperbesar kekuatan noise mutasi
                population[i] += np.random.normal(0, 0.5, size=population[i].shape)

        gen += 1

    return best_centroids, best_inertia


def jalankan_kmeans(df, n_clusters=3, save_path="models/model_utama/"):
    os.makedirs(save_path, exist_ok=True)
    
    total_inertia = 0
    
    # 1. Siapkan kolom baseline pada Dataframe Utama
    df['Cluster'] = 0
    for i in range(n_clusters):
        df[f'Distance_to_Centroid_{i}'] = 0.0
    df['Distance_to_Assigned_Centroid'] = 0.0
    
    if 'PROGRAM STUDI' not in df.columns:
        print("⚠️ Kolom PROGRAM STUDI tidak ada! Menjalankan mode Global (Fallback).")
        df_scaled, le_map, scaler = prepare_data(df)
        best_centroids, best_inertia = genetic_algorithm_kmeans(df_scaled, n_clusters=n_clusters)
        total_inertia = best_inertia
        kmeans = KMeans(n_clusters=n_clusters, init=best_centroids, n_init=1).fit(df_scaled)
        df['Cluster'] = kmeans.labels_
        centroid_dist = pairwise_distances(df_scaled, kmeans.cluster_centers_)
        for i in range(n_clusters):
            df[f'Distance_to_Centroid_{i}'] = centroid_dist[:, i]
        df['Distance_to_Assigned_Centroid'] = centroid_dist[np.arange(len(df)), kmeans.labels_]
    else:
        # 2. Mode Clustering Per Prodi
        prodi_groups = df.groupby('PROGRAM STUDI')
        print(f"🔄 Memulai clustering terisolasi untuk {len(prodi_groups)} Program Studi...")
        
        for prodi_name, subset in prodi_groups:
            idx = subset.index
            
            if len(subset) < n_clusters:
                print(f"  -> [Abaikan] {prodi_name}: {len(subset)} baris (<{n_clusters})")
                continue # Biarkan bernilai 0
                
            try:
                df_scaled, _, _ = prepare_data(subset)
            except Exception as e:
                print(f"  -> [Error] {prodi_name}: {e}")
                continue
                
            print(f"  -> Memproses {prodi_name}: {len(subset)} baris. Memutar GA {n_clusters} Kluster...")
            # Populasi dan max_stagnant dikurangi sedikit agar tidak memakan waktu berjam-jam untuk puluhan prodi
            best_centroids, best_inertia = genetic_algorithm_kmeans(df_scaled, n_clusters=n_clusters, pop_size=20, max_stagnant=15)
            total_inertia += best_inertia
            
            # Map KMeans
            kmeans = KMeans(n_clusters=n_clusters, init=best_centroids, n_init=1).fit(df_scaled)
            df.loc[idx, 'Cluster'] = kmeans.labels_
            
            # Distance Logic per Subset Index (loc)
            centroid_dist = pairwise_distances(df_scaled, kmeans.cluster_centers_)
            for i in range(n_clusters):
                df.loc[idx, f'Distance_to_Centroid_{i}'] = centroid_dist[:, i]
            df.loc[idx, 'Distance_to_Assigned_Centroid'] = centroid_dist[np.arange(len(subset)), kmeans.labels_]

    output_path = os.path.join(save_path, f"hasil_kmeans_{n_clusters}_cluster.xlsx")
    df.to_excel(output_path, index=False)

    with open(os.path.join(save_path, f"hasil_kmeans_3cluster.pkl"), "wb") as f:
        pickle.dump({"data": df}, f)

    print("\n✅ KMeans + GA (Per-Prodi) selesai, hasil disimpan:", output_path)

    # Menyingkat keluaran stat agar tidak nge-print baris sangat panjang di konsol
    print("\n🧾 Statistik Singkat KMeans + GA")
    print(f"Total Inertia Gabungan (Semua Prodi): {total_inertia:.4f}")
    if total_inertia > 0:
        print("\nDistribusi Global Cluster (%) :")
        print(df['Cluster'].value_counts(normalize=True).mul(100).round(1))

    # Ekspor summary
    for i in range(n_clusters):
        df_cluster = df[df['Cluster'] == i].copy()
        if df_cluster.empty: continue
        
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
