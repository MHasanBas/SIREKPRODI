import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score
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

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("Tidak ada kolom numerik yang tersedia untuk scaling!")

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(numeric_df)

    return df_scaled, label_encoders, scaler


def evaluate_scores_kmeans(data, centroids, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1)
    kmeans.fit(data)
    labels = kmeans.labels_
    if len(set(labels)) < 2:
        return float('inf'), -1
    dbi = davies_bouldin_score(data, labels)
    sil = silhouette_score(data, labels)
    return dbi, sil


def crossover(parent1, parent2):
    cp = random.randint(1, parent1.shape[1] - 1)
    child1 = np.concatenate((parent1[:, :cp], parent2[:, cp:]), axis=1)
    child2 = np.concatenate((parent2[:, :cp], parent1[:, cp:]), axis=1)
    return child1, child2


def genetic_algorithm_kmeans(data, n_clusters=3, pop_size=30, mutation_rate=0.01, max_stagnant=30):
    population = np.random.rand(pop_size, n_clusters, data.shape[1])
    best_centroids = None
    best_dbi = np.inf
    best_sil = -1
    stagnant = 0
    gen = 0

    while stagnant < max_stagnant:
        scores = [evaluate_scores_kmeans(data, ind, n_clusters) for ind in population]
        ranked = sorted(zip(scores, population), key=lambda x: x[0][0])  # sort by DBI
        (current_dbi, current_sil), current_best_centroids = ranked[0]

        if current_dbi < best_dbi:
            best_dbi = current_dbi
            best_sil = current_sil
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
                population[i] += np.random.normal(0, 0.1, size=population[i].shape)

        print(f"Gen {gen + 1}: DBI = {best_dbi:.4f} | Silhouette = {best_sil:.4f}")
        gen += 1

    return best_centroids, best_dbi, best_sil


def jalankan_kmeans(df, n_clusters=3, save_path="models/model_utama/"):
    from clustering.preprocessing import prepare_data  # pastikan ada
    
    df_scaled, le_map, scaler = prepare_data(df)
    best_centroids, best_dbi, best_sil = genetic_algorithm_kmeans(df_scaled, n_clusters=n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, init=best_centroids, n_init=1).fit(df_scaled)
    df['Cluster'] = kmeans.labels_

    centroid_dist = pairwise_distances(df_scaled, kmeans.cluster_centers_)
    for i in range(n_clusters):
        df[f'Distance_to_Centroid_{i}'] = centroid_dist[:, i]
    df['Distance_to_Assigned_Centroid'] = centroid_dist[np.arange(len(df)), kmeans.labels_]

    # ✅ Ganti path output ke save_path
    os.makedirs(save_path, exist_ok=True)
    output_path = os.path.join(save_path, f"hasil_kmeans_{n_clusters}_cluster.xlsx")
    df['DBI_SCORE'] = best_dbi
    df['SILHOUETTE_SCORE'] = best_sil
    df.to_excel(output_path, index=False)

    # ✅ Simpan versi .pkl juga (untuk dashboard)
    with open(os.path.join(save_path, f"hasil_kmeans_3cluster.pkl"), "wb") as f:
        pickle.dump({"data": df}, f)  # wajib dict dengan key 'data'

    print("✅ KMeans + GA selesai, hasil disimpan:", output_path)

    # Statistik cetakan kamu tetap dipertahankan
    print("\n🧾 Statistik Lengkap untuk KMeans + GA")
    print(f"Total DBI: {best_dbi:.4f}")
    print(f"Total Silhouette Score: {best_sil:.4f}\n")

    for i in range(n_clusters):
        cluster_data = df[df['Cluster'] == i]
        print(f"Cluster {i}:")
        print(f"  - Jumlah titik: {len(cluster_data)}")
        print(f"  - Rata-rata jarak ke centroid (Kepadatan): {cluster_data[f'Distance_to_Centroid_{i}'].mean():.2f}")
        pemisahan = pairwise_distances(kmeans.cluster_centers_).mean()
        print(f"  - Rata-rata jarak antar centroid (Pemisahan): {pemisahan:.2f}")
        print(f"  - Centroid: {kmeans.cluster_centers_[i]}")

    print("\nPersentase Jumlah Data di Masing-Masing Cluster:")
    print(df['Cluster'].value_counts(normalize=True).mul(100).round(1))

    for col in ['ASAL SEKOLAH', 'KOTA SEKOLAH', 'PROVINSI SEKOLAH', 'NILAI KESELURUHAN']:
        if col in df.columns:
            print(f"\nDominasi untuk kolom {col}:")
            print(df.groupby('Cluster')[col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A'))

    print("\nRata-Rata 'NILAI KESELURUHAN' di Setiap Cluster:")
    print(df.groupby('Cluster')['NILAI KESELURUHAN'].mean())

    print("\nStatistik Deskriptif 'NILAI KESELURUHAN':")
    print(df.groupby('Cluster')['NILAI KESELURUHAN'].agg(['mean', 'median', lambda x: x.mode().iloc[0], 'min', 'max']).rename(columns={'<lambda_0>': 'mode'}))

    # Ekspor detail per cluster
    for i in range(n_clusters):
        df_cluster = df[df['Cluster'] == i].copy()
        df_cluster_group = df_cluster.groupby(['ASAL SEKOLAH', 'KOTA SEKOLAH', 'PROVINSI SEKOLAH']).agg(
            Cluster=('Cluster', 'first'),
            **{
                'Jumlah Mahasiswa': ('NILAI KESELURUHAN', 'count'),
                'Nilai Mean': ('NILAI KESELURUHAN', 'mean'),
                'Nilai Median': ('NILAI KESELURUHAN', 'median'),
                'Nilai Modus': ('NILAI KESELURUHAN', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
                'Nilai Min': ('NILAI KESELURUHAN', 'min'),
                'Nilai Max': ('NILAI KESELURUHAN', 'max'),
                'Nilai Range': ('NILAI KESELURUHAN', lambda x: x.max() - x.min()),
                'Deviasi Standar': ('NILAI KESELURUHAN', 'std'),
            }
        ).reset_index()

        cluster_dir = os.path.join(save_path, f'cluster_{i}')
        os.makedirs(cluster_dir, exist_ok=True)
        df_cluster_group.to_excel(os.path.join(cluster_dir, f'detail_cluster_{i}.xlsx'), index=False)
