import pandas as pd
import numpy as np
import os
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score
from scipy.spatial.distance import cdist
from deap import base, creator, tools
import pickle

# -------------------- PREPROCESSING --------------------
def prepare_data(df):
    random.seed(42)
    np.random.seed(42)
    df = df.copy()
    if df.empty:
        raise ValueError("DataFrame kosong! Tidak bisa diproses.")

    categorical_cols = ['ASAL SEKOLAH', 'KOTA SEKOLAH', 'PROVINSI SEKOLAH']
    optional_categorical = ['PROGRAM STUDI']
    label_encoders = {}
    for col in categorical_cols + optional_categorical:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else 'UNKNOWN')
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        elif col in categorical_cols:
            raise ValueError(f"Kolom kategorikal '{col}' tidak ditemukan di DataFrame!")

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numerical_cols:
        raise ValueError("Tidak ada kolom numerik yang tersedia untuk scaling!")

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[numerical_cols])
    return df_scaled, label_encoders, scaler

# -------------------- FCM CORE --------------------
def fuzzy_c_means_manual(data, n_clusters, m=2, max_iter=100, error=1e-5):
    n_samples = data.shape[0]
    n_features = data.shape[1]
    u = np.random.rand(n_clusters, n_samples)
    u = u / np.sum(u, axis=0)
    centroids = None

    for iteration in range(max_iter):
        um = u ** m
        try:
            centroids = (um @ data) / np.sum(um, axis=1, keepdims=True)
        except ZeroDivisionError:
            print("❌ Error: pembagian nol saat hitung centroid")
            break

        dist = np.zeros((n_clusters, n_samples))
        for i in range(n_clusters):
            dist[i] = np.linalg.norm(data - centroids[i], axis=1)
        dist = np.fmax(dist, np.finfo(np.float64).eps)

        new_u = np.zeros_like(u)
        for i in range(n_clusters):
            for j in range(n_samples):
                try:
                    new_u[i, j] = 1.0 / np.sum((dist[i, j] / dist[:, j]) ** (2 / (m - 1)))
                except ZeroDivisionError:
                    new_u[i, j] = 0.0

        if np.linalg.norm(new_u - u) < error:
            u = new_u
            break
        u = new_u

    if centroids is None:
        raise ValueError("❌ Gagal membentuk centroid — kemungkinan data tidak valid (semua sama atau kosong).")

    labels = np.argmax(u, axis=0)
    return centroids, u, labels

# -------------------- EVALUASI DBI & SIL --------------------
def evaluate_scores_fcm(data, u, centroids):
    labels = np.argmax(u, axis=0)
    if len(set(labels)) < 2:
        return float('inf'), -1
    dbi = davies_bouldin_score(data, labels)
    sil = silhouette_score(data, labels)
    return dbi, sil

# -------------------- GENETIC ALGORITHM OPTIMIZATION --------------------
def genetic_algorithm_fcm(data, n_clusters=3, pop_size=30, max_stagnant=30, m=2):
    n_features = data.shape[1]

    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    data_min, data_max = data.min(axis=0), data.max(axis=0)

    def init_centroids():
        return [random.uniform(data_min[i % n_features], data_max[i % n_features]) for i in range(n_clusters * n_features)]

    def evaluate_centroids(individual):
        centroids = np.array(individual).reshape((n_clusters, n_features))
        dist = np.zeros((n_clusters, data.shape[0]))
        for i in range(n_clusters):
            dist[i] = np.linalg.norm(data - centroids[i], axis=1)
        dist = np.fmax(dist, np.finfo(np.float64).eps)

        u = np.zeros((n_clusters, data.shape[0]))
        for i in range(n_clusters):
            for j in range(data.shape[0]):
                u[i, j] = 1.0 / np.sum((dist[i, j] / dist[:, j]) ** (2 / (m - 1)))

        labels = np.argmax(u, axis=0)
        if len(set(labels)) < 2:
            return (float('inf'),)
        dbi = davies_bouldin_score(data, labels)
        return (dbi,)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, init_centroids)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_centroids)

    population = toolbox.population(n=pop_size)
    stagnant = 0
    best = None
    best_dbi = float('inf')
    best_sil = -1
    gen = 0

    while stagnant < max_stagnant:
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        for m_ind in offspring:
            if random.random() < 0.2:
                toolbox.mutate(m_ind)
                del m_ind.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring
        current_best = tools.selBest(population, 1)[0]

        centroids = np.array(current_best).reshape((n_clusters, n_features))
        dist = np.zeros((n_clusters, data.shape[0]))
        for i in range(n_clusters):
            dist[i] = np.linalg.norm(data - centroids[i], axis=1)
        dist = np.fmax(dist, np.finfo(np.float64).eps)

        u = np.zeros((n_clusters, data.shape[0]))
        for i in range(n_clusters):
            for j in range(data.shape[0]):
                u[i, j] = 1.0 / np.sum((dist[i, j] / dist[:, j]) ** (2 / (m - 1)))

        dbi, sil = evaluate_scores_fcm(data, u, centroids)

        if dbi < best_dbi:
            best = current_best
            best_dbi = dbi
            best_sil = sil
            stagnant = 0
        else:
            stagnant += 1

        print(f"Gen {gen+1} | DBI: {dbi:.4f} | Silhouette: {sil:.4f}")
        gen += 1

    best_centroids = np.array(best).reshape((n_clusters, n_features))
    return best_centroids, best_dbi, best_sil, m

# Jalankan full proses FCM GA

def jalankan_fcm(df, n_clusters=3, save_path="models/model_utama/"):
    os.makedirs(save_path, exist_ok=True)
    df_scaled, le_map, scaler = prepare_data(df)
    cntr, best_dbi, best_sil, m = genetic_algorithm_fcm(df_scaled, n_clusters=n_clusters)
    centroids, u, labels = fuzzy_c_means_manual(df_scaled, n_clusters, m=m)
    df['Cluster'] = labels
    for i in range(n_clusters):
        df[f'Membership_{i}'] = u[i]
    dist_to_centroids = cdist(df_scaled, cntr)
    for i in range(n_clusters):
        df[f'Distance_to_Centroid_{i}'] = dist_to_centroids[:, i]
    df['Distance_to_Assigned_Centroid'] = dist_to_centroids[np.arange(len(df_scaled)), labels]
    with open(os.path.join(save_path, f"hasil_fcm_3cluster.pkl"), "wb") as f:
        pickle.dump({"data": df}, f)
    df.to_excel(os.path.join(save_path, "hasil_fcm.xlsx"), index=False)
    print("\n✅ FCM selesai, hasil disimpan:", save_path)
    print("\n📊 Total DBI:", round(best_dbi, 4))
    print("📊 Total Silhouette Score:", round(best_sil, 4))
    for i in range(n_clusters):
        cluster_data = df[df['Cluster'] == i]
        print(f"\nCluster {i}:")
        print(f"  - Jumlah titik: {len(cluster_data)}")
        print(f"  - Rata-rata jarak ke centroid (Kepadatan): {cluster_data[f'Distance_to_Centroid_{i}'].mean():.2f}")
        print(f"  - Centroid: {cntr[i]}")
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
