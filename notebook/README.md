# SIREKPRODI — Clustering K-Means + Genetic Algorithm

Notebook untuk memproses data mahasiswa menggunakan K-Means yang dioptimasi dengan Genetic Algorithm (GA), menghasilkan kluster dan rekomendasi sekolah per Program Studi.

---

## Persyaratan

```bash
pip install pandas numpy scikit-learn openpyxl matplotlib seaborn
```

Atau jalankan langsung di **Google Colab** (upload file `.ipynb` ke [colab.research.google.com](https://colab.research.google.com)).

---

## Format Dataset

File `.xlsx` dengan kolom berikut:

| Kolom | Wajib | Keterangan |
|---|:---:|---|
| `ASAL SEKOLAH` | ✅ | Nama sekolah asal |
| `NPSN SEKOLAH` | ✅ | Nomor Pokok Sekolah Nasional |
| `KOTA SEKOLAH` | ✅ | Kota sekolah |
| `PROVINSI SEKOLAH` | ✅ | Provinsi sekolah |
| `PROGRAM STUDI` | ✅ | Program Studi mahasiswa |
| `IPK` | ✅ | IPK mahasiswa (0–4) |
| `STATUS` | ✅ | Status mahasiswa (`AKTIF` / lainnya) |
| `PRESTASI 1`, `PRESTASI 2` | ⬜ | Deskripsi prestasi (opsional) |

---

## Alur Notebook

| Sel | Proses |
|---|---|
| [1] | Instalasi library |
| [2] | Import & konfigurasi seed |
| [3] | Muat dataset `.xlsx` |
| [4] | Preprocessing (filter Status, validasi IPK) |
| [5] | Eksplorasi data — distribusi IPK & top prodi |
| [6] | Fungsi `prepare_data` & `genetic_algorithm_kmeans` |
| [7] | Clustering K-Means + GA per Program Studi |
| [8] | Visualisasi distribusi cluster & IPK rata-rata |
| [9] | Pelabelan A / B / C per Prodi |
| [10] | Top 5 sekolah per label cluster |
| [11] | Ekspor hasil ke `hasil_clustering_SIREKPRODI.xlsx` |
| [12] | *(Opsional)* Simpan model `.pkl` untuk aplikasi web |

---

## Output

- **`hasil_clustering_SIREKPRODI.xlsx`** — data lengkap dengan kolom tambahan:
  - `Cluster` — ID kluster (0/1/2), bersifat lokal per prodi
  - `Label` — Peringkat A/B/C (A = terbaik di prodinya)
  - `Distance_to_Centroid_*` — jarak tiap data ke masing-masing centroid

- **`models/model_notebook/hasil_kmeans_3cluster.pkl`** *(opsional)* — file model yang bisa diaktifkan di halaman **Pelatihan Model** pada aplikasi web.

---

## Catatan Penting

- **Label bersifat relatif per prodi.** Cluster A di Prodi Teknik Informatika ≠ Cluster A di Prodi Akuntansi.
- **Ubah `FILE_PATH`** di sel [3] sesuai lokasi file dataset Anda.
- Prodi dengan jumlah mahasiswa < 3 akan dilewati secara otomatis.
