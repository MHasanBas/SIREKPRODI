import pandas as pd
from utils.helpers import _norm_series, _norm_in_group

def combine_t(x):
    strs = []
    for v in x:
        if v:
            for s in str(v).split(' | '):
                s = s.strip()
                if s and s not in strs:
                    strs.append(s)
    res = " • ".join(strs)
    return res[:400] + ("..." if len(res) > 400 else "") or "-"

def process_dashboard_data(df_kmeans, prodi_column, tampilkan='top10'):
    rekom_prodi_per_sekolah = []
    rekom_sekolah_per_prodi = []

    if prodi_column:
        # Sekolah unggulan berdasarkan ipk/mahasiswa/deviasi
        sekolah_stats = df_kmeans.groupby('ASAL SEKOLAH', as_index=False).agg(
            mahasiswa=('NILAI KESELURUHAN', 'count'),
            ipk=('NILAI KESELURUHAN', 'mean'),
            deviasi=('NILAI KESELURUHAN', 'std'),
            poin_akademik=('POIN_AKADEMIK', 'sum'),
            poin_non_akademik=('POIN_NON_AKADEMIK', 'sum')
        ).fillna(0)
        sekolah_stats['skor'] = (
            0.35 * _norm_series(sekolah_stats['mahasiswa']) +
            0.35 * _norm_series(sekolah_stats['ipk']) +
            0.15 * (1 - _norm_series(sekolah_stats['deviasi'])) +
            0.10 * _norm_series(sekolah_stats['poin_akademik']) +
            0.05 * _norm_series(sekolah_stats['poin_non_akademik'])
        )

        info_sekolah = df_kmeans.groupby('ASAL SEKOLAH').agg(
            kota=('KOTA SEKOLAH', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
            provinsi=('PROVINSI SEKOLAH', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        )

        sekolah_unggulan = sekolah_stats.sort_values('skor', ascending=False)

        # Top prodi per sekolah (tanpa filter berulang per sekolah)
        prodi_agg = df_kmeans.groupby(['ASAL SEKOLAH', prodi_column], as_index=False).agg(
            mahasiswa=('NILAI KESELURUHAN', 'count'),
            ipk=('NILAI KESELURUHAN', 'mean'),
            deviasi=('NILAI KESELURUHAN', 'std'),
            poin_akademik=('POIN_AKADEMIK', 'sum'),
            poin_non_akademik=('POIN_NON_AKADEMIK', 'sum')
        ).fillna(0).rename(columns={prodi_column: 'prodi'})

        prodi_agg['mhs_norm'] = _norm_in_group(prodi_agg, 'ASAL SEKOLAH', 'mahasiswa')
        prodi_agg['ipk_norm'] = _norm_in_group(prodi_agg, 'ASAL SEKOLAH', 'ipk')
        prodi_agg['dev_norm'] = _norm_in_group(prodi_agg, 'ASAL SEKOLAH', 'deviasi')
        prodi_agg['p_akd_norm'] = _norm_in_group(prodi_agg, 'ASAL SEKOLAH', 'poin_akademik')
        prodi_agg['p_non_norm'] = _norm_in_group(prodi_agg, 'ASAL SEKOLAH', 'poin_non_akademik')

        prodi_agg['skor'] = (
            0.35 * prodi_agg['mhs_norm'] +
            0.35 * prodi_agg['ipk_norm'] +
            0.15 * (1 - prodi_agg['dev_norm']) +
            0.10 * prodi_agg['p_akd_norm'] +
            0.05 * prodi_agg['p_non_norm']
        ).round(3)

        top3_prodi = (
            prodi_agg.sort_values(['ASAL SEKOLAH', 'skor'], ascending=[True, False])
            .groupby('ASAL SEKOLAH', as_index=False)
            .head(3)
        )
        top3_map = {
            sekolah: grp[['prodi', 'ipk', 'mahasiswa', 'skor']].to_dict(orient='records')
            for sekolah, grp in top3_prodi.groupby('ASAL SEKOLAH')
        }

        for _, row in sekolah_unggulan.iterrows():
            nama_sekolah = row['ASAL SEKOLAH']
            info = info_sekolah.loc[nama_sekolah]
            rekom_prodi_per_sekolah.append({
                "sekolah": nama_sekolah,
                "kota": info['kota'],
                "provinsi": info['provinsi'],
                "ipk_mean": round(row['ipk'], 2),
                "mahasiswa": int(row['mahasiswa']),
                "top_prodi": top3_map.get(nama_sekolah, [])
            })

        # Sekolah sasaran per prodi
        sekolah_prodi_agg = df_kmeans.groupby([prodi_column, 'ASAL SEKOLAH'], as_index=False).agg(
            mahasiswa=('NILAI KESELURUHAN', 'count'),
            ipk=('NILAI KESELURUHAN', 'mean'),
            deviasi=('NILAI KESELURUHAN', 'std'),
            poin_akademik=('POIN_AKADEMIK', 'sum'),
            poin_non_akademik=('POIN_NON_AKADEMIK', 'sum'),
            kota=('KOTA SEKOLAH', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
            provinsi=('PROVINSI SEKOLAH', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        ).fillna(0).rename(columns={prodi_column: 'prodi', 'ASAL SEKOLAH': 'sekolah'})

        sekolah_prodi_agg['mhs_norm'] = _norm_in_group(sekolah_prodi_agg, 'prodi', 'mahasiswa')
        sekolah_prodi_agg['ipk_norm'] = _norm_in_group(sekolah_prodi_agg, 'prodi', 'ipk')
        sekolah_prodi_agg['dev_norm'] = _norm_in_group(sekolah_prodi_agg, 'prodi', 'deviasi')
        sekolah_prodi_agg['p_akd_norm'] = _norm_in_group(sekolah_prodi_agg, 'prodi', 'poin_akademik')
        sekolah_prodi_agg['p_non_norm'] = _norm_in_group(sekolah_prodi_agg, 'prodi', 'poin_non_akademik')
        sekolah_prodi_agg['skor'] = (
            0.35 * sekolah_prodi_agg['mhs_norm'] +
            0.35 * sekolah_prodi_agg['ipk_norm'] +
            0.15 * (1 - sekolah_prodi_agg['dev_norm']) +
            0.10 * sekolah_prodi_agg['p_akd_norm'] +
            0.05 * sekolah_prodi_agg['p_non_norm']
        ).round(3)

        top5 = (
            sekolah_prodi_agg.sort_values(['prodi', 'skor'], ascending=[True, False])
            .groupby('prodi', as_index=False)
            .head(5)
        )
        top5_map = {
            prodi: grp[['sekolah', 'kota', 'provinsi', 'mahasiswa', 'ipk', 'skor']].to_dict(orient='records')
            for prodi, grp in top5.groupby('prodi')
        }

        prodi_overall = df_kmeans.groupby(prodi_column, as_index=False).agg(
            total_mahasiswa=('NILAI KESELURUHAN', 'count'),
            ipk_mean=('NILAI KESELURUHAN', 'mean')
        ).rename(columns={prodi_column: 'prodi'})
        overall_map = {
            row['prodi']: (int(row['total_mahasiswa']), round(row['ipk_mean'], 2))
            for _, row in prodi_overall.iterrows()
        }

        for prodi in sorted(df_kmeans[prodi_column].dropna().unique()):
            total_mhs, ipk_mean = overall_map.get(prodi, (0, 0))
            rekom_sekolah_per_prodi.append({
                "prodi": prodi,
                "top_sekolah": top5_map.get(prodi, []),
                "total_mahasiswa": total_mhs,
                "ipk_mean": ipk_mean
            })

    def siapkan_data(df, nama_algo, tampilkan='top10'):
        # Satu baris = satu kombinasi sekolah + lokasi (nama sama di kota beda ≠ satu entitas)
        kunci_sekolah = ['ASAL SEKOLAH', 'KOTA SEKOLAH', 'PROVINSI SEKOLAH']

        df_dev = df.groupby(['Cluster'] + kunci_sekolah).agg(
            mahasiswa=('NILAI KESELURUHAN', 'count'),
            ipk=('NILAI KESELURUHAN', 'mean'),
            deviasi=('NILAI KESELURUHAN', 'std'),
            poin_akademik=('POIN_AKADEMIK', 'sum'),
            poin_non_akademik=('POIN_NON_AKADEMIK', 'sum'),
            teks_akademik=('TEKS_AKADEMIK', combine_t),
            teks_non_akademik=('TEKS_NON_AKADEMIK', combine_t)
        ).reset_index()

        cluster_summary = df.groupby('Cluster').agg(
            total_mahasiswa=('NILAI KESELURUHAN', 'count'),
            ipk_rata2=('NILAI KESELURUHAN', 'mean'),
            deviasi=('NILAI KESELURUHAN', 'std')
        ).reset_index()

        cluster_summary['mahasiswa_norm'] = (cluster_summary['total_mahasiswa'] - cluster_summary['total_mahasiswa'].min()) / (cluster_summary['total_mahasiswa'].max() - cluster_summary['total_mahasiswa'].min())
        cluster_summary['ipk_norm'] = (cluster_summary['ipk_rata2'] - cluster_summary['ipk_rata2'].min()) / (cluster_summary['ipk_rata2'].max() - cluster_summary['ipk_rata2'].min())
        cluster_summary['deviasi_norm'] = (cluster_summary['deviasi'] - cluster_summary['deviasi'].min()) / (cluster_summary['deviasi'].max() - cluster_summary['deviasi'].min())
        cluster_summary['ranking'] = (
            0.4 * cluster_summary['mahasiswa_norm'] +
            0.3 * cluster_summary['ipk_norm'] +
            0.3 * (1 - cluster_summary['deviasi_norm'])
        )

        ordered_clusters = cluster_summary.sort_values(by='ranking', ascending=False)['Cluster'].tolist()
        label_cluster = ['A', 'B', 'C']
        if ordered_clusters:
            ordered_clusters = ordered_clusters + [ordered_clusters[-1]] * (len(label_cluster) - len(ordered_clusters))
        result = {'algoritma': nama_algo}
        
        summary_dict = {}
        for idx, label in enumerate(label_cluster):
            cluster_id = ordered_clusters[idx]
            cluster_data = cluster_summary[cluster_summary['Cluster'] == cluster_id].iloc[0]
            summary_dict[label] = {
                'ipk': cluster_data['ipk_rata2'],
                'mahasiswa': cluster_data['total_mahasiswa'],
                'deviasi': cluster_data['deviasi']
            }
        
        def bandingkan(c1, c2, key):
            return "lebih tinggi" if c1[key] > c2[key] else "lebih rendah" if c1[key] < c2[key] else "sama"

        deskripsi_cluster = {}
        deskripsi_cluster['A'] = (
            f"Jumlah mahasiswa dari tiap sekolah dengan tingkat keberlanjutan studi di Polinema {bandingkan(summary_dict['A'], summary_dict['B'], 'mahasiswa')} dari cluster B "
            f"dan {bandingkan(summary_dict['A'], summary_dict['C'], 'mahasiswa')} dari cluster C. "
            f"IPK rata-rata {bandingkan(summary_dict['A'], summary_dict['B'], 'ipk')} dari cluster B "
            f"dan {bandingkan(summary_dict['A'], summary_dict['C'], 'ipk')} dari cluster C. "
            f"Standar deviasi {bandingkan(summary_dict['A'], summary_dict['B'], 'deviasi')} dari cluster B "
            f"dan {bandingkan(summary_dict['A'], summary_dict['C'], 'deviasi')} dari cluster C."
        )

        deskripsi_cluster['B'] = (
            f"Jumlah mahasiswa dari tiap sekolah dengan tingkat keberlanjutan studi di Polinema {bandingkan(summary_dict['B'], summary_dict['A'], 'mahasiswa')} dari cluster A "
            f"dan {bandingkan(summary_dict['B'], summary_dict['C'], 'mahasiswa')} dari cluster C. "
            f"IPK rata-rata {bandingkan(summary_dict['B'], summary_dict['A'], 'ipk')} dari cluster A "
            f"dan {bandingkan(summary_dict['B'], summary_dict['C'], 'ipk')} dari cluster C. "
            f"Standar deviasi {bandingkan(summary_dict['B'], summary_dict['A'], 'deviasi')} dari cluster A "
            f"dan {bandingkan(summary_dict['B'], summary_dict['C'], 'deviasi')} dari cluster C."
        )

        deskripsi_cluster['C'] = (
            f"Jumlah mahasiswa dari tiap sekolah dengan tingkat keberlanjutan studi di Polinema {bandingkan(summary_dict['C'], summary_dict['A'], 'mahasiswa')} dari cluster A "
            f"dan {bandingkan(summary_dict['C'], summary_dict['B'], 'mahasiswa')} dari cluster B. "
            f"IPK rata-rata {bandingkan(summary_dict['C'], summary_dict['A'], 'ipk')} dari cluster A "
            f"dan {bandingkan(summary_dict['C'], summary_dict['B'], 'ipk')} dari cluster B. "
            f"Standar deviasi {bandingkan(summary_dict['C'], summary_dict['A'], 'deviasi')} dari cluster A "
            f"dan {bandingkan(summary_dict['C'], summary_dict['B'], 'deviasi')} dari cluster B."
        )

        result['deskripsi_cluster'] = deskripsi_cluster

        for idx, label in enumerate(label_cluster):
            cluster_id = ordered_clusters[idx]
            df_final = df[df['Cluster'] == cluster_id].copy()
            df_sub = df_dev[df_dev['Cluster'] == cluster_id].copy()

            mhs_norm = _norm_series(df_sub['mahasiswa'])
            ipk_norm = _norm_series(df_sub['ipk'])
            dev_norm = _norm_series(df_sub['deviasi'])
            p_akd_norm = _norm_series(df_sub['poin_akademik'])
            p_non_norm = _norm_series(df_sub['poin_non_akademik'])

            df_sub['skor'] = (
                0.35 * mhs_norm +
                0.35 * ipk_norm +
                0.15 * (1 - dev_norm) +
                0.10 * p_akd_norm +
                0.05 * p_non_norm
            ).round(3)


            df_sorted = df_sub.sort_values(by='skor', ascending=False)
            if tampilkan == 'semua':
                tabel_rekomendasi = df_sorted[
                    ['ASAL SEKOLAH', 'KOTA SEKOLAH', 'PROVINSI SEKOLAH', 'mahasiswa', 'ipk', 'deviasi', 'poin_akademik', 'poin_non_akademik', 'teks_akademik', 'teks_non_akademik']
                ].to_dict(orient='records')
            else:
                tabel_rekomendasi = df_sorted.head(10)[
                    ['ASAL SEKOLAH', 'KOTA SEKOLAH', 'PROVINSI SEKOLAH', 'mahasiswa', 'ipk', 'deviasi', 'poin_akademik', 'poin_non_akademik', 'teks_akademik', 'teks_non_akademik']
                ].to_dict(orient='records')


            bins = [2.0, 2.5, 3.0, 3.5, 4.0]
            labels = ['2.0-2.5', '2.5-3.0', '3.0-3.5', '3.5-4.0']
            df_final.loc[:, 'ipk_bin'] = pd.cut(df_final['NILAI KESELURUHAN'], bins=bins, labels=labels, include_lowest=True)
            distribusi_raw = df_final['ipk_bin'].value_counts().sort_index().reset_index(name='count')
            distribusi_raw.columns = ['jumlah', 'count']
            distribusi_ipk = [
                {"jumlah": str(row['jumlah']), "count": int(row['count']) if pd.notnull(row['count']) else 0}
                for _, row in distribusi_raw.iterrows()
            ]

            line_top10 = df_final['KOTA SEKOLAH'].value_counts().head(10)
            line_chart = [{"kota": k, "jumlah": int(v)} for k, v in line_top10.items()]

            top_row = df_sub.sort_values(by='skor', ascending=False).iloc[0]
            result[label] = {
                'total_mahasiswa': int(df_final.shape[0]),
                'ipk_mean': round(df_final['NILAI KESELURUHAN'].mean(), 2),
                'kota_terbaik': top_row['KOTA SEKOLAH'],
                'sekolah_terbaik': top_row['ASAL SEKOLAH'],
                'tabel_rekomendasi': tabel_rekomendasi,
                'distribusi_ipk': distribusi_ipk,
                'mahasiswa_per_kota_top10': line_chart,
                'deskripsi': deskripsi_cluster[label]
            }

        return result

    kmeans_data = siapkan_data(df_kmeans, "K-Means", tampilkan)

    return rekom_prodi_per_sekolah, rekom_sekolah_per_prodi, kmeans_data
