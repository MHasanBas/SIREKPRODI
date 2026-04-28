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

    if 'Cluster' in df_kmeans.columns:
        c_sum = df_kmeans.groupby('Cluster').agg(
            total_mhs=('NILAI KESELURUHAN', 'count'),
            ipk_mean=('NILAI KESELURUHAN', 'mean'),
            dev=('NILAI KESELURUHAN', 'std')
        ).fillna(0).reset_index()

        def _safe_norm(s):
            denom = s.max() - s.min()
            return s * 0 if denom == 0 else (s - s.min()) / denom

        c_sum['ranking'] = (
            0.4 * _safe_norm(c_sum['total_mhs']) +
            0.3 * _safe_norm(c_sum['ipk_mean']) +
            0.3 * (1 - _safe_norm(c_sum['dev']))
        )
        ordered_clusters = c_sum.sort_values(by='ranking', ascending=False)['Cluster'].tolist()
        superior_clusters = ordered_clusters[:2] if len(ordered_clusters) >= 2 else ordered_clusters
        df_kmeans['is_superior'] = df_kmeans['Cluster'].isin(superior_clusters)
    else:
        df_kmeans['is_superior'] = True

    if prodi_column:
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
        
        school_superior_ratio = df_kmeans.groupby('ASAL SEKOLAH')['is_superior'].mean()
        valid_schools = school_superior_ratio[school_superior_ratio >= 0.50].index
        sekolah_unggulan = sekolah_unggulan[sekolah_unggulan['ASAL SEKOLAH'].isin(valid_schools)]
        
        sekolah_unggulan = sekolah_unggulan[sekolah_unggulan['mahasiswa'] >= 3]

        prodi_agg = df_kmeans.groupby(['ASAL SEKOLAH', prodi_column], as_index=False).agg(
            mahasiswa=('NILAI KESELURUHAN', 'count'),
            ipk=('NILAI KESELURUHAN', 'mean'),
            deviasi=('NILAI KESELURUHAN', 'std'),
            poin_akademik=('POIN_AKADEMIK', 'sum'),
            poin_non_akademik=('POIN_NON_AKADEMIK', 'sum')
        ).fillna(0).rename(columns={prodi_column: 'prodi'})
        
        prodi_agg = prodi_agg[prodi_agg['mahasiswa'] >= 3]

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

        max_prodi_per_kartu = 50
        prodi_per_sekolah_sorted = prodi_agg.sort_values(
            ['ASAL SEKOLAH', 'skor'], ascending=[True, False]
        )
        all_prodi_map = {}
        for sekolah, grp in prodi_per_sekolah_sorted.groupby('ASAL SEKOLAH'):
            slice_df = grp.head(max_prodi_per_kartu)
            all_prodi_map[sekolah] = slice_df[
                ['prodi', 'ipk', 'mahasiswa', 'skor', 'poin_akademik', 'poin_non_akademik']
            ].to_dict(orient='records')

        jumlah_prodi_per_sekolah = prodi_per_sekolah_sorted.groupby('ASAL SEKOLAH').size().to_dict()

        for _, row in sekolah_unggulan.iterrows():
            nama_sekolah = row['ASAL SEKOLAH']
            info = info_sekolah.loc[nama_sekolah]
            tp = all_prodi_map.get(nama_sekolah, [])
            rekom_prodi_per_sekolah.append({
                "sekolah": nama_sekolah,
                "kota": info['kota'],
                "provinsi": info['provinsi'],
                "ipk_mean": round(row['ipk'], 2),
                "mahasiswa": int(row['mahasiswa']),
                "poin_akademik": int(row['poin_akademik']),
                "poin_non_akademik": int(row['poin_non_akademik']),
                "top_prodi": tp,
                "prodi_total_asal": int(jumlah_prodi_per_sekolah.get(nama_sekolah, 0)),
            })

        sekolah_prodi_agg = df_kmeans.groupby([prodi_column, 'ASAL SEKOLAH'], as_index=False).agg(
            mahasiswa=('NILAI KESELURUHAN', 'count'),
            ipk=('NILAI KESELURUHAN', 'mean'),
            deviasi=('NILAI KESELURUHAN', 'std'),
            poin_akademik=('POIN_AKADEMIK', 'sum'),
            poin_non_akademik=('POIN_NON_AKADEMIK', 'sum'),
            kota=('KOTA SEKOLAH', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
            provinsi=('PROVINSI SEKOLAH', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        ).fillna(0).rename(columns={prodi_column: 'prodi', 'ASAL SEKOLAH': 'sekolah'})

        prodi_school_sup_ratio = df_kmeans.groupby([prodi_column, 'ASAL SEKOLAH'])['is_superior'].mean().reset_index(name='sup_ratio')
        valid_prodi_school = prodi_school_sup_ratio[prodi_school_sup_ratio['sup_ratio'] >= 0.50]
        sekolah_prodi_agg = pd.merge(sekolah_prodi_agg, valid_prodi_school[[prodi_column, 'ASAL SEKOLAH']], 
                                     left_on=['prodi', 'sekolah'], right_on=[prodi_column, 'ASAL SEKOLAH'], how='inner')

        sekolah_prodi_agg = sekolah_prodi_agg[sekolah_prodi_agg['mahasiswa'] >= 3]

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

        sekolah_per_prodi_default = 20
        sekolah_per_prodi_df = (
            sekolah_prodi_agg.sort_values(['prodi', 'skor'], ascending=[True, False])
        )

        top5 = sekolah_per_prodi_df.groupby('prodi', as_index=False).head(5)
        topN = sekolah_per_prodi_df.groupby('prodi', as_index=False).head(sekolah_per_prodi_default)

        top5_map = {
            prodi: grp[['sekolah', 'kota', 'provinsi', 'mahasiswa', 'ipk', 'skor', 'poin_akademik', 'poin_non_akademik']].to_dict(orient='records')
            for prodi, grp in top5.groupby('prodi')
        }
        topN_map = {
            prodi: grp[['sekolah', 'kota', 'provinsi', 'mahasiswa', 'ipk', 'skor', 'poin_akademik', 'poin_non_akademik']].to_dict(orient='records')
            for prodi, grp in topN.groupby('prodi')
        }

        jumlah_sekolah_per_prodi = sekolah_per_prodi_df.groupby('prodi').size().to_dict()

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
            df_p = df_kmeans[df_kmeans[prodi_column] == prodi]
            p_ak = int(df_p["POIN_AKADEMIK"].sum()) if "POIN_AKADEMIK" in df_p.columns else 0
            p_na = int(df_p["POIN_NON_AKADEMIK"].sum()) if "POIN_NON_AKADEMIK" in df_p.columns else 0
            ts5 = top5_map.get(prodi, [])
            tsN = topN_map.get(prodi, [])
            poin_ak_top5 = int(sum(int(float(s.get("poin_akademik", 0) or 0)) for s in ts5))
            poin_na_top5 = int(sum(int(float(s.get("poin_non_akademik", 0) or 0)) for s in ts5))
            rekom_sekolah_per_prodi.append({
                "prodi": prodi,
                "top_sekolah": tsN,
                "total_mahasiswa": total_mhs,
                "ipk_mean": ipk_mean,
                "poin_akademik": p_ak,
                "poin_non_akademik": p_na,
                "poin_akademik_top5": poin_ak_top5,
                "poin_non_akademik_top5": poin_na_top5,
                "sekolah_total_asal": int(jumlah_sekolah_per_prodi.get(prodi, 0)),
                "sekolah_default_limit": int(sekolah_per_prodi_default),
            })

    def siapkan_data(df, nama_algo, tampilkan='top10'):
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

def get_crosscheck_data(df_kmeans):
    if df_kmeans.empty or 'ASAL SEKOLAH' not in df_kmeans.columns:
        return []

    df_cross = df_kmeans.copy()

    if 'Cluster' in df_cross.columns:
        c_sum = df_cross.groupby('Cluster').agg(
            total_mhs=('NILAI KESELURUHAN', 'count'),
            ipk_mean=('NILAI KESELURUHAN', 'mean'),
            dev=('NILAI KESELURUHAN', 'std')
        ).fillna(0).reset_index()

        def _safe_norm(s):
            denom = s.max() - s.min()
            return s * 0 if denom == 0 else (s - s.min()) / denom

        c_sum['ranking'] = (
            0.4 * _safe_norm(c_sum['total_mhs']) +
            0.3 * _safe_norm(c_sum['ipk_mean']) +
            0.3 * (1 - _safe_norm(c_sum['dev']))
        )
        ordered_clusters = c_sum.sort_values(by='ranking', ascending=False)['Cluster'].tolist()
        superior_clusters = ordered_clusters[:2] if len(ordered_clusters) >= 2 else ordered_clusters
        df_cross['is_superior'] = df_cross['Cluster'].isin(superior_clusters)
    else:
        df_cross['is_superior'] = True

    prodi_col = None
    for kandidat in ['PROGRAM STUDI', 'PROGRAM_STUDI', 'PRODI']:
        if kandidat in df_cross.columns:
            prodi_col = kandidat
            break

    if prodi_col:
        stats = df_cross.groupby(['ASAL SEKOLAH', prodi_col]).agg(
            mahasiswa=('NILAI KESELURUHAN', 'count'),
            superior_count=('is_superior', 'sum'),
            ipk_mean=('NILAI KESELURUHAN', 'mean')
        ).reset_index()
    else:
        stats = df_cross.groupby('ASAL SEKOLAH').agg(
            mahasiswa=('NILAI KESELURUHAN', 'count'),
            superior_count=('is_superior', 'sum'),
            ipk_mean=('NILAI KESELURUHAN', 'mean')
        ).reset_index()

    stats['rasio'] = (stats['superior_count'] / stats['mahasiswa']) * 100

    def get_status(row):
        reasons = []
        if row['mahasiswa'] < 3:
            reasons.append("Mahasiswa < 3")
        if row['rasio'] < 50.0:
            reasons.append("Rasio < 50%")
            
        if not reasons:
            return "Lolos Syarat"
        return "Gagal: " + ", ".join(reasons)

    stats['status'] = stats.apply(get_status, axis=1)
    stats = stats.sort_values(['rasio', 'mahasiswa'], ascending=[False, False])
    
    return stats.to_dict(orient='records')
