import os
import pandas as pd

def proses_upload_data(folder_path):
    def _normalise_columns(df):
        df = df.copy()
        df.columns = (
            df.columns.astype(str)
            .str.replace(r'\n', ' ', regex=True)
            .str.replace(r'[_]+', ' ', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
            .str.upper()
        )
        return df

    # 1) Coba baca format baru (satu file dengan PROGRAM STUDI/IPK)
    all_new_format = []
    print("📁 Daftar file di folder:", os.listdir(folder_path))

    for file in os.listdir(folder_path):
        if not file.lower().endswith('.xlsx'):
            continue
        path_file = os.path.join(folder_path, file)
        try:
            df_raw = pd.read_excel(path_file)
        except Exception as err:
            print(f"⚠️ Gagal membaca {file}: {err}")
            continue

        df_raw = _normalise_columns(df_raw)
        df_raw = df_raw.rename(columns={
            'PROP SEKOLAH': 'PROVINSI SEKOLAH',
            'PROV SEKOLAH': 'PROVINSI SEKOLAH',
            'PROVINSI': 'PROVINSI SEKOLAH',
            'SEKOLAH': 'ASAL SEKOLAH',
            'NPSN': 'NPSN SEKOLAH',
            'PROGRAMSTUDI': 'PROGRAM STUDI',
            'PRODI': 'PROGRAM STUDI',
            'PROGRAM_STUDI': 'PROGRAM STUDI',
            'IPK': 'IPK',
            'PRESTASI1': 'PRESTASI 1',
            'PRESTASI2': 'PRESTASI 2'
        })

        required_new = {'ASAL SEKOLAH', 'NPSN SEKOLAH', 'KOTA SEKOLAH', 'PROVINSI SEKOLAH', 'IPK'}
        if required_new.issubset(set(df_raw.columns)):
            df_new = df_raw.copy()

            # Pastikan tipe data rapi
            for col in ['ASAL SEKOLAH', 'KOTA SEKOLAH', 'PROVINSI SEKOLAH', 'PROGRAM STUDI']:
                if col in df_new.columns:
                    df_new[col] = df_new[col].astype(str).str.strip()

            df_new['IPK'] = (
                df_new['IPK']
                .astype(str)
                .str.replace(',', '.', regex=False)
                .str.extract(r'([-+]?[0-9]*\.?[0-9]+)')[0]
            )
            df_new['IPK'] = pd.to_numeric(df_new['IPK'], errors='coerce')

            if 'STATUS' in df_new.columns:
                df_new['STATUS'] = df_new['STATUS'].astype(str).str.upper().str.strip()
                df_new = df_new[df_new['STATUS'] == 'AKTIF']  # hanya mahasiswa aktif

            df_new = df_new.dropna(subset=['IPK'])
            df_new = df_new[df_new['IPK'] != 0]
            df_new = df_new.rename(columns={'IPK': 'NILAI KESELURUHAN'})

            wanted_cols = [
                'ASAL SEKOLAH', 'NPSN SEKOLAH', 'KOTA SEKOLAH',
                'PROVINSI SEKOLAH', 'PROGRAM STUDI', 'NILAI KESELURUHAN',
                'STATUS', 'JURUSAN', 'JURUSAN SEKOLAH', 'PRESTASI 1', 'PRESTASI 2'
            ]
            df_new = df_new[[c for c in wanted_cols if c in df_new.columns]]
            all_new_format.append(df_new)
            print(f"✅ Format baru terdeteksi pada {file}, baris: {df_new.shape[0]}")

    if all_new_format:
        df = pd.concat(all_new_format, ignore_index=True)
        print("📊 Sebelum pembersihan (format baru):", df.shape)

        # ⚙️ Bersihkan placeholder tanpa menghapus data valid
        df = df.replace({'-': pd.NA, '--': pd.NA})
        required_cols = [c for c in [
            'ASAL SEKOLAH', 'NPSN SEKOLAH', 'KOTA SEKOLAH',
            'PROVINSI SEKOLAH', 'NILAI KESELURUHAN'
        ] if c in df.columns]
        if required_cols:
            df = df.dropna(subset=required_cols)

        print("📉 Setelah pembersihan (format baru):", df.shape)
    else:
        # 2) Fallback: format lama (pasangan data_xxxx & nilai_xxxx)
        all_data = []
        for file in os.listdir(folder_path):
            if "data" in file.lower():
                tahun = ''.join(filter(str.isdigit, file))
                file_data = os.path.join(folder_path, file)
                file_nilai = os.path.join(folder_path, f"nilai_{tahun}.xlsx")

                if os.path.exists(file_nilai):
                    biodata = pd.read_excel(file_data, skiprows=4)
                    nilai = pd.read_excel(file_nilai, skiprows=4)

                    biodata.columns = biodata.columns.str.strip().str.upper()
                    nilai.columns = nilai.columns.str.strip().str.upper()

                    biodata = biodata[biodata['NIM'].notna()]
                    nilai = nilai[nilai['NIM'].notna()]

                    print("📁 Membaca:", file_data, "dan", file_nilai)
                    print("📄 Kolom biodata:", biodata.columns.tolist())
                    print("📄 Kolom nilai:", nilai.columns.tolist())

                    kolom_data = ['NIM', 'ASAL SEKOLAH', 'NPSN SEKOLAH', 'KOTA SEKOLAH', 'PROVINSI SEKOLAH', 'JALUR MASUK', 'STATUS']
                    kolom_nilai = ['NIM', 'NILAI KESELURUHAN']

                    if all(k in biodata.columns for k in kolom_data) and all(k in nilai.columns for k in kolom_nilai):
                        gabung = pd.merge(
                            biodata[kolom_data],
                            nilai[kolom_nilai],
                            on='NIM',
                            how='left'
                        )
                        all_data.append(gabung)
                    else:
                        print(f"⚠️ Kolom tidak lengkap untuk tahun {tahun}, dilewati.")

        if not all_data:
            print("❌ Tidak ada data valid yang ditemukan di folder uploads.")
            return pd.DataFrame()

        df = pd.concat(all_data, ignore_index=True)
        print("📊 Sebelum pembersihan:", df.shape)

        df['ASAL SEKOLAH'] = df['ASAL SEKOLAH'].astype(str)
        df['STATUS'] = df['STATUS'].astype(str)
        df['JALUR MASUK'] = df['JALUR MASUK'].astype(str)

        df = df[~df['JALUR MASUK'].isin([
            'RPL Transfer SKS D2 ke D3',
            'RPL Transfer SKS Dari D3 ke D4',
            'Pertukaran Mahasiswa Merdeka'
        ])]
        df = df[~df['STATUS'].isin([
            'MD MABA',
            'Keluar/Mengundurkan Diri',
            'Tidak Aktif Mengulang TA (Mahasiswa Tidak Jelas)'
        ])]
        df = df[~df['ASAL SEKOLAH'].isin([
            'POLITEKNIK NEGERI MALANG',
            'POLITEKNIK NEGERI JEMBER',
            'POLITEKNIK NEGERI BANYUWANGI',
            'POLITEKNIK NEGERI MADIUN'
        ])]

        df = df.drop(columns=['JALUR MASUK', 'STATUS'], errors='ignore')
        df = df[~df.apply(lambda row: row.astype(str).str.contains('-').any(), axis=1)]
        df = df[df['NILAI KESELURUHAN'] != 0]
        df = df.dropna()
        print("📉 Setelah pembersihan:", df.shape)

    os.makedirs("outputs", exist_ok=True)
    df.to_excel("outputs/data_gabungan_clean.xlsx", index=False)
    return df

def prepare_data(df):
    import numpy as np
    import random
    from sklearn.preprocessing import LabelEncoder, StandardScaler

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
