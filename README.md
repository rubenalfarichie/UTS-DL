# UTS Deep Learning — Kelompok 4

**Hands-on Deep Learning: Multimodal Classification, Time-Series RUL Regression, and Multimodal Clustering**

## Anggota Kelompok 4

| Nama | NIM | Kelas | GitHub | Peran & Kontribusi |
| :--- | :--- | :--- | :--- | :--- |
| **Ruben Alfa Richie** | `101032300008` | TK-47-06 | [@rubenalfarichie](https://github.com/rubenalfarichie) | Pembuat repo utama, struktur proyek, EDA, **Tugas 1 (Klasifikasi Multimodal)** & **Tugas 2 (Regresi RUL)** |
| **Omar Al Farizy Hutagalung** | `101032300034` | TK-47-03 | [@omralfa](https://github.com/omralfa) | Fork & Pull Request, **Tugas 3 (Klastering Multimodal)**, **Laporan (`laporan.pdf`)**, & `kontribusi.md` |

---

## Struktur Repositori

```text
.
├── laporan.pdf                  # Laporan lengkap Tugas 1–3 (<= 18 halaman)
├── prediksi_klasifikasi.csv     # Prediksi Tugas 1 (16.000 unit, kolom: id_unit,kelas)
├── prediksi_rul.csv             # Prediksi Tugas 2 (300 mesin, kolom: id_unit,rul)
├── cluster.csv                  # Hasil Klaster Tugas 3 (36.000 sampel, kolom: id_sampel,cluster)
├── kontribusi.md                # Pembagian peran, tautan PR, dan deklarasi penggunaan AI
├── weights/                     # Bobot model final PyTorch (.pt, < 100 MB)
└── kode/
    ├── README.md                # Dokumentasi ini
    ├── requirements.txt         # Daftar dependensi dan versi pustaka
    ├── run_all.py               # Menjalankan pipeline eksperimen dan verifikasi
    ├── utils.py                 # Utilitas bersama, metrik evaluasi, dan random seed
    ├── eda_figures.py           # Generasi grafik EDA untuk Tugas 1–3
    ├── task1_classification.py  # Pipeline lengkap Tugas 1 (ablation, noise, domain shift, 3-seed)
    ├── task2_rul_regression.py  # Pipeline lengkap Tugas 2 (preprocessing kausal, 4 model, MC-Dropout)
    ├── task3_clustering.py      # Pipeline lengkap Tugas 3 (SimCLR+InfoNCE, K-Means/GMM/DEC, t-SNE)
    └── verify_submission.py     # Verifikasi otomatis format, baris, kolom, dan rentang
```

---

## Lingkungan & Versi Pustaka

Seluruh eksperimen dijalankan dan diverifikasi pada:

- **GPU**: NVIDIA GeForce RTX 3070 Ti Laptop GPU (8 GB VRAM, CUDA 12.6)
- **Python**: `3.15.0`
- **PyTorch**: `2.14.1+cu126`, **torchvision**: `0.29.1+cu126`
- **NumPy**: `2.5.4`, **Pandas**: `3.1.1`, **SciPy**: `1.18.1`
- **scikit-learn**: `1.9.1`, **LightGBM**: `4.7.0`
- **Matplotlib**: `3.11.2`, **Pillow**: `12.3.0`, **ReportLab**: `5.1.0`
- **Random Seeds**: `42` (utama & validasi), serta `42, 102, 202` untuk evaluasi 3-seed dan ensemble.

---

## Cara Reproduksi (Satu Perintah)

1. **Siapkan dataset**: Letakkan seluruh berkas dari `data_uts_dl.zip` ke dalam folder `data/` di akar repositori:

   ```text
   data/
     kamus_data.md
     klasifikasi_train.npz, klasifikasi_train_meta.csv
     klasifikasi_test.npz,  klasifikasi_test_meta.csv
     rul_train.csv.gz,      rul_test.csv.gz
     clustering.npz,        clustering_meta.csv
   ```

2. **Instal dependensi**:

   ```bash
   pip install -r kode/requirements.txt
   ```

3. **Jalankan pipeline eksperimen** (EDA, Tugas 1–3, dan verifikasi tanpa membuat ulang `laporan.pdf`):

   ```bash
   python kode/run_all.py --skip-report
   ```

4. **Menjalankan tugas secara terpisah** (opsional, misalnya saat ujian lisan / *oral exam*):

   ```bash
   python kode/task1_classification.py
   python kode/task2_rul_regression.py
   python kode/task3_clustering.py
   python kode/verify_submission.py
   ```