# Segmentasi Tingkat Risiko Stroke Berdasarkan Indikator Kesehatan Menggunakan Pemodelan Klasifikasi Machine Learning

Proyek ini mengelompokkan pasien ke dalam tiga segmen tingkat risiko stroke (**Rendah, Sedang, Tinggi**) berdasarkan indikator kesehatan dan gaya hidup. Segmen dibentuk dari probabilitas risiko model klasifikasi yang dilatih dengan memaksimalkan *Recall*, lalu divalidasi dengan angka stroke sebenarnya di tiap segmen.

Seluruh pekerjaan ada di satu notebook: [`strokePrediction.ipynb`](strokePrediction.ipynb).

---

## 1. Business Understanding

- **Masalah**: stroke adalah penyebab utama kematian dan disabilitas jangka panjang. Fasilitas kesehatan membutuhkan penapisan (*screening*) awal untuk menandai pasien yang perlu perhatian lebih.
- **Tujuan**: membagi pasien ke segmen risiko Rendah, Sedang, dan Tinggi.
- **Metrik utama**: **Recall** (kasus berisiko yang terlewat harus sesedikit mungkin), dilengkapi Precision, F1, F2, ROC-AUC, dan PR-AUC. *Accuracy* tidak dijadikan patokan karena data sangat timpang.
- **Catatan interpretasi**: label `stroke` berarti pasien *sudah pernah* stroke. Segmen Tinggi berarti profil kesehatannya mirip pasien yang pernah stroke, bukan prediksi pasti kejadian di masa depan.

## 2. Dataset

- **Nama**: Stroke Prediction Dataset (`healthcare-dataset-stroke-data.csv`), Kaggle oleh *fedesoriano*, lisensi publik (CC0).
- **Ukuran**: 5.110 baris, 12 kolom. Target `stroke` (0 = tidak, 1 = ya), hanya **4,87%** pasien berstatus stroke.
- **Fitur**: `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`. Kolom `id` tidak dipakai.

## 3. Alur Pengerjaan

| Tahap | Isi | Bagian notebook |
| :--- | :--- | :-: |
| **Progress 1** | Business & data understanding, data dictionary, representativitas dan bias sampel | 1 |
| **Progress 2** | Audit kualitas data, EDA, cleaning, preprocessing, feature selection, penanganan imbalance | 2 - 6 |
| **Progress 3** | Pemilihan model, training, pemilihan parameter, hasil, perbandingan model, segmentasi risiko | 7 |

### Ringkasan Progress 2 (data preparation)
- Cleaning: kolom `id` dibuang dan 1 baris `gender = Other` dihapus (menjadi 5.109 baris). Outlier glukosa dan BMI **dipertahankan** karena secara klinis valid.
- Missing value hanya pada `bmi` (201 baris, lebih banyak pada pasien stroke: 16,06% vs 3,31%), diimputasi **median** di dalam pipeline setelah *split* untuk mencegah *data leakage*.
- Split 80:20 dengan `stratify=y`. Numerik: imputasi median + `StandardScaler`. Kategorikal: imputasi modus + `OneHotEncoder` (total 20 fitur).
- Feature selection: skor *Mutual Information*, seluruh 20 fitur dipertahankan.
- Imbalance: **random oversampling hanya pada data train** (3.888 : 199 menjadi 3.888 : 3.888). Data uji tidak diubah.

### Ringkasan Progress 3 (pemodelan)
- **Model** (materi praktikum): Logistic Regression, Decision Tree, K-Nearest Neighbors, ditambah DummyClassifier sebagai patokan.
- **Parameter**: `C` (Logistic Regression), `max_depth` (Decision Tree), `n_neighbors` (KNN) dipilih dengan **Stratified 5-Fold Cross-Validation** pada data train. Oversampling dilakukan di dalam tiap fold. Kriteria pemilihan adalah **F2**, yang mengutamakan recall tetapi tetap menghukum salah alarm berlebihan.
- **Reprodusibilitas**: `random_state = 42` di semua proses acak, dan sel uji ulang membuktikan prediksi identik setelah training ulang.

## 4. Hasil Awal (data uji: 1.022 pasien, 50 kasus stroke)

| Model | Parameter terpilih | Recall | Precision | F2 | ROC-AUC | PR-AUC |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| Dummy (patokan) | - | 0,00 | 0,000 | 0,000 | 0,500 | 0,049 |
| **Logistic Regression** | C = 1 | **0,80** | 0,134 | **0,401** | **0,838** | **0,267** |
| Decision Tree | max_depth = 3 | 0,84 | 0,105 | 0,349 | 0,821 | 0,183 |
| KNN | k = 101 | 0,80 | 0,119 | 0,372 | 0,815 | 0,191 |

Logistic Regression dipakai sebagai dasar segmentasi karena F2 cross-validation-nya tertinggi.

| Segmen | Batas probabilitas | Pasien uji | Kasus stroke | % stroke |
| :--- | :-: | :-: | :-: | :-: |
| Rendah | < 0,5 | 723 | 10 | 1,4% |
| Sedang | 0,5 - 0,8 | 194 | 16 | 8,2% |
| Tinggi | >= 0,8 | 105 | 24 | 22,9% |

Persentase stroke naik konsisten dari Rendah ke Tinggi pada data train maupun data uji. Segmen Sedang + Tinggi mencakup 29% pasien tetapi memuat 80% kasus stroke. Faktor pembeda utama: usia, hipertensi, kadar glukosa, dan penyakit jantung.

**Keterbatasan**: data uji hanya memuat 50 kasus stroke, precision rendah (banyak salah alarm) sebagai konsekuensi mengejar recall, dan label bersifat historis (sudah pernah stroke). Penjelasan lengkap ada di bagian 7.6 notebook.

## 5. Pemetaan ke Daftar Capaian Progress 3

| No | Aspek | Capaian | Lokasi di notebook |
| :-: | :--- | :--- | :-: |
| 1 | Pemilihan model | Model dari materi praktikum beserta alasan terhadap data dan tujuan Business Understanding | 7.1 |
| 2 | Pemilihan model | Alasan memilih masing-masing dari tiga model | 7.1 |
| 3 | Proses training | Training memakai data train hasil Progress 2, dengan komentar penjelasan di tiap blok kode | 7.2 - 7.4 |
| 4 | Proses training | `random_state = 42` ditetapkan, dibuktikan dengan uji training ulang | 7.2, 7.7 |
| 5 | Pengaturan parameter | Pencarian `C`, `max_depth`, `n_neighbors` dengan cross-validation, disertai tabel dan grafik | 7.3 |
| 6 | Hasil modelling | Prediksi, metrik sesuai Business Understanding, dan pembagian segmen risiko | 7.4, 7.5 |
| 7 | Hasil modelling | Tabel perbandingan seluruh model | 7.4 |
| 8 | Hasil modelling | Penjelasan arti hasil awal dan keterbatasannya | 7.6 |

## 6. Cara Menjalankan

```bash
pip install -r requirements.txt
jupyter notebook strokePrediction.ipynb
```

Letakkan `healthcare-dataset-stroke-data.csv` di folder yang sama dengan notebook, lalu jalankan **Restart & Run All**. Seluruh proses (termasuk cross-validation) selesai dalam beberapa menit.

## 7. Struktur Repositori

```
.
├── healthcare-dataset-stroke-data.csv   # dataset
├── strokePrediction.ipynb               # notebook analisis (Progress 1 - 3)
└── readme.md
```
