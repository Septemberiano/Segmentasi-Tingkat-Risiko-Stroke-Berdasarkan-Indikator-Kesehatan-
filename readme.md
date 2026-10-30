#  Healthcare Stroke Prediction - Data Preprocessing Pipeline

Proyek ini berisi *pipeline* pemrosesan data (*data preprocessing*) menggunakan **Python**, **Pandas**, dan **Scikit-Learn** untuk mempersiapkan dataset `healthcare-dataset-stroke-data.csv` sebelum dilatih menggunakan algoritma Machine Learning.

---

##  Fitur & Metodologi Preprocessing

1. **Data Cleaning**:
   - Menghapus kolom `id` karena tidak memiliki nilai prediktif.
   - Menghapus sampel dengan `gender == 'Other'` (outlier minoritas extreme).
2. **Pemisahan Fitur & Target**:
   - Target: `stroke` (0 = Tidak, 1 = Ya).
   - Fitur dipisah menjadi kelompok **Numerik**, **Kategorikal**, dan **Biner**.
3. **Automated Pipeline (`ColumnTransformer` & `Pipeline`)**:
   - **Fitur Numerik** (`age`, `avg_glucose_level`, `bmi`):
     - Imputasi nilai hilang (*missing values*) menggunakan **Median**.
     - Penyesuaian skala menggunakan **`StandardScaler`**.
   - **Fitur Kategorikal** (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`):
     - Imputasi nilai hilang menggunakan **Modus** (`most_frequent`).
     - Pengkodean menggunakan **`OneHotEncoder`** (`handle_unknown='ignore'`).
4. **Data Splitting**:
   - Membagi data menjadi **80% Data Latih (Train)** dan **20% Data Uji (Test)**.
   - Menggunakan `stratify=y` untuk menjaga rasio ketidakseimbangan kelas (*class imbalance*) pada target `stroke`.
5. **Handling Class Imbalance**:
   - Menyediakan tahap opsional penyeimbangan data latih menggunakan **SMOTE** (*Synthetic Minority Over-sampling Technique*).

---

##  Cara Penggunaan di Jupyter Notebook (`.ipynb`)

Jalankan sel-sel berikut secara berurutan pada Jupyter Notebook Anda:

### 1. Import Library
```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE