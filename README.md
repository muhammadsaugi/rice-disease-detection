# 🌾 PadiScan — Deteksi Penyakit Daun Padi

Aplikasi web berbasis Deep Learning untuk mendeteksi 4 penyakit utama pada daun padi:
**Bacterial Blight**, **Blast**, **Brown Spot**, dan **Tungro**.

**Model:** ResNet-50 | **Akurasi:** 100% (Test Set Bersih) | **5-Fold CV:** 99.74% ± 0.14%

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-NAME.streamlit.app)

---

## 📁 Struktur Folder

```
rice-disease-app/
├── app.py                  ← Aplikasi utama Streamlit
├── resnet50_final.pth      ← Model terlatih (dari Google Drive)
├── model_metadata.json     ← Metadata model
├── requirements.txt        ← Dependensi Python
├── .streamlit/
│   └── config.toml         ← Konfigurasi tema Streamlit
└── assets/
    └── style.css           ← Custom CSS
```

---

## 🚀 Panduan Deploy — Step by Step

### STEP 1: Siapkan Repository GitHub

**1.1. Buat repository baru di GitHub**
- Buka [github.com](https://github.com) → klik **New**
- Nama repo: `rice-disease-detection` (atau nama lain)
- Visibilitas: **Public** (wajib agar Streamlit bisa akses)
- Jangan centang "Add README" (sudah ada)
- Klik **Create repository**

**1.2. Upload semua file ke repo**

> ⚠️ File `.pth` berukuran ~90 MB, terlalu besar untuk GitHub biasa.
> Gunakan **Git LFS** (Large File Storage).

Buka terminal / Git Bash, jalankan:

```bash
# Clone repo kosong
git clone https://github.com/USERNAME/rice-disease-detection.git
cd rice-disease-detection

# Install Git LFS (jika belum)
git lfs install

# Track file .pth agar disimpan di LFS
git lfs track "*.pth"
git add .gitattributes

# Salin semua file dari folder rice-disease-app ke sini
# (app.py, requirements.txt, model_metadata.json, dll)

# Download resnet50_final.pth dari Google Drive, taruh di sini

# Tambahkan semua file
git add .
git commit -m "Initial commit: PadiScan rice disease detection app"
git push origin main
```

**1.3. Verifikasi di GitHub**
- Pastikan semua file terupload
- File `.pth` harus terlihat dengan ikon **LFS** di GitHub
- Jika repo masih kosong: refresh browser

---

### STEP 2: Deploy ke Streamlit Community Cloud

**2.1. Buka Streamlit Cloud**
- Kunjungi [share.streamlit.io](https://share.streamlit.io)
- Login dengan akun GitHub Anda

**2.2. Deploy aplikasi**
- Klik **New app**
- Pilih repository: `rice-disease-detection`
- Branch: `main`
- Main file path: `app.py`
- Klik **Deploy!**

**2.3. Tunggu proses build**
- Streamlit akan install semua dependensi dari `requirements.txt`
- Proses biasanya 3-5 menit
- Status bisa dilihat di bagian **Manage apps**

**2.4. Akses aplikasi**
- URL format: `https://USERNAME-rice-disease-detection-app-XXXXX.streamlit.app`
- Salin URL ini untuk dibagikan

---

### STEP 3: Alternatif — Jika File .pth Tidak Bisa di GitHub LFS

Jika LFS quota habis atau bermasalah, gunakan Google Drive + `gdown`:

**3.1. Buat file `download_model.py`:**

```python
import gdown
import os

MODEL_ID = "GANTI_DENGAN_GOOGLE_DRIVE_FILE_ID"

if not os.path.exists("resnet50_final.pth"):
    print("Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}",
                   "resnet50_final.pth", quiet=False)
    print("Model downloaded!")
```

**3.2. Tambahkan di awal `app.py`:**

```python
# Di bagian paling atas, setelah import
import subprocess
if not os.path.exists("resnet50_final.pth"):
    subprocess.run(["python", "download_model.py"])
```

**3.3. Tambahkan `gdown` ke `requirements.txt`:**

```
gdown==5.1.0
```

**3.4. Cara dapat File ID Google Drive:**
- Klik kanan file `.pth` di Google Drive → **Share** → **Anyone with the link**
- Salin link: `https://drive.google.com/file/d/FILE_ID_DI_SINI/view`
- Ambil bagian `FILE_ID_DI_SINI`

---

### STEP 4: Update Model / Kode

Setiap kali ada perubahan:

```bash
git add .
git commit -m "Update: deskripsi perubahan"
git push origin main
```

Streamlit akan otomatis re-deploy dalam 1-2 menit.

---

## 🖥️ Jalankan Lokal (Testing)

```bash
# Install dependensi
pip install -r requirements.txt

# Pastikan resnet50_final.pth ada di folder yang sama
# Jalankan aplikasi
streamlit run app.py
```

Buka browser: `http://localhost:8501`

---

## 📊 Informasi Model

| Metrik | Nilai |
|--------|-------|
| Arsitektur | ResNet-50 |
| Test Accuracy | 100.00% |
| 5-Fold CV | 99.74% ± 0.14% |
| F1-Score | 1.0000 |
| ECE (setelah TS) | 0.0000 |
| Uji Validasi | 5/5 Lulus |

---

## 📚 Kelas Penyakit

| Kelas | Nama Indonesia | Patogen | Keparahan |
|-------|---------------|---------|-----------|
| Bacterial_Blight | Hawar Daun Bakteri | *Xanthomonas oryzae* | Tinggi |
| Blast | Blas | *Magnaporthe oryzae* | Sangat Tinggi |
| Brown_Spot | Bercak Coklat | *Cochliobolus miyabeanus* | Sedang |
| Tungro | Tungro | RTBV + RTSV | Sangat Tinggi |

---

## 📄 Sitasi

Jika menggunakan model ini dalam penelitian:

```
[Nama Anda], et al. (2026). Deteksi Penyakit Daun Padi Menggunakan 
Deep Learning dengan ResNet-50 dan Validasi Bias Komprehensif. 
Jurnal [Nama Jurnal Sinta 2].
```

---

*Dibuat untuk keperluan penelitian jurnal Sinta 2.*
