import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import json
import os
import time

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="PadiScan — Deteksi Penyakit Daun Padi",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load custom CSS ───────────────────────────────────────────
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
CLASS_NAMES = ["Bacterial_Blight", "Blast", "Brown_Spot", "Tungro"]
IMG_SIZE    = 224
MODEL_PATH  = "resnet50_final.pth"
META_PATH   = "model_metadata.json"

DISEASE_INFO = {
    "Bacterial_Blight": {
        "id"         : "Hawar Daun Bakteri",
        "en"         : "Bacterial Blight",
        "pathogen"   : "Xanthomonas oryzae pv. oryzae",
        "severity"   : "Tinggi",
        "color"      : "#E05C5C",
        "icon"       : "🔴",
        "symptoms"   : "Tepi daun menguning dan menjadi coklat mulai dari ujung atau tepi, bercak berair yang berubah menjadi kuning kecoklatan, daun layu pada serangan parah.",
        "cause"      : "Bakteri yang menyebar melalui air irigasi, hujan, dan angin. Berkembang pesat pada kondisi lembab dan suhu 25–34°C.",
        "prevention" : "Gunakan varietas tahan (IR64, Ciherang), hindari pemupukan nitrogen berlebih, bersihkan sisa tanaman terinfeksi, atur jarak tanam.",
        "treatment"  : "Aplikasi bakterisida berbahan aktif tembaga (copper hydroxide), drainase lahan yang baik, cabut dan bakar tanaman yang terinfeksi parah.",
        "loss"       : "Kehilangan hasil 20–30%, pada serangan berat bisa mencapai 70%.",
    },
    "Blast": {
        "id"         : "Blas",
        "en"         : "Blast",
        "pathogen"   : "Magnaporthe oryzae",
        "severity"   : "Sangat Tinggi",
        "color"      : "#D97B3A",
        "icon"       : "🟠",
        "symptoms"   : "Bercak belah ketupat (diamond-shaped) berwarna abu-abu dengan tepi coklat pada daun, bercak pada leher malai (neck blast) menyebabkan bulir hampa.",
        "cause"      : "Jamur yang sporanya menyebar melalui angin. Berkembang optimal pada suhu 24–28°C dengan kelembaban tinggi dan embun pagi.",
        "prevention" : "Varietas tahan blast (Inpari 13, 32, 42), hindari tanam serempak, kurangi pupuk nitrogen, rotasi tanaman.",
        "treatment"  : "Fungisida propiconazole, tricyclazole, atau isoprothiolane. Aplikasi pada fase anakan dan menjelang berbunga.",
        "loss"       : "Kehilangan hasil 10–30%, neck blast bisa menyebabkan 100% bulir hampa.",
    },
    "Brown_Spot": {
        "id"         : "Bercak Coklat",
        "en"         : "Brown Spot",
        "pathogen"   : "Cochliobolus miyabeanus",
        "severity"   : "Sedang",
        "color"      : "#A0714F",
        "icon"       : "🟤",
        "symptoms"   : "Bercak oval atau bulat berwarna coklat dengan halo kuning, tersebar di seluruh permukaan daun. Bercak pada sekam menyebabkan biji berubah warna.",
        "cause"      : "Jamur yang berkembang pada tanah miskin hara (kekurangan kalium dan silika). Dipicu stres abiotik pada tanaman.",
        "prevention" : "Pemupukan berimbang (terutama kalium dan silika), gunakan benih sehat, hindari kekeringan, rotasi tanaman.",
        "treatment"  : "Perlakuan benih dengan fungisida (thiram, iprodione), fungisida mancozeb atau propiconazole pada daun.",
        "loss"       : "Kehilangan hasil 5–45%, biji ternoda menurunkan kualitas beras.",
    },
    "Tungro": {
        "id"         : "Tungro",
        "en"         : "Tungro",
        "pathogen"   : "Rice Tungro Bacilliform Virus (RTBV) + Rice Tungro Spherical Virus (RTSV)",
        "severity"   : "Sangat Tinggi",
        "color"      : "#6B9E4E",
        "icon"       : "🟡",
        "symptoms"   : "Daun menguning–oranye mulai dari ujung, tanaman kerdil, jumlah anakan berkurang, malai kecil dan bulir hampa.",
        "cause"      : "Virus yang ditularkan oleh wereng hijau (Nephotettix virescens). Tidak menular langsung antar tanaman.",
        "prevention" : "Kendalikan populasi wereng hijau dengan insektisida, tanam varietas tahan (Tukad Unda, Bondoyudo), tanam serempak.",
        "treatment"  : "Tidak ada obat untuk virus. Fokus pada pengendalian vektor wereng hijau (imidakloprid, BPMC) dan cabut tanaman terinfeksi.",
        "loss"       : "Kehilangan hasil 30–70%, pada serangan berat seluruh pertanaman bisa gagal panen.",
    },
}

# ── Model loader ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        temperature = checkpoint.get("temperature", 1.0)

        model = models.resnet50(weights=None)
        in_f  = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_f, len(CLASS_NAMES))
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        meta = {}
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                meta = json.load(f)

        return model, temperature, meta
    except Exception as e:
        return None, 1.0, {}

def predict(model, temperature, pil_img):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
    ])
    tensor = transform(pil_img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor) / max(temperature, 0.1)
        probs  = F.softmax(logits, dim=1)[0].numpy()
    pred_idx  = int(np.argmax(probs))
    pred_cls  = CLASS_NAMES[pred_idx]
    pred_conf = float(probs[pred_idx])
    return pred_cls, pred_conf, probs

# ── Load model ────────────────────────────────────────────────
model, temperature, meta = load_model()

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-badge">Deep Learning · ResNet-50 · PyTorch</div>
  <h1 class="hero-title">PadiScan</h1>
  <p class="hero-sub">Deteksi Penyakit Daun Padi berbasis Kecerdasan Buatan</p>
  <div class="hero-stats">
    <div class="stat-pill">🎯 Akurasi 100%</div>
    <div class="stat-pill">📊 5-Fold CV 99.74%</div>
    <div class="stat-pill">🌿 4 Jenis Penyakit</div>
    <div class="stat-pill">✅ 5/5 Uji Validasi</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔬 Diagnosa Penyakit", "📚 Ensiklopedia Penyakit", "📈 Tentang Model"])

# ══════════════════════════════════
# TAB 1 — DIAGNOSA
# ══════════════════════════════════
with tab1:
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<p class="section-label">UNGGAH GAMBAR DAUN</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="upload-hint">
          Unggah foto daun padi yang ingin didiagnosa. Pastikan gambar cukup jelas
          dan fokus pada area daun yang terlihat gejala.
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            label="",
            type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed",
        )

        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            st.markdown('<div class="img-frame">', unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True, caption="Gambar yang diunggah")
            st.markdown('</div>', unsafe_allow_html=True)

            # Info gambar
            w, h = pil_img.size
            st.markdown(f"""
            <div class="img-meta">
              <span>📐 {w} × {h} px</span>
              <span>🗂️ {uploaded.type}</span>
              <span>💾 {uploaded.size // 1024} KB</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-upload">
              <div class="empty-icon">🌾</div>
              <div class="empty-text">Belum ada gambar</div>
              <div class="empty-sub">Format: JPG, PNG, BMP</div>
            </div>
            """, unsafe_allow_html=True)

    with col_result:
        st.markdown('<p class="section-label">HASIL DIAGNOSA</p>', unsafe_allow_html=True)

        if uploaded and model:
            with st.spinner("Menganalisis gambar..."):
                time.sleep(0.4)  # UX pause
                pred_cls, pred_conf, all_probs = predict(model, temperature, pil_img)

            info = DISEASE_INFO[pred_cls]
            conf_pct = pred_conf * 100
            sev_color = info["color"]

            # ── Hasil utama ──
            st.markdown(f"""
            <div class="result-card" style="border-left: 4px solid {sev_color}">
              <div class="result-header">
                <span class="result-icon">{info["icon"]}</span>
                <div>
                  <div class="result-name">{info["id"]}</div>
                  <div class="result-en">{info["en"]}</div>
                </div>
                <div class="result-conf" style="color:{sev_color}">{conf_pct:.1f}%</div>
              </div>
              <div class="result-pathogen">🦠 {info["pathogen"]}</div>
              <div class="severity-badge" style="background:{sev_color}22; color:{sev_color}; border:1px solid {sev_color}44">
                Tingkat Keparahan: {info["severity"]}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Probability bars ──
            st.markdown('<p class="section-label" style="margin-top:1.5rem">PROBABILITAS PER KELAS</p>',
                        unsafe_allow_html=True)

            for i, (cls, prob) in enumerate(zip(CLASS_NAMES, all_probs)):
                inf   = DISEASE_INFO[cls]
                pct   = prob * 100
                is_top = cls == pred_cls
                bar_style = f"background: {inf['color']}; width: {pct:.1f}%"
                row_style = "font-weight:600;" if is_top else "opacity:0.7;"

                st.markdown(f"""
                <div class="prob-row" style="{row_style}">
                  <span class="prob-label">{inf["icon"]} {inf["id"]}</span>
                  <div class="prob-bar-bg">
                    <div class="prob-bar" style="{bar_style}"></div>
                  </div>
                  <span class="prob-pct" style="color:{inf['color']}">{pct:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

            # ── Rekomendasi cepat ──
            st.markdown(f"""
            <div class="quick-reco">
              <div class="reco-title">💊 Penanganan Segera</div>
              <div class="reco-text">{info["treatment"]}</div>
              <div class="reco-loss">⚠️ Potensi kerugian: {info["loss"]}</div>
            </div>
            """, unsafe_allow_html=True)

        elif uploaded and not model:
            st.error("⚠️ Model tidak ditemukan. Pastikan file `resnet50_final.pth` ada di direktori yang sama.")
        else:
            st.markdown("""
            <div class="result-empty">
              <div class="result-empty-icon">🔬</div>
              <div class="result-empty-text">Unggah gambar daun untuk memulai diagnosa</div>
              <div class="result-empty-sub">Model akan menganalisis dan menentukan jenis penyakit secara otomatis</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════
# TAB 2 — ENSIKLOPEDIA
# ══════════════════════════════════
with tab2:
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("""
    <p class="page-intro">
      Kenali empat penyakit utama daun padi yang dapat dideteksi oleh model ini.
      Pengetahuan tentang gejala dan penanganan membantu petani bertindak cepat.
    </p>
    """, unsafe_allow_html=True)

    for cls, info in DISEASE_INFO.items():
        c = info["color"]
        with st.expander(f"{info['icon']}  {info['id']}  —  {info['en']}", expanded=False):
            col_a, col_b = st.columns(2, gap="large")

            with col_a:
                st.markdown(f"""
                <div class="ency-block">
                  <div class="ency-label" style="color:{c}">PATOGEN</div>
                  <div class="ency-val italic">🦠 {info["pathogen"]}</div>
                </div>
                <div class="ency-block">
                  <div class="ency-label" style="color:{c}">GEJALA</div>
                  <div class="ency-val">{info["symptoms"]}</div>
                </div>
                <div class="ency-block">
                  <div class="ency-label" style="color:{c}">PENYEBAB & KONDISI</div>
                  <div class="ency-val">{info["cause"]}</div>
                </div>
                """, unsafe_allow_html=True)

            with col_b:
                st.markdown(f"""
                <div class="ency-block">
                  <div class="ency-label" style="color:{c}">PENCEGAHAN</div>
                  <div class="ency-val">{info["prevention"]}</div>
                </div>
                <div class="ency-block">
                  <div class="ency-label" style="color:{c}">PENANGANAN</div>
                  <div class="ency-val">{info["treatment"]}</div>
                </div>
                <div class="ency-block">
                  <div class="ency-label" style="color:{c}">POTENSI KEHILANGAN HASIL</div>
                  <div class="ency-val loss-text">{info["loss"]}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="sev-bar-wrap">
              <span style="color:{c}; font-weight:600">Tingkat Keparahan:</span>
              <span class="sev-chip" style="background:{c}22; color:{c}; border:1px solid {c}55">
                {info["severity"]}
              </span>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════
# TAB 3 — TENTANG MODEL
# ══════════════════════════════════
with tab3:
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="info-card">
          <div class="info-card-title">🧠 Arsitektur Model</div>
          <table class="info-table">
            <tr><td>Model</td><td><b>ResNet-50</b></td></tr>
            <tr><td>Framework</td><td><b>PyTorch</b></td></tr>
            <tr><td>Input size</td><td><b>224 × 224 px</b></td></tr>
            <tr><td>Parameters</td><td><b>25.6 juta</b></td></tr>
            <tr><td>Pretrained</td><td><b>ImageNet</b></td></tr>
            <tr><td>Fine-tuning</td><td><b>Gradual Unfreezing (Top-30)</b></td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
          <div class="info-card-title">⚙️ Pipeline Training</div>
          <table class="info-table">
            <tr><td>Loss function</td><td><b>Label Smoothing (ε=0.1)</b></td></tr>
            <tr><td>Optimizer</td><td><b>AdamW (wd=1e-4)</b></td></tr>
            <tr><td>Scheduler</td><td><b>CosineAnnealingLR</b></td></tr>
            <tr><td>Augmentasi</td><td><b>Flip, Rotate, Jitter, Mixup</b></td></tr>
            <tr><td>Phase 1</td><td><b>8 epoch, LR=1e-3</b></td></tr>
            <tr><td>Phase 2</td><td><b>20 epoch, LR=1e-4</b></td></tr>
            <tr><td>Kalibrasi</td><td><b>Temperature Scaling (T=0.1)</b></td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
          <div class="info-card-title">📊 Hasil Evaluasi</div>
          <table class="info-table">
            <tr><td>Test Accuracy</td><td><b style="color:#4CAF50">100.00%</b></td></tr>
            <tr><td>F1-Score (Weighted)</td><td><b style="color:#4CAF50">1.0000</b></td></tr>
            <tr><td>5-Fold CV Mean</td><td><b>99.74%</b></td></tr>
            <tr><td>5-Fold CV Std</td><td><b>± 0.14%</b></td></tr>
            <tr><td>ECE (setelah TS)</td><td><b>0.0000</b></td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
          <div class="info-card-title">✅ Hasil Uji Validasi Bias</div>
          <div class="uji-row pass">✅ UJI 1 — Data Leakage Check</div>
          <div class="uji-row pass">✅ UJI 2 — Background Sensitivity</div>
          <div class="uji-row pass">✅ UJI 3 — Robustness (Noise/Blur)</div>
          <div class="uji-row pass">✅ UJI 4 — Confidence Calibration (TS)</div>
          <div class="uji-row pass">✅ UJI 5 — 5-Fold Cross-Validation</div>
          <div class="uji-score">Score: 5 / 5</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
          <div class="info-card-title">📁 Dataset</div>
          <table class="info-table">
            <tr><td>Kelas</td><td><b>4 penyakit</b></td></tr>
            <tr><td>Train</td><td><b>3.579 gambar</b></td></tr>
            <tr><td>Validasi</td><td><b>875 gambar</b></td></tr>
            <tr><td>Test</td><td><b>879 gambar</b></td></tr>
            <tr><td>Leakage removed</td><td><b>587 gambar</b></td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div>PadiScan · Deteksi Penyakit Daun Padi dengan Deep Learning</div>
  <div class="footer-sub">ResNet-50 · PyTorch · Streamlit · Dibuat untuk Penelitian Jurnal Sinta 2</div>
</div>
""", unsafe_allow_html=True)
