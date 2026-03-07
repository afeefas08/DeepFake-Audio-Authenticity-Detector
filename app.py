import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io, os, time

st.set_page_config(
    page_title="Deepfake Audio Authenticity Detection",
    page_icon="🎙",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
    background: #f5f5f7 !important;
    color: #1d1d1f;
}

[data-testid="collapsedControl"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none !important; }

.block-container {
    max-width: 720px !important;
    padding: 0 1.5rem 5rem !important;
    margin: 0 auto;
}

/* ── Hero ── */
.hero { padding: 4rem 0 3rem; text-align: center; }
.hero-chip {
    display: inline-block;
    background: rgba(0,113,227,0.09);
    color: #0071e3;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    padding: .35rem 1rem;
    border-radius: 999px;
    margin-bottom: 1.4rem;
}
.hero-title {
    font-size: clamp(2.4rem, 6vw, 3.6rem);
    font-weight: 700;
    letter-spacing: -.04em;
    color: #1d1d1f;
    line-height: 1.05;
    margin-bottom: .7rem;
}
.hero-sub {
    font-size: 1.05rem;
    font-weight: 400;
    color: #6e6e73;
    line-height: 1.55;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #d2d2d7 30%, #d2d2d7 70%, transparent);
    margin: 2rem 0;
}

/* ── Section label ── */
.section-label {
    font-size: .72rem;
    font-weight: 600;
    color: #aeaeb2;
    text-transform: uppercase;
    letter-spacing: .1em;
    margin-bottom: 1rem;
}

/* ── Upload — native Streamlit uploader, styled ── */
div[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
}
div[data-testid="stFileUploader"] section {
    border: 1.5px dashed #d2d2d7 !important;
    border-radius: 18px !important;
    background: transparent !important;
    padding: 2rem 1.5rem !important;
    transition: border-color .2s !important;
}
div[data-testid="stFileUploader"] section:hover {
    border-color: #0071e3 !important;
}
/* Browse files button — blue pill */
div[data-testid="stFileUploader"] button {
    background: #0071e3 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 980px !important;
    font-size: .85rem !important;
    font-weight: 600 !important;
    padding: .5rem 1.4rem !important;
    box-shadow: 0 2px 8px rgba(0,113,227,.28) !important;
    transition: background .15s !important;
}
div[data-testid="stFileUploader"] button:hover {
    background: #0077ed !important;
}

/* ── Stat grid ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: .7rem;
}
.stat-cell {
    border: 1.5px solid #d2d2d7;
    border-radius: 14px;
    padding: 1rem .9rem;
    background: transparent;
}
.s-label { font-size: .65rem; font-weight: 600; color: #aeaeb2; text-transform: uppercase; letter-spacing: .07em; }
.s-value { font-size: 1.1rem; font-weight: 700; color: #1d1d1f; margin-top: .25rem; line-height: 1; }
.s-unit  { font-size: .65rem; color: #aeaeb2; }

/* ── Analyse button ── */
.stButton > button {
    background: #0071e3 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 980px !important;
    padding: .8rem 2.4rem !important;
    font-size: .95rem !important;
    font-weight: 600 !important;
    letter-spacing: -.01em !important;
    width: 100% !important;
    transition: background .15s, transform .1s, box-shadow .15s !important;
    box-shadow: 0 2px 12px rgba(0,113,227,.3) !important;
}
.stButton > button:hover {
    background: #0077ed !important;
    box-shadow: 0 4px 20px rgba(0,113,227,.4) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: scale(0.98) !important; }

/* ── Result ── */
.result-wrap {
    margin-top: 1.8rem;
    text-align: center;
    animation: fadeUp .45s cubic-bezier(.16,1,.3,1) both;
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0); }
}
.result-pill {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .09em;
    text-transform: uppercase;
    padding: .38rem 1.1rem;
    border-radius: 999px;
    margin-bottom: .9rem;
}
.pill-real { background: transparent; border: 1.5px solid #34c759; color: #1c7a34; }
.pill-fake { background: transparent; border: 1.5px solid #ff3b30; color: #c9000a; }

.result-number {
    font-size: clamp(4.5rem, 14vw, 7rem);
    font-weight: 700;
    letter-spacing: -.05em;
    line-height: 1;
}
.num-real { color: #059669; }
.num-fake { color: #dc2626; }

.result-sub {
    font-size: 1rem;
    font-weight: 500;
    color: #6e6e73;
    margin-top: .5rem;
    letter-spacing: -.01em;
}

/* ── Uploader dropzone text ── */
div[data-testid="stFileUploaderDropzoneInstructions"] span,
div[data-testid="stFileUploaderDropzoneInstructions"] small,
div[data-testid="stFileUploader"] section span,
div[data-testid="stFileUploader"] section small {
    color: #3b3b40 !important;
    font-weight: 500 !important;
}

/* ── Uploaded filename + file size — grey ── */
div[data-testid="stFileUploader"] span[title],
div[data-testid="uploadedFileData"] span,
div[data-testid="stFileUploaderFile"] span,
div[data-testid="stFileUploaderFile"] small,
div[data-testid="stFileUploaderFile"] p,
div[data-testid="stFileUploader"] [class*="UploadedFile"] *,
div[data-testid="stFileUploader"] [class*="uploadedFile"] *,
div[data-testid="stFileUploader"] li span,
div[data-testid="stFileUploader"] li p,
div[data-testid="stFileUploader"] li div {
    color: #8e8e93 !important;
    font-weight: 500 !important;
    font-size: .88rem !important;
}

/* ── Audio player ── */
audio { width: 100%; border-radius: 12px; margin-bottom: .5rem; }

/* ── Spinner ── */
div[data-testid="stSpinner"] p { color: #6e6e73 !important; font-size: .88rem; }

/* ── Progress bar ── */
div[data-testid="stProgressBar"] > div { background: #e5e5ea !important; border-radius: 999px !important; height: 4px !important; }
div[data-testid="stProgressBar"] > div > div { background: #0071e3 !important; border-radius: 999px !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(path="audio_deepfake_model.h5"):
    try:
        import tensorflow as tf
        for p in [path, "deepfake_audio_cnn.h5", "model.h5"]:
            if os.path.exists(p):
                return tf.keras.models.load_model(p)
    except Exception:
        pass
    return None

def extract_mel(raw_bytes, sr=22050, n_mels=128):
    y, sr = librosa.load(io.BytesIO(raw_bytes), sr=sr, mono=True)
    mel   = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000)
    return librosa.power_to_db(mel, ref=np.max), y, sr

def preprocess(mel_db, train_min, train_max, size=(128, 128)):
    import tensorflow as tf
    mel = mel_db.astype("float32")
    mel = (mel - train_min) / (train_max - train_min + 1e-9)
    mel = np.expand_dims(mel, axis=-1)
    mel = tf.image.resize(mel, size).numpy()
    return mel[np.newaxis]

def plot_mel(mel_db, sr):
    fig, ax = plt.subplots(figsize=(7, 2.8), facecolor="#f5f5f7")
    ax.set_facecolor("#f5f5f7")
    im = librosa.display.specshow(mel_db, sr=sr, x_axis="time",
                                  y_axis="mel", fmax=8000, ax=ax, cmap="Blues")
    fig.colorbar(im, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram", color="#1d1d1f", fontsize=9, fontweight="500", pad=6)
    ax.tick_params(colors="#aeaeb2", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#e5e5ea")
    plt.tight_layout(pad=.8)
    return fig

def audio_features(y, sr):
    dur     = len(y) / sr
    rms     = float(np.sqrt(np.mean(y**2)))
    sc      = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    return dur, rms, sc, rolloff


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-chip">Deep Learning · CNN · Mel Spectrogram</div>
    <div class="hero-title">Audio Authenticity Detector</div>
    <div class="hero-sub">Detect AI-generated and deepfake audio<br>with neural network analysis.</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Upload Audio File</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "",
    type=["wav", "mp3", "ogg", "flac"],
    label_visibility="collapsed"
)

# ── If file uploaded ──────────────────────────────────────────────────────────
if uploaded:
    raw = uploaded.read()

    with st.spinner("Analysing audio…"):
        mel_db, y, sr = extract_mel(raw)
        dur, rms, sc, rolloff = audio_features(y, sr)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Playback
    st.markdown('<div class="section-label">Playback</div>', unsafe_allow_html=True)
    st.audio(raw, format=uploaded.type)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Signal metrics
    st.markdown('<div class="section-label">Signal Metrics</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-cell">
            <div class="s-label">Duration</div>
            <div class="s-value">{dur:.1f}<span class="s-unit"> s</span></div>
        </div>
        <div class="stat-cell">
            <div class="s-label">RMS Energy</div>
            <div class="s-value">{rms:.4f}</div>
        </div>
        <div class="stat-cell">
            <div class="s-label">Spectral Centroid</div>
            <div class="s-value">{sc:.0f}<span class="s-unit"> Hz</span></div>
        </div>
        <div class="stat-cell">
            <div class="s-label">Rolloff</div>
            <div class="s-value">{rolloff:.0f}<span class="s-unit"> Hz</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Mel spectrogram only — no waveform
    st.markdown('<div class="section-label">Mel Spectrogram</div>', unsafe_allow_html=True)
    st.pyplot(plot_mel(mel_db, sr), use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Analyse button
    run = st.button("Analyse", use_container_width=True)

    if run:
        prog   = st.progress(0)
        status = st.empty()
        steps  = ["Loading model…", "Preprocessing spectrogram…",
                  "Running inference…", "Interpreting results…"]
        for i, s in enumerate(steps):
            status.markdown(
                f"<p style='text-align:center;color:#aeaeb2;font-size:.85rem;margin-top:.5rem'>{s}</p>",
                unsafe_allow_html=True
            )
            time.sleep(0.22)
            prog.progress(int((i + 1) / len(steps) * 100))
        prog.empty(); status.empty()

        model = load_model()

        train_min = float(np.load("train_min.npy")) if os.path.exists("train_min.npy") else -80.0
        train_max = float(np.load("train_max.npy")) if os.path.exists("train_max.npy") else 0.0

        if model is not None:
            inp       = preprocess(mel_db, train_min, train_max)
            pred      = model.predict(inp, verbose=0)
            fake_prob = float(pred[0][0] if pred.shape[-1] == 1 else pred[0][1])
        else:
            np.random.seed(abs(hash(uploaded.name)) % (2**31))
            fake_prob = float(np.clip(0.35 + 0.3 * np.random.randn(), 0.05, 0.95))
            st.info("Model file not found — demo mode. Place `audio_deepfake_model.h5` next to `app.py`.")

        real_prob = 1.0 - fake_prob
        is_fake   = fake_prob >= 0.5
        conf_pct  = fake_prob * 100 if is_fake else real_prob * 100

        pill_cls = "pill-fake" if is_fake else "pill-real"
        num_cls  = "num-fake"  if is_fake else "num-real"
        pill_txt = "⚠ Deepfake Detected" if is_fake else "✓ Authentic Audio"
        sub_txt  = f"{'Fake' if is_fake else 'Real'} — {conf_pct:.1f}% confident"

        st.markdown(f"""
        <div class="result-wrap">
            <div class="result-pill {pill_cls}">{pill_txt}</div>
            <div class="result-number {num_cls}">{conf_pct:.1f}<span style="font-size:2rem;font-weight:400;opacity:.4">%</span></div>
            <div class="result-sub">{sub_txt}</div>
        </div>
        """, unsafe_allow_html=True)