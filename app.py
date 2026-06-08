import streamlit as st
import keras
import numpy as np
from PIL import Image
from keras.applications.mobilenet_v3 import preprocess_input

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(
    page_title="Deteksi Penyakit Cabai",
    page_icon="🌶️",
    layout="centered"
)

# =========================
# LOGO BINUS
# =========================
st.image(
    "assets/binus_logo.png",
    width=100
)

# =========================
# LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_my_model():
    return keras.models.load_model(
        "model/MobileNetV3_model.keras",
        compile=False
    )

model = load_my_model()

# =========================
# CLASS NAMES
# =========================
class_names = [
    "Antraknosa",
    "Busuk Buah",
    "Cercospora",
    "Lalat Buah",
    "Healthy"
]

# =========================
# UI HEADER
# =========================
st.title("🌶️ Deteksi Penyakit Tanaman Cabai")

st.write(
    "Upload gambar atau gunakan kamera "
    "untuk mendeteksi penyakit pada cabai."
)

st.divider()

# =========================
# PILIH SUMBER INPUT
# =========================
option = st.radio(
    "Pilih sumber gambar:",
    ["Upload Gambar", "Gunakan Kamera"]
)

image = None

# =========================
# INPUT DARI FILE
# =========================
if option == "Upload Gambar":
    uploaded_file = st.file_uploader(
        "Pilih gambar...",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

# =========================
# INPUT DARI KAMERA
# =========================
elif option == "Gunakan Kamera":
    camera_image = st.camera_input("Ambil gambar")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

# =========================
# PROSES JIKA ADA GAMBAR
# =========================
if image is not None:

    st.image(
        image,
        caption="Gambar yang digunakan",
        use_container_width=True
    )

    # =========================
    # PREPROCESSING
    # =========================
    img = image.resize((224, 224))

    img_array = keras.utils.img_to_array(img)   # ← pakai keras, bukan tf.keras

    img_array = np.expand_dims(img_array, axis=0)

    img_preprocessed = preprocess_input(img_array)

    # =========================
    # PREDIKSI
    # =========================
    with st.spinner("🔍 Menganalisis gambar..."):

        prediction = model.predict(img_preprocessed)

        index = np.argmax(prediction)

        confidence = np.max(prediction) * 100

    # =========================
    # VALIDASI HASIL
    # =========================
    THRESHOLD = 70

    if index < len(class_names) and confidence >= THRESHOLD:
        result = class_names[index]
        is_valid = True
    else:
        result = "Error: Hasil tidak valid"
        is_valid = False

    # =========================
    # HASIL
    # =========================
    st.divider()
    st.subheader("📊 Hasil Prediksi")

    if is_valid:
        st.success(f"**{result}**")
        st.progress(int(confidence))
        st.write(f"Tingkat Keyakinan: **{confidence:.2f}%**")

        st.subheader("📈 Probabilitas Semua Kelas")
        for i, class_name in enumerate(class_names):
            prob = prediction[0][i] * 100
            st.write(f"{class_name}: {prob:.2f}%")

        if confidence < 80:
            st.warning(
                "⚠️ Keyakinan cukup rendah, "
                "pastikan gambar jelas."
            )
    else:
        st.error(result)

# =========================
# FOOTER
# =========================
st.divider()
st.caption(
    "Aplikasi Deteksi Penyakit Cabai "
    "menggunakan Deep Learning (MobileNetV3)"
)