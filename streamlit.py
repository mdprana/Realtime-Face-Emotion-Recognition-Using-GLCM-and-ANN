import streamlit as st
import cv2
import numpy as np
from joblib import load
import tensorflow as tf
from PIL import Image

st.set_page_config(layout="wide")

st.markdown("""
<style>
.sidebar {
    background-color: #1a1a1a;
    color: white;
    padding: 20px;
}
.profile-section {
    background-color: #2d2d2d;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.copyright {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: rgba(0,0,0,0.8);
    padding: 10px;
    text-align: center;
    color: #888;
    font-size: 12px;
    border-top: 1px solid rgba(255,255,255,0.1);
}
.stSelectbox {
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model = tf.keras.models.load_model('model/model.h5')
    scale = load('model/scaling.pkl')
    label = load('model/label.pkl')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return model, scale, label, face_cascade

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    radius = 2
    n_points = 8 * radius
    lbp = cv2.calcHist([gray], [0], None, [256], [0, 256])
    mean = np.mean(gray)
    std = np.std(gray)
    var = np.var(gray)
    features = np.concatenate([lbp.flatten(), [mean, std, var]])
    return features

def process_image(image, model, scale, label, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_image = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
        features = extract_features(face_image)
        features = features[:30]
        prediction = model.predict(np.array(scale.transform([features])))
        emotion = label.inverse_transform([np.argmax(prediction)])[0]
        cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    return image

def main():
   with st.sidebar:
       st.image("emotion-recognition.webp")
       mode = st.selectbox("Pilih Mode", ["Kamera", "Unggah Gambar"])
       
       st.markdown("""
       <div style='margin-top: 30px; padding: 20px; background: linear-gradient(145deg, #2d2d2d, #1a1a1a); 
                   border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); text-align: center;'>
           <h3 style='color: #4CAF50; margin-bottom: 10px; font-size: 20px;'>Made Pranajaya Dibyacita</h3>
           <p style='color: #BBBBBB; margin-bottom: 5px; font-size: 16px;'>2208561122</p>
           <p style='color: #BBBBBB; font-size: 16px;'>Universitas Udayana</p>
       </div>
       """, unsafe_allow_html=True)
       
       st.markdown("""
       <div style='position: fixed; bottom: 0; left: 0; width: 100%; background-color: rgba(0,0,0,0.8);
                   padding: 10px; text-align: center; border-top: 1px solid rgba(255,255,255,0.1);'>
           <p style='margin: 0; color: #888; font-size: 12px;'>
               © 2025 Made Pranajaya Dibyacita<br>
               Hak Cipta Dilindungi
           </p>
       </div>
       """, unsafe_allow_html=True)

   st.title("EmotiScan: Aplikasi Pendeteksi Emosi Wajah Secara Realtime")
   model, scale, label, face_cascade = load_models()

   col1, col2 = st.columns([2, 1])
   
   with col1:
       if mode == "Kamera":
           st.markdown("### 📸 Mode Kamera")
           img_file_buffer = st.camera_input("Ambil Gambar")
           if img_file_buffer:
               bytes_data = img_file_buffer.getvalue()
               img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
               processed_img = process_image(img, model, scale, label, face_cascade)
               st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
               
       else:
           st.markdown("### 📤 Mode Unggah")
           uploaded_file = st.file_uploader("Unggah gambar Anda", type=["jpg", "jpeg", "png"])
           if uploaded_file:
               file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
               img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
               processed_img = process_image(img, model, scale, label, face_cascade)
               st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
   
   with col2:
       st.markdown("### Petunjuk")
       st.info("""
       1. Pilih mode dari sidebar
       2. Untuk kamera: Klik "Take Photo" (ambil gambar)
       3. Untuk unggah gambar: Pilih "Browser files" lalu pilih gambar dengan ekpresi wajah yang ingin di deteksi
       4. Lihat emosi yang terdeteksi
       """)

if __name__ == '__main__':
    main()
