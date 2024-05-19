import cv2
import numpy as np
from joblib import load
from skimage import feature, util
import tensorflow as tf

# Cek apakah GPU tersedia
if tf.test.is_gpu_available():
    print('Berjalan di GPU')
    print('GPU #0?')
    print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
else:
    print('Berjalan di CPU')

# Memuat model, skala, dan label encoder
model = tf.keras.models.load_model('model/model.h5')
scale = load('model/scaling.pkl')
label = load('model/label.pkl')

# Memuat file xml Haar cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Mendefinisikan GLCM dan fungsi preprocessing
def preprocessing_1(image):
    features = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
    angle = [0, 45, 90, 135, 180]
    res = np.array([])
    image = util.img_as_ubyte(image)
    
    for j in range(len(features)):
        a = feature.graycomatrix(image, distances=[2], angles=angle, levels=256,
                            symmetric=True, normed=True)
        a = feature.graycoprops(a, prop=features[j]).flatten()
        for k in range(len(angle)):
            res = np.append(a[k], res)
    
    return res

# Mulai menangkap video
cap = cv2.VideoCapture(0)

while True:
    # Tangkap frame demi frame
    ret, frame = cap.read()

    # Konversi frame ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dalam frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Untuk setiap wajah yang terdeteksi
    for (x, y, w, h) in faces:
        # Gambar persegi di sekitar wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Deteksi mata dalam wajah
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Deteksi mulut dalam wajah
        mouth = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
        for (mx, my, mw, mh) in mouth:
            cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 255, 0), 2)

        # Praproses gambar wajah dan buat prediksi
        face_image = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
        face_image = preprocessing_1(face_image)
        res = model.predict(np.array(scale.transform([face_image])))
        predicted_class = np.argmax(res)

        # Tampilkan kelas yang diprediksi pada gambar
        predicted_label = label.inverse_transform([predicted_class])
        cv2.putText(frame, str(predicted_label[0]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Tampilkan frame hasil
    cv2.imshow('Emotion Recognition', frame)

    # Keluar dari loop jika menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan capture dan tutup semua jendela ketika selesai
cap.release()
cv2.destroyAllWindows()
