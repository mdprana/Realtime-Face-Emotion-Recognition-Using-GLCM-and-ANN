# Realtime Face Emotion Recognition

Our GitHub repository houses a cutting-edge project titled "Realtime Face Emotion Recognition System with GLCM and Artificial Neural Networks (ANN)"—a groundbreaking exploration into real-time emotion analysis. Leveraging the power of Gray-Level Co-occurrence Matrix (GLCM) and the versatility of Artificial Neural Networks (ANN), this project aims to revolutionize emotion recognition technology.

Emotions play a pivotal role in human communication and interaction. Understanding and interpreting emotions accurately in real-time scenarios is a challenging task with significant implications across various domains, including human-computer interaction, healthcare, and entertainment.

The core of our project lies in the utilization of GLCM, a texture analysis technique, which captures spatial dependencies within images. By extracting texture features from facial images, we can discern subtle patterns indicative of different emotions. This allows our system to robustly analyze facial expressions and infer underlying emotions with remarkable precision.

## Overview

This is a Python 3 based project to display facial expressions (happy, sad, anger, fear, disgust, surprise, neutral, contempt) by performing fast & accurate face detection with OpenCV using a pre-trained neural networks face detector model shipped with the library.

The model is trained on the **AffectNet** dataset which was published on kaggle. This dataset consists of 29042 RGB pictures, 96x96 sized face images with 8 emotions - angry, disgusted, fearful, happy, neutral, sad, surprise, and contempt.

## Dataset

Source: https://www.kaggle.com/datasets/noamsegal/affectnet-training-data

## Dependencies

1. Python 3.x, OpenCV 3 or 4, Tensorflow, TFlearn, Keras
2. Open terminal and enter the file path to the desired directory and install the following libraries
   * ``` pip install numpy```
   * ``` pip install opencv-python```
   * ``` pip install tensorflow```
   * ``` pip install tflearn```
   * ``` pip install keras```
   * ``` pip install pandas```
   * ```  pip install seaborn ```
   * ``` pip install matplotlib```
   * ``` pip install scikit-image```
   * ``` pip install scikit-learn```
   * ```  pip install pillow ```
   * ``` pip install keras-tuner```
   * ``` pip install joblib```

## Usage
1. Clone the repository.
   ```sh
   git clone https://github.com/mdprana/Realtime-Face-Emotion-Recognition-Using-GLCM-and-ANN.git
   cd Realtime-Face-Emotion-Recognition-Using-GLCM-and-ANN
   ```
2. Install the required dependencies.
3. Adjust model and haarcascade file location in demo.py with your path location (if required).
   ```sh
   model = tf.keras.models.load_model('...yourpath/model/model.h5')
   scale = load('...yourpath/model/scaling.pkl')
   label = load('...yourpath/model/label.pkl')
   ```
   ```sh
   face_cascade = cv2.CascadeClassifier('...yourpath/haarcascade_frontalface_default.xml')
   eye_cascade = cv2.CascadeClassifier('...yourpath/haarcascade_eye.xml')
   mouth_cascade = cv2.CascadeClassifier('...yourpath/haarcascade_smile.xml')
   ```
5. Run demo.py on your IDE or run on CMD (Terminal) with:
   ```sh
   python demo.py
   ```
6. The system will display the video feed with detected faces and recognized emotions.
7. Press the  ``` q ``` key on keyboard to quit the application.

## Realtime Demo Preview

<img src="https://github.com/mdprana/Realtime-Face-Emotion-Recognition-Using-GLCM-and-ANN/assets/95018619/d1168abd-fd68-44b9-bf18-14397621ba50" alt="Picture1" width="300" height="250">
<img src="https://github.com/mdprana/Realtime-Face-Emotion-Recognition-Using-GLCM-and-ANN/assets/95018619/2932711f-5606-455d-95df-2f4b53bc2ca0" alt="Picture2" width="300" height="250">
<br/><br/>

**Detect From Picture**


<img src="https://github.com/mdprana/Realtime-Face-Emotion-Recognition-Using-GLCM-and-ANN/assets/95018619/c8838b3c-0b21-48b9-a808-f6aeafe50c65" alt="Picture3" width="300" height="250">
<br/><br/>

**Detect 2 Emotions, From Picture and Human Face**


<img src="https://github.com/mdprana/Realtime-Face-Emotion-Recognition-Using-GLCM-and-ANN/assets/95018619/099a737b-2eda-4559-b5d5-615219f3f831" alt="Picture4" width="300" height="250">

## YouTube
[![Video Demo](https://img.youtube.com/vi/xBfPaHDOllo/maxresdefault.jpg)](https://www.youtube.com/watch?v=xBfPaHDOllo)
Source: https://youtu.be/xBfPaHDOllo

<br/><br/>
Mata Kuliah Pengantar Pemrosesan Data Multimedia <br/>
Program Studi Informatika
<br/><br/>
**Universitas Udayana** <br/>
**Tahun Ajaran 2023/2024** <br/>
