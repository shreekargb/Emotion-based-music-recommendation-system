# -------------------- IMPORTS --------------------
import numpy as np
import streamlit as st
import cv2
import pandas as pd
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# -------------------- UI STYLE --------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #1e3c72, #2a5298);
    color: white;
}
.stButton>button {
    background-color: #ff4b2b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 220px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🎧 Emotion-Based Music Recommender</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Detect your mood & get songs instantly</h4>", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
df = pd.read_csv("muse_v3.csv")

df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']

df = df[['name','emotional','pleasant','link','artist']]
df = df.sort_values(by=["emotional", "pleasant"])

df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]

# -------------------- FUNCTIONS --------------------
def recommend(emotions):
    data = pd.DataFrame()

    for e in emotions:
        if e == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=10)])
        elif e == 'Angry':
            data = pd.concat([data, df_angry.sample(n=10)])
        elif e == 'Fearful':
            data = pd.concat([data, df_fear.sample(n=10)])
        elif e == 'Happy':
            data = pd.concat([data, df_happy.sample(n=10)])
        else:
            data = pd.concat([data, df_sad.sample(n=10)])

    return data

def preprocess(emotions):
    return list(Counter(emotions).keys())

# -------------------- MODEL --------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.load_weights("model.h5")

emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

# -------------------- CAMERA INPUT --------------------
st.markdown("### 📸 Capture your emotion")

img_file = st.camera_input("Take a picture")

emotion_list = []

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("No face detected. Try again.")
    else:
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48,48))
            roi = np.reshape(roi, (1,48,48,1))

            prediction = model.predict(roi, verbose=0)
            label = emotion_dict[np.argmax(prediction)]

            emotion_list.append(label)

            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(frame, label, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        st.image(frame, channels="BGR")
        st.success(f"Detected Emotion: {label}")

        # -------------------- RECOMMEND --------------------
        processed = preprocess(emotion_list)
        songs = recommend(processed)

        st.markdown("## 🎶 Recommended Songs")

        for i, (link, artist, name) in enumerate(zip(songs["link"], songs["artist"], songs["name"])):
            st.markdown(f"**{i+1}. [{name}]({link})**")
            st.caption(artist)