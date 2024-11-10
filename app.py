import os
import shutil
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing import image

# Load the model once when the server starts
model = tf.keras.models.load_model('model/deepfake_detection_model.h5')

# Set the media directory for storing uploaded files and frames
UPLOAD_FOLDER = 'static/uploads'
FRAMES_FOLDER = 'static/frames'

# Create directories if they do not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

# Function to extract frames from video
def frame_capture(path):
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = True

    # Clean up the frames folder before storing new frames
    shutil.rmtree(FRAMES_FOLDER, ignore_errors=True)
    os.makedirs(FRAMES_FOLDER)

    while success:
        success, img = vidObj.read()
        if not success:
            break
        if count % 20 == 0:
            frame_path = os.path.join(FRAMES_FOLDER, f"frame{count}.jpg")
            cv2.imwrite(frame_path, img)
        count += 1

# Function to evaluate frames for deepfake
def evaluate_frames(directory):
    total_confidence = 0
    num_frames = 0
    results = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img_path = os.path.join(directory, filename)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            confidence = model.predict(img_array)[0][0]
            total_confidence += confidence
            num_frames += 1

            if confidence >= 0.5:
                results.append((filename, "Fake", confidence))
            else:
                results.append((filename, "Real", confidence))

    if num_frames > 0:
        average_confidence = total_confidence / num_frames
        overall_prediction = "The video is predicted as a deepfake." if average_confidence >= 0.5 else "The video is predicted as real."
    else:
        average_confidence = 0
        overall_prediction = "No frames found."

    return results, average_confidence, overall_prediction

# Streamlit app layout
st.title("Deepfake Detection")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save the video to the upload folder
    video_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)  # Use the uploaded file's name directly
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract frames from the video
    frame_capture(video_path)

    # Evaluate the frames
    results, avg_confidence, overall_prediction = evaluate_frames(FRAMES_FOLDER)

    # Display results
    st.subheader("Results")
    st.write(f"Average Confidence: {avg_confidence:.2f}")
    st.write(overall_prediction)

    for filename, label, confidence in results:
        st.write(f"Frame: {filename}, Prediction: {label}, Confidence: {confidence:.2f}")

    # Optionally display the frames
    st.subheader("Extracted Frames")
    for filename in results:
        frame_image_path = os.path.join(FRAMES_FOLDER, filename[0])
        st.image(frame_image_path, caption=filename[0])