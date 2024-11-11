import os
import tempfile
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing import image

# Load the model once when the server starts
model = tf.keras.models.load_model('model/deepfake_detection_model.h5')

# Function to extract frames from video in memory
def frame_capture_in_memory(video_path):
    vidObj = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    success = True

    while success:
        success, img = vidObj.read()
        if not success:
            break
        # Only process every 20th frame
        if count % 20 == 0:
            frames.append(img)
        count += 1

    vidObj.release()
    return frames

# Function to evaluate frames for deepfake
def evaluate_frames_in_memory(frames):
    total_confidence = 0
    num_frames = 0
    results = []

    for count, img in enumerate(frames):
        if img is not None:
            # Resize and preprocess the frame
            img = cv2.resize(img, (224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            confidence = model.predict(img_array)[0][0]
            total_confidence += confidence
            num_frames += 1

            # Interpret the model's confidence output
            label = "Fake" if confidence >= 0.5 else "Real"
            results.append((f"Frame {count}", label, confidence))

    # Calculate average confidence and final prediction
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
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_video_path = tmp_file.name

    # Extract frames from the video in memory
    frames = frame_capture_in_memory(temp_video_path)

    # Evaluate the frames in memory
    results, avg_confidence, overall_prediction = evaluate_frames_in_memory(frames)

    # Display results
    st.subheader("Results")
    st.write(f"Average Confidence: {avg_confidence:.2f}")
    st.write(overall_prediction)

    for frame_info in results:
        st.write(f"{frame_info[0]}, Prediction: {frame_info[1]}, Confidence: {frame_info[2]:.2f}")

    # Optionally display the frames
    st.subheader("Extracted Frames")
    for count, img in enumerate(frames):
        if img is not None:
            # Convert frame from BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption=f"Frame {count}")

    # Clean up temporary file after processing
    os.remove(temp_video_path)
