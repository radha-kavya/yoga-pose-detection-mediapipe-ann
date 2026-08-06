import av
import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st

from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ------------------------------------
# Page Config
# ------------------------------------
st.set_page_config(
    page_title="Yoga Pose Detection",
    layout="centered"
)

st.title("🧘 Yoga Pose Detection using ANN + MediaPipe")

# ------------------------------------
# Load Model
# ------------------------------------
model = load_model("yoga_ann_model.keras")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")

# ------------------------------------
# MediaPipe
# ------------------------------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# ------------------------------------
# Sidebar
# ------------------------------------
option = st.sidebar.selectbox(
    "Choose Input",
    ["Webcam", "Upload Image"]
)

# ====================================
# WEBCAM
# ====================================
if option == "Webcam":

    class PoseProcessor(VideoProcessorBase):

        def __init__(self):

            self.pose = mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

        def recv(self, frame):

            image = frame.to_ndarray(format="bgr24")

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            results = self.pose.process(rgb)

            if results.pose_landmarks:

                landmarks = []

                for lm in results.pose_landmarks.landmark:
                    landmarks.extend([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

                landmarks = np.array(
                    landmarks
                ).reshape(1, -1)

                landmarks = scaler.transform(
                    landmarks
                )

                prediction = model.predict(
                    landmarks,
                    verbose=0
                )

                class_id = np.argmax(prediction)

                pose_name = encoder.inverse_transform(
                    [class_id]
                )[0]

                confidence = np.max(prediction)

                cv2.putText(
                    image,
                    f"{pose_name} {confidence*100:.2f}%",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

                mp_draw.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            return av.VideoFrame.from_ndarray(
                image,
                format="bgr24"
            )

    webrtc_streamer(
        key="webcam",
        video_processor_factory=PoseProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

# ====================================
# IMAGE UPLOAD
# ====================================
else:

    uploaded_file = st.file_uploader(
        "Upload Yoga Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        pose = mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5
        )

        results = pose.process(rgb)

        if results.pose_landmarks:

            landmarks = []

            for lm in results.pose_landmarks.landmark:
                landmarks.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            landmarks = np.array(
                landmarks
            ).reshape(1, -1)

            landmarks = scaler.transform(
                landmarks
            )

            prediction = model.predict(
                landmarks,
                verbose=0
            )

            class_id = np.argmax(prediction)

            pose_name = encoder.inverse_transform(
                [class_id]
            )[0]

            confidence = np.max(prediction)

            cv2.putText(
                image,
                f"{pose_name} {confidence*100:.2f}%",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            mp_draw.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                ),
                caption=f"Prediction: {pose_name} ({confidence*100:.2f}%)",
                use_container_width=True
            )

            st.success(f"Detected Pose: {pose_name}")
            st.info(f"Confidence: {confidence*100:.2f}%")

        else:
            st.error("No pose detected in the uploaded image.")