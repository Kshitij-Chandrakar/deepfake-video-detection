import streamlit as st
import requests
import tempfile

# FastAPI URL
API_URL = "http://127.0.0.1:8000/predict-video/"

st.title("🎥 Deepfake Video Detection")

st.write("Upload a video to check if it is REAL or FAKE")

# Upload video
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:

    # Show video
    st.video(uploaded_file)

    if st.button("Analyze"):

        with st.spinner("Processing..."):

            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                temp.write(uploaded_file.read())
                temp_path = temp.name

            # Send to FastAPI
            with open(temp_path, "rb") as f:
                files = {"file": f}
                response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                result = response.json()

                st.success("Prediction Complete!")

                st.write("### Result:")
                st.write(f"Prediction: **{result['prediction']}**")
                st.write(f"Confidence: **{result['confidence']:.4f}**")
                st.write(f"Frames used: **{result['frames_used']}**")

            else:
                st.error("Error from API")
