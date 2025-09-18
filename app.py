import streamlit as st
from transformers import pipeline
import whisper
import tempfile
import os
import pandas as pd
from fpdf import FPDF
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from langdetect import detect, DetectorFactory
import soundfile as sf
import numpy as np

DetectorFactory.seed = 0  # ensures consistent results

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_toxic_model():
    # ✅ Valid multilingual toxicity model
    return pipeline(
        "text-classification",
        model="textdetox/xlmr-large-toxicity-classifier-v2",
        framework="pt"
    )

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")  # Whisper supports multilingual transcription

classifier = load_toxic_model()
whisper_model = load_whisper_model()

# -----------------------------
# Store results
# -----------------------------
results_list = []

def save_result(text, label, confidence, lang):
    results_list.append({
        "Input Text": text,
        "Language": lang,
        "Prediction": label,
        "Confidence %": confidence
    })

# -----------------------------
# Helper: detect language
# -----------------------------
def detect_language(text):
    try:
        lang = detect(text)
    except:
        lang = "unknown"
    return lang

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌍 Multilingual Hate Speech Detection (with Language ID)")

option = st.radio("Choose input type:", ["Text", "Upload Audio", "Mic Recording"])

# -----------------------------
# Option 1: Text Input
# -----------------------------
if option == "Text":
    user_input = st.text_area("Enter text here (any language):")
    if st.button("Predict Text"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text.")
        else:
            lang = detect_language(user_input)
            st.info(f"🌐 Detected Language: {lang}")

            result = classifier(user_input)[0]
            label = result["label"]
            score = round(result["score"] * 100, 2)
            st.write(f"Prediction: {label} ({score}% confidence)")

            save_result(user_input, label, score, lang)

# -----------------------------
# Option 2: Upload Audio File
# -----------------------------
elif option == "Upload Audio":
    audio_file = st.file_uploader("Upload an audio file (WAV/MP3/M4A)", type=["wav", "mp3", "m4a"])
    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")
        if st.button("Transcribe & Predict"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_file.read())
                temp_path = temp_audio.name

            transcription = whisper_model.transcribe(temp_path)
            text = transcription["text"]
            st.write("🗣️ Transcribed Text:", text)

            lang = detect_language(text)
            st.info(f"🌐 Detected Language: {lang}")

            result = classifier(text)[0]
            label = result["label"]
            score = round(result["score"] * 100, 2)
            st.write(f"Prediction: {label} ({score}% confidence)")

            save_result(text, label, score, lang)

            os.remove(temp_path)

# -----------------------------
# Option 3: Mic Recording
# -----------------------------
elif option == "Mic Recording":
    st.info("🎤 Click 'Start' below and speak in ANY language.")

    class AudioProcessor(AudioProcessorBase):
        def recv_audio_frame(self, frame):
            # Convert audio frame to numpy array
            audio = frame.to_ndarray().flatten().astype("float32")

            # Save as WAV file with 16kHz sample rate (needed for Whisper)
            sf.write("mic_input.wav", audio, 16000)

            return frame

    webrtc_streamer(
        key="mic",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

    if st.button("Process Mic Audio"):
        if os.path.exists("mic_input.wav"):
            transcription = whisper_model.transcribe("mic_input.wav")
            text = transcription["text"]
            st.write("🗣️ Transcribed Text:", text)

            lang = detect_language(text)
            st.info(f"🌐 Detected Language: {lang}")

            result = classifier(text)[0]
            label = result["label"]
            score = round(result["score"] * 100, 2)
            st.write(f"Prediction: {label} ({score}% confidence)")

            save_result(text, label, score, lang)
        else:
            st.warning("⚠️ No mic input found. Please record first.")

# -----------------------------
# Export Results Section
# -----------------------------
if results_list:
    st.subheader("📊 Export Results")

    df = pd.DataFrame(results_list)

    # Download as CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name="hate_speech_results.csv", mime="text/csv")

    # Download as PDF
    if st.button("⬇️ Download PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Hate Speech Detection Report", ln=True, align="C")

        for i, row in df.iterrows():
            pdf.multi_cell(
                0, 10,
                f"Input: {row['Input Text']}\nLanguage: {row['Language']}\nPrediction: {row['Prediction']} ({row['Confidence %']}%)\n"
            )

        pdf.output("hate_speech_results.pdf")
        with open("hate_speech_results.pdf", "rb") as f:
            st.download_button("Download PDF", f, file_name="hate_speech_results.pdf", mime="application/pdf")
