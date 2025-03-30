import cv2
import os
import time
import base64
import threading
import smtplib
from email.message import EmailMessage
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(
    page_title="Fire/Smoke Detection System",
    page_icon="🔥",
    layout="wide"
)

# ✅ Set up Gemini AI
GOOGLE_API_KEY = ""  # 🔴 Replace with your Gemini API Key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# ✅ Email Configuration
EMAIL_SENDER = ""  # 🔴 Replace with your email
EMAIL_PASSWORD = ""  # 🔴 Replace with your email app password
EMAIL_RECEIVER = ""  # 🔴 Replace with recipient email

IMAGE_PATH = "latest_frame.jpg"  # ✅ Overwrites previous frame
SEND_INTERVAL = 5  # ✅ Send image every 5 seconds
last_sent_time = 0  # ✅ Tracks last AI request time

# Streamlit session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'frame' not in st.session_state:
    st.session_state.frame = None
if 'alert_sent' not in st.session_state:
    st.session_state.alert_sent = False
if 'last_alert' not in st.session_state:
    st.session_state.last_alert = ""

def send_email_alert(subject, body):
    """Sends an AI-generated email alert with the detected fire/smoke image."""
    if not os.path.exists(IMAGE_PATH):
        st.warning("⚠️ Email alert skipped: No image file found.")
        return

    try:
        # ✅ Remove newline & carriage return from subject
        subject = subject.replace("\n", " ").replace("\r", "").strip()

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg.set_content(body)

        with open(IMAGE_PATH, "rb") as img_file:
            msg.add_attachment(img_file.read(), maintype="image", subtype="jpeg", filename="fire_alert.jpg")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        st.session_state.alert_sent = True
        st.session_state.last_alert = f"📧 AI Email Alert Sent at {time.strftime('%H:%M:%S')} - {subject}"
        st.success("📧 AI Email Alert Sent Successfully!")

    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")

def analyze_with_gemini():
    """Sends the latest frame to Gemini AI to check for fire or smoke."""
    if not os.path.exists(IMAGE_PATH):
        st.warning("⚠️ No image available for analysis. Skipping AI check.")
        return

    try:
        with open(IMAGE_PATH, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "text", "text": """
                Analyze the image and determine if fire or smoke is present.
                If fire or smoke is detected, generate a complete email automatically.
                - Write a **clear and urgent subject** (avoid long text).
                - Write a **professional but urgent email body**.
                - **Include emergency contact numbers** for Fire Department and Ambulance (local numbers for MUMBAI MAHARASHTRA).
                - If no fire or smoke is detected, simply respond with "No fire detected."
                """},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )

        response = gemini_model.invoke([message])
        result = response.content.strip()
        st.session_state.last_alert = f"🔍 Last AI Detection at {time.strftime('%H:%M:%S')}: {result[:100]}..."
        st.info(f"🔍 AI Detection Result: {result}")

        if "No fire detected" in result:
            st.info("✅ No fire detected. Skipping email alert.")
            return

        # ✅ Extract subject and body automatically
        lines = result.split("\n")
        subject = lines[0] if len(lines) > 0 else "Fire Alert!"
        body = "\n".join(lines[1:]) if len(lines) > 1 else "Possible fire detected. Take immediate action."

        # ✅ Send email alert
        send_email_alert(subject, body)

        # ✅ Safe file deletion check
        if os.path.exists(IMAGE_PATH):
            os.remove(IMAGE_PATH)

    except Exception as e:
        st.warning("⚠️ AI Analysis Skipped: Image could not be processed.")

def process_frame(frame):
    """Saves the latest frame every 5 seconds and starts AI analysis."""
    global last_sent_time
    current_time = time.time()

    if current_time - last_sent_time >= SEND_INTERVAL:
        last_sent_time = current_time
        cv2.imwrite(IMAGE_PATH, frame)  # ✅ Overwrite latest frame

        ai_thread = threading.Thread(target=analyze_with_gemini)
        ai_thread.daemon = True
        add_script_run_ctx(ai_thread)  # Add streamlit context to thread
        ai_thread.start()

def video_processing(video_file):
    """Reads video frames and processes them."""
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        st.error("❌ Error: Could not open video source.")
        return

    while st.session_state.running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))
        st.session_state.frame = frame.copy()
        process_frame(frame)

        time.sleep(0.03)  # Control processing speed

    cap.release()
    st.session_state.running = False
    st.info("✅ Monitoring Completed.")

def main():
    st.title("🔥 Fire/Smoke Detection System")
    st.markdown("""
    This system uses AI to detect fire or smoke in real-time video streams. 
    When detected, it automatically sends email alerts with the captured image.
    """)

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        video_source = st.selectbox(
            "Video Source",
            ["Sample Video", "Webcam", "Custom Video File"],
            index=0
        )
        
        if video_source == "Custom Video File":
            uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        else:
            uploaded_file = None

        st.header("Email Settings")
        global EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER
        EMAIL_SENDER = st.text_input("Sender Email", EMAIL_SENDER)
        EMAIL_PASSWORD = st.text_input("Sender Password", EMAIL_PASSWORD, type="password")
        EMAIL_RECEIVER = st.text_input("Receiver Email", EMAIL_RECEIVER)

        st.header("AI Settings")
        global GOOGLE_API_KEY
        GOOGLE_API_KEY = st.text_input("Gemini API Key", GOOGLE_API_KEY, type="password")
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Feed")
        video_placeholder = st.empty()

        if st.button("Start Monitoring", disabled=st.session_state.running):
            st.session_state.running = True
            st.session_state.alert_sent = False
            
            if video_source == "Sample Video":
                video_file = "fire.mp4"
            elif video_source == "Webcam":
                video_file = 0
            else:
                if uploaded_file is not None:
                    video_file = uploaded_file.name
                    with open(video_file, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                else:
                    st.warning("Please upload a video file")
                    return

            processing_thread = threading.Thread(target=video_processing, args=(video_file,))
            processing_thread.daemon = True
            add_script_run_ctx(processing_thread)
            processing_thread.start()

        if st.button("Stop Monitoring", disabled=not st.session_state.running):
            st.session_state.running = False

    with col2:
        st.subheader("Detection Alerts")
        alert_placeholder = st.empty()

        if st.session_state.last_alert:
            st.info(st.session_state.last_alert)

        if st.session_state.frame is not None:
            st.subheader("Latest Frame")
            st.image(st.session_state.frame, channels="BGR", use_column_width=True)

    # Update the video feed
    while st.session_state.running:
        if st.session_state.frame is not None:
            video_placeholder.image(st.session_state.frame, channels="BGR")
        time.sleep(0.03)

if __name__ == "__main__":
    main()