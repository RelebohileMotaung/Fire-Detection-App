import streamlit as st
import requests
import asyncio
import json
import os
import base64
import websockets
import threading
import queue
from datetime import datetime
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Configuration
BACKEND_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Fire/Smoke Detection System",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #f0f0f0;
    }
    .alert-box {
        padding: 1em;
        border-radius: 10px;
        margin: 1em 0;
    }
    .success-alert {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-alert {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .danger-alert {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'last_status' not in st.session_state:
    st.session_state.last_status = {}
if 'latest_frame' not in st.session_state:
    st.session_state.latest_frame = None
if 'ws_connected' not in st.session_state:
    st.session_state.ws_connected = False

# Helper functions
def get_status():
    try:
        response = requests.get(f"{BACKEND_URL}/status")
        return response.json()
    except:
        return {"running": False}

def start_monitoring(video_source, file=None):
    files = {}
    if file:
        files = {"file": file}
    
    data = {"video_source": video_source}
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/start",
            data=data,
            files=files
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def stop_monitoring():
    try:
        response = requests.post(f"{BACKEND_URL}/stop")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def test_email_settings(email_config):
    try:
        response = requests.post(
            f"{BACKEND_URL}/test_email",
            json=email_config
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_recordings():
    try:
        response = requests.get(f"{BACKEND_URL}/recordings")
        return response.json()
    except:
        return {"files": []}

# Main UI
st.title("🔥 Fire/Smoke Detection System")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Email Configuration
    st.subheader("Email Settings")
    email_sender = st.text_input("Sender Email", placeholder="your-email@gmail.com")
    email_password = st.text_input("App Password", type="password", placeholder="your-app-password")
    email_receiver = st.text_input("Receiver Email", placeholder="receiver-email@gmail.com")
    
    if st.button("Test Email Settings"):
        if email_sender and email_password and email_receiver:
            email_config = {
                "sender": email_sender,
                "password": email_password,
                "receiver": email_receiver
            }
            result = test_email_settings(email_config)
            if "success" in result:
                st.success("✅ Test email sent successfully!")
            else:
                st.error(f"❌ {result.get('error', 'Failed to send test email')}")
        else:
            st.warning("Please fill all email fields first")
    
    # AI Configuration
    st.subheader("AI Settings")
    api_key = st.text_input("Google API Key", type="password", placeholder="your-gemini-api-key")
    
    if st.button("Save Configuration"):
        if email_sender and email_password and email_receiver:
            email_config = {
                "sender": email_sender,
                "password": email_password,
                "receiver": email_receiver
            }
            requests.post(f"{BACKEND_URL}/configure/email", json=email_config)
        
        if api_key:
            ai_config = {"google_api_key": api_key}
            requests.post(f"{BACKEND_URL}/configure/ai", json=ai_config)
        
        st.success("✅ Configuration saved!")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📹 Monitoring")
    
    # Video source selection
    video_source = st.selectbox(
        "Select Video Source",
        ["sample", "webcam", "custom"],
        format_func=lambda x: {
            "sample": "Sample Video",
            "webcam": "Webcam",
            "custom": "Custom File"
        }[x]
    )
    
    uploaded_file = None
    if video_source == "custom":
        uploaded_file = st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov'])
    
    # Confidence threshold slider
    confidence = st.slider("Detection Confidence", 0.1, 0.9, 0.5, 0.05)
    
    # Control buttons
    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("🚀 Start Monitoring", key="start"):
            email_config = {
                "sender": email_sender,
                "password": email_password,
                "receiver": email_receiver
            }
            ai_config = {"google_api_key": api_key}
            
            if email_config["sender"] and email_config["password"] and email_config["receiver"]:
                requests.post(f"{BACKEND_URL}/configure/email", json=email_config)
            
            if ai_config["google_api_key"]:
                requests.post(f"{BACKEND_URL}/configure/ai", json=ai_config)
            
            requests.post(f"{BACKEND_URL}/configure/threshold", data={"threshold": confidence})
            
            result = start_monitoring(video_source, uploaded_file)
            if "error" not in result:
                st.session_state.running = True
                st.success("✅ Monitoring started!")
            else:
                st.error(f"❌ {result['error']}")
    
    with col_stop:
        if st.button("⏹️ Stop Monitoring", key="stop"):
            result = stop_monitoring()
            if "error" not in result:
                st.session_state.running = False
                st.success("✅ Monitoring stopped!")
            else:
                st.error(f"❌ {result['error']}")

    # Status display
    status = get_status()
    if status.get("running"):
        st.info("🟢 System is monitoring...")

with col2:
    st.header("📊 Status & Alerts")
    
    status = get_status()
    if status.get("last_alert"):
        st.markdown(f"**Last Alert:** {status['last_alert']}")
    
    st.subheader("🎥 Recorded Incidents")
    recordings = get_recordings()
    
    if recordings.get("files"):
        for recording in recordings["files"]:
            st.video(f"{BACKEND_URL}/recordings/{recording['filename']}")
            st.caption(f"Recorded at {recording['timestamp']}")
    else:
        st.info("No incidents recorded yet")

# Footer
st.markdown("---")
st.markdown("### 📝 System Logs")
if st.button("Refresh Logs"):
    try:
        with open("detections.log", "r") as f:
            logs = f.read()
            st.text_area("Recent Events", logs, height=200)
    except:
        st.info("No logs available yet")
