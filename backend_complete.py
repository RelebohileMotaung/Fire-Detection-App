import os
import time
import base64
import asyncio
import logging
import io
import json
from logging.handlers import RotatingFileHandler
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import smtplib
from email.message import EmailMessage
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional, List
from pydantic import BaseModel
import uvicorn
import aiofiles

# Configure logging
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        RotatingFileHandler("app.log", maxBytes=1000000, backupCount=3),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fire/Smoke Detection System API")

# CORS configuration (to allow Streamlit frontend to communicate)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration models
class EmailConfig(BaseModel):
    sender: str
    password: str
    receiver: str

class AIConfig(BaseModel):
    google_api_key: str

# Global state
class State:
    def __init__(self):
        self.running = False
        self.frame = None
        self.alert_sent = False
        self.last_alert = ""
        self.last_sent_time = 0
        self.send_interval = 5  # seconds
        self.image_path = "latest_frame.jpg"
        self.email_config = EmailConfig(
            sender="",
            password="",
            receiver=""
        )
        self.ai_config = AIConfig(google_api_key="")
        self.gemini_model = None
        self.log_file = "detections.log"
        self.recording = False
        self.video_writer = None
        self.recording_start = None
        self.recording_dir = "recordings"
        os.makedirs(self.recording_dir, exist_ok=True)
        
    def log_event(self, event_type: str, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {event_type.upper()}: {message}\n"
        with open(self.log_file, "a") as f:
            f.write(log_entry)
        logger.info(f"{event_type}: {message}")
        
    def start_recording(self, frame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.recording_dir}/incident_{timestamp}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(
            filename,
            fourcc,
            20.0,
            (frame.shape[1], frame.shape[0])
        )
        self.recording = True
        self.recording_start = time.time()
        self.log_event("recording", f"Started recording: {filename}")
        
    def add_frame(self, frame):
        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)
            
    def stop_recording(self):
        if self.recording and self.video_writer is not None:
            duration = time.time() - self.recording_start
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            self.log_event("recording", 
                          f"Stopped recording (duration: {duration:.2f}s)")

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        disconnected_connections = []
        for connection in self.active_connections:
            try:
                if connection.client_state == WebSocketState.CONNECTED:
                    await connection.send_text(message)
                else:
                    disconnected_connections.append(connection)
            except Exception as e:
                logger.error(f"WebSocket broadcast error: {e}")
                disconnected_connections.append(connection)
        
        # Clean up disconnected connections
        for conn in disconnected_connections:
            if conn in self.active_connections:
                self.active_connections.remove(conn)

manager = ConnectionManager()

state = State()

# Email and AI functions
async def send_email_alert(subject: str, body: str):
    """Sends an AI-generated email alert with the detected fire/smoke image."""
    if not os.path.exists(state.image_path):
        return {"warning": "Email alert skipped: No image file found."}
    
    try:
        subject = subject.replace("\n", " ").replace("\r", "").strip()
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = state.email_config.sender
        msg["To"] = state.email_config.receiver
        msg.set_content(body)
        
        async with aiofiles.open(state.image_path, "rb") as img_file:
            img_data = await img_file.read()
            msg.add_attachment(
                img_data,
                maintype="image",
                subtype="jpeg",
                filename="fire_alert.jpg"
            )
        
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(state.email_config.sender, state.email_config.password)
            server.send_message(msg)
        
        state.alert_sent = True
        state.last_alert = f"AI Email Alert Sent at {time.strftime('%H:%M:%S')} - {subject}"
        return {"success": "AI Email Alert Sent Successfully!"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {e}")

async def analyze_with_gemini():
    """Sends the latest frame to Gemini AI with proper error handling."""
    if not os.path.exists(state.image_path):
        return {"status": "warning", "message": "No image available for analysis. Skipping AI check."}
    
    try:
        # Check if API key is configured
        if not state.ai_config.google_api_key:
            return {"status": "error", "message": "Gemini API key not configured."}

        async with aiofiles.open(state.image_path, "rb") as img_file:
            img_data = await img_file.read()
            base64_image = base64.b64encode(img_data).decode("utf-8")
        
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Analyze the image for fire or smoke. If detected, provide a clear alert message. If no fire/smoke, respond with 'No fire detected'."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        )
        
        if state.gemini_model is None:
            state.gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        
        # Add timeout and retry for Gemini
        try:
            response = await asyncio.wait_for(
                state.gemini_model.ainvoke([message]),
                timeout=10.0  # 10-second timeout
            )
        except asyncio.TimeoutError:
            return {"status": "error", "message": "Gemini API timeout. Try again later."}
        
        result = response.content.strip()
        
        if "No fire detected" in result:
            return {"status": "info", "message": "No fire detected."}
        
        # Send email if fire detected
        email_result = await send_email_alert(
            subject="Fire Alert!",
            body=result
        )
        
        if "error" in email_result.get("status", ""):
            return email_result
        
        return {"status": "success", "message": result}
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"AI analysis failed: {str(e)}",
            "details": "The system will continue monitoring without AI."
        }

async def process_frame(frame):
    """Saves the latest frame at intervals and starts AI analysis."""
    current_time = time.time()
    
    # Broadcast frame update via WebSocket
    _, buffer = cv2.imencode('.jpg', frame)
    await manager.broadcast(json.dumps({
        "type": "frame_update",
        "data": base64.b64encode(buffer).decode('utf-8'),
        "timestamp": datetime.now().isoformat()
    }))
    
    # Handle AI analysis at intervals
    if current_time - state.last_sent_time >= state.send_interval:
        state.last_sent_time = current_time
        cv2.imwrite(state.image_path, frame)
        ai_result = await analyze_with_gemini()
        # Send update via WebSocket
        await manager.broadcast(json.dumps({
            "type": "analysis_update",
            "data": ai_result,
            "timestamp": datetime.now().isoformat()
        }))
        return ai_result

async def video_processing(video_file: str):
    """Reads video frames and processes them."""
    try:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video source.")
        
        while state.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (1020, 500))
            state.frame = frame.copy()
            await process_frame(frame)
            await asyncio.sleep(0.03)
            
        cap.release()
        state.running = False
        return {"info": "Monitoring completed."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing error: {e}")

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            # Echo back any received messages (for testing)
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# API endpoints
@app.post("/configure/email")
async def configure_email(email_config: EmailConfig):
    state.email_config = email_config
    return {"message": "Email configuration updated successfully"}

@app.post("/configure/ai")
async def configure_ai(ai_config: AIConfig):
    state.ai_config = ai_config
    os.environ["GOOGLE_API_KEY"] = state.ai_config.google_api_key
    state.gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    return {"message": "AI configuration updated successfully"}

@app.post("/start")
async def start_monitoring(video_source: str = Form(...), file: Optional[UploadFile] = File(None)):
    if state.running:
        return {"warning": "Monitoring is already running"}
    
    state.running = True
    state.alert_sent = False
    
    video_file = ""
    if video_source == "sample":
        video_file = "fire.mp4"
    elif video_source == "webcam":
        video_file = 0
    elif video_source == "custom" and file:
        video_file = file.filename
        async with aiofiles.open(video_file, "wb") as f:
            await f.write(await file.read())
    else:
        raise HTTPException(status_code=400, detail="Invalid video source or missing file")
    
    asyncio.create_task(video_processing(video_file))
    return {"message": "Monitoring started successfully"}

@app.post("/stop")
async def stop_monitoring():
    state.running = False
    return {"message": "Monitoring stopped successfully"}

@app.get("/status")
async def get_status():
    return {
        "running": state.running,
        "alert_sent": state.alert_sent,
        "last_alert": state.last_alert,
        "frame_available": state.frame is not None
    }

@app.get("/frame")
async def get_latest_frame():
    if state.frame is None:
        raise HTTPException(status_code=404, detail="No frame available")
    
    _, buffer = cv2.imencode('.jpg', state.frame)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

# New endpoints for requested features
@app.post("/test_email")
async def test_email(config: EmailConfig):
    """Test email configuration by sending a test email."""
    try:
        msg = EmailMessage()
        msg["Subject"] = "Fire Detection System Test"
        msg["From"] = config.sender
        msg["To"] = config.receiver
        msg.set_content("This is a test email from the Fire Detection System.")
        
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(config.sender, config.password)
            server.send_message(msg)
        
        state.log_event("email_test", "Test email sent successfully")
        return {"status": "success", "message": "Test email sent successfully"}
    except Exception as e:
        state.log_event("error", f"Email test failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to send test email: {str(e)}"
        )

@app.get("/recordings")
async def list_recordings():
    """List all recorded incident videos."""
    recordings = []
    try:
        for filename in os.listdir(state.recording_dir):
            if filename.endswith(".avi"):
                filepath = os.path.join(state.recording_dir, filename)
                timestamp = datetime.fromtimestamp(
                    os.path.getctime(filepath)
                ).strftime("%Y-%m-%d %H:%M:%S")
                recordings.append({
                    "filename": filename,
                    "timestamp": timestamp,
                    "url": f"/recordings/{filename}"
                })
    except Exception as e:
        logger.error(f"Failed to list recordings: {str(e)}")
    return {"files": recordings}

@app.get("/recordings/{filename}")
async def get_recording(filename: str):
    """Serve a specific recording file."""
    filepath = os.path.join(state.recording_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Recording not found")
    return FileResponse(filepath)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
