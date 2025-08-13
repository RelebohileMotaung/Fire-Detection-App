import os
import time
import base64
import asyncio
import logging
import io
import json
import hashlib
from logging.handlers import RotatingFileHandler
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from starlette_prometheus import metrics, PrometheusMiddleware
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torch.quantization import quantize_dynamic
import smtplib
from email.message import EmailMessage
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional, List, Dict
from pydantic import BaseModel
import uvicorn
import aiofiles
from cachetools import TTLCache
import shutil
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

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

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics)

# Prometheus metrics
class PrometheusMetrics:
    def __init__(self):
        # Detection counters
        self.detection_counter = Counter(
            'fire_detections_total',
            'Total number of fire detections',
            ['type', 'source']
        )
        
        # Processing time histogram
        self.frame_processing_time = Histogram(
            'frame_processing_seconds',
            'Time taken to process each frame'
        )
        
        # System status gauges
        self.system_status = Gauge(
            'system_status',
            'Current system status',
            ['component']
        )
        
        # Alert status gauges
        self.alert_status = Gauge(
            'alert_status',
            'Current alert status',
            ['type']
        )
        
        # Recording status
        self.recording_status = Gauge(
            'recording_status',
            'Recording status (1=recording, 0=not recording)'
        )
        
        # Video source status
        self.video_source_status = Gauge(
            'video_source_status',
            'Video source connection status'
        )
        
        # WebSocket connections
        self.websocket_connections = Gauge(
            'websocket_connections_total',
            'Number of active WebSocket connections'
        )
        
        # Email alerts sent
        self.email_alerts_sent = Counter(
            'email_alerts_sent_total',
            'Total number of email alerts sent'
        )
        
        # AI verification results
        self.ai_verification_results = Counter(
            'ai_verification_results_total',
            'AI verification results',
            ['result', 'type']
        )
        
        # Frame processing errors
        self.frame_processing_errors = Counter(
            'frame_processing_errors_total',
            'Total number of frame processing errors',
            ['error_type']
        )
        
        # Recording duration
        self.recording_duration = Histogram(
            'recording_duration_seconds',
            'Duration of incident recordings'
        )

# Configuration models
class EmailConfig(BaseModel):
    sender: str
    password: str
    receiver: str

class AIConfig(BaseModel):
    google_api_key: str

# Global state
class QuantizedYOLO:
    def __init__(self, model_path, quantize_mode='fp16'):
        self.original_model = YOLO(model_path)
        self.quantize_mode = quantize_mode
        self.model = self._prepare_quantized_model()
        
    def _prepare_quantized_model(self):
        model = self.original_model.model
        model.eval()
        
        if self.quantize_mode == 'fp16':
            model = model.half()
        elif self.quantize_mode == 'int8':
            model = quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
        return model
        
    def __call__(self, *args, **kwargs):
        # Convert input to right dtype
        if self.quantize_mode == 'fp16':
            kwargs['imgsz'] = kwargs.get('imgsz', 640)
            kwargs['half'] = True
        return self.original_model(*args, **kwargs)

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
        self.ai_cache = TTLCache(maxsize=1000, ttl=300)
        self.detection_threshold = 0.5  # YOLO confidence threshold
        self.last_fire_time = 0
        self.verification_stats = {
            "yolo_detections": 0,
            "gemini_confirmations": 0,
            "false_positive_rate": 0.0
        }
        self.benchmark_stats = {
            'fp32': {'fps': 0, 'latency': 0},
            'fp16': {'fps': 0, 'latency': 0},
            'int8': {'fps': 0, 'latency': 0}
        }
        self.yolo_model = QuantizedYOLO("best.pt", quantize_mode='fp16')
        self.class_names = self.yolo_model.original_model.model.names
        os.makedirs(self.recording_dir, exist_ok=True)
        
        # Model management and evaluation
        self.test_dataset_path = "test_dataset"
        self.model_versions = {
            "current": "best.pt",
            "candidate": None  # Path to candidate model
        }
        self.evaluation_metrics = {
            "precision": 0,
            "recall": 0,
            "f1": 0
        }
        os.makedirs("model_versions", exist_ok=True)
        os.makedirs("active_learning", exist_ok=True)
        
        # Active learning
        self.active_learning_threshold = 0.4  # Confidence threshold for uncertain cases
        self.active_learning_dir = "active_learning"
        
        # Prometheus metrics
        self.metrics = PrometheusMetrics()
        
    def log_event(self, event_type: str, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {event_type.upper()}: {message}\n"
        with open(self.log_file, "a") as f:
            f.write(log_entry)
        logger.info(f"{event_type}: {message}")

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        state.metrics.websocket_connections.set(len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            state.metrics.websocket_connections.set(len(self.active_connections))

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
async def send_email_alert(subject: str, body: str, image_path: str = None):
    """Sends an AI-generated email alert with the detected fire/smoke image."""
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = state.email_config.sender
        msg["To"] = state.email_config.receiver
        msg.set_content(body)
        
        if image_path and os.path.exists(image_path):
            async with aiofiles.open(image_path, "rb") as img_file:
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
        state.last_alert = f"AI Email Alert Sent at {datetime.now().strftime('%H:%M:%S')} - {subject}"
        state.metrics.email_alerts_sent.inc()
        return {"success": "AI Email Alert Sent Successfully!"}
    
    except Exception as e:
        state.metrics.frame_processing_errors.labels(error_type='email_send').inc()
        raise HTTPException(status_code=500, detail=f"Failed to send email: {e}")

async def analyze_with_gemini(image_path: str):
    """Sends the image to Gemini AI for fire/smoke detection."""
    if not os.path.exists(image_path):
        return {"status": "warning", "message": "No image available for analysis"}
    
    try:
        if not state.ai_config.google_api_key:
            return {"status": "error", "message": "Gemini API key not configured"}

        async with aiofiles.open(image_path, "rb") as img_file:
            img_data = await img_file.read()
            base64_image = base64.b64encode(img_data).decode("utf-8")
        
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Analyze this image for fire or smoke. If detected, provide a clear alert message including what you see. If no fire/smoke, respond with 'No fire detected'."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        )
        
        if state.gemini_model is None:
            state.gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        
        response = await asyncio.wait_for(
            state.gemini_model.ainvoke([message]),
            timeout=10.0
        )
        
        result = response.content.strip()
        
        if "No fire detected" in result:
            state.metrics.ai_verification_results.labels(
                result='no_fire', 
                type='ai_verification'
            ).inc()
            return {"status": "info", "message": "No fire detected"}
        
        state.metrics.ai_verification_results.labels(
            result='fire_detected', 
            type='ai_verification'
        ).inc()
        return {"status": "success", "message": result}
    
    except asyncio.TimeoutError:
        state.metrics.ai_verification_results.labels(
            result='timeout', 
            type='ai_verification'
        ).inc()
        return {"status": "error", "message": "Gemini API timeout"}
    except Exception as e:
        state.metrics.ai_verification_results.labels(
            result='error', 
            type='ai_verification'
        ).inc()
        return {"status": "error", "message": f"AI analysis failed: {str(e)}"}

async def detect_fire_yolo(frame):
    """Run YOLO fire detection on the frame."""
    try:
        results = state.yolo_model(frame, conf=state.detection_threshold)
        fire_detected = False
        detections = []
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    class_name = state.class_names[cls]
                    
                    if class_name.lower() in ['fire', 'smoke']:
                        fire_detected = True
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class': class_name
                        })
        
        return fire_detected, detections
    except Exception as e:
        state.metrics.frame_processing_errors.labels(error_type='yolo_detection').inc()
        logger.error(f"YOLO detection error: {e}")
        return False, []

async def process_frame(frame):
    """Process a single frame for fire detection."""
    start_time = time.time()
    current_time = start_time
    
    # Update system status metrics
    state.metrics.system_status.labels(component='processing').set(1)
    
    # Run YOLO detection
    fire_detected, detections = await detect_fire_yolo(frame)
    
    # Update detection counter
    if fire_detected:
        state.metrics.detection_counter.labels(type='yolo', source='yolo').inc()
    
    # Draw bounding boxes
    annotated_frame = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{detection['class']} {detection['confidence']:.2f}"
        cv2.putText(annotated_frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save frame for AI analysis
    cv2.imwrite(state.image_path, frame)
    
    # Broadcast frame via WebSocket
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    await manager.broadcast(json.dumps({
        "type": "frame_update",
        "data": base64.b64encode(buffer).decode('utf-8'),
        "timestamp": datetime.now().isoformat(),
        "detections": detections
    }))
    
    # Handle fire detection
    if fire_detected:
        state.verification_stats["yolo_detections"] += 1
        
        # Check cooldown
        if current_time - state.last_sent_time >= state.send_interval:
            state.last_sent_time = current_time
            
            # AI verification
            ai_result = await analyze_with_gemini(state.image_path)
            
            # Update AI verification metrics
            if ai_result.get("status") == "success":
                state.metrics.detection_counter.labels(type='verified', source='ai').inc()
                state.metrics.ai_verification_results.labels(
                    result='fire_detected', 
                    type='ai_verification'
                ).inc()
                state.verification_stats["gemini_confirmations"] += 1
                
                # Calculate false positive rate
                total = state.verification_stats["yolo_detections"]
                confirmed = state.verification_stats["gemini_confirmations"]
                state.verification_stats["false_positive_rate"] = (
                    (total - confirmed) / total * 100 if total > 0 else 0
                )
                
                # Send email alert
                email_result = await send_email_alert(
                    subject="Fire/Smoke Alert!",
                    body=ai_result["message"],
                    image_path=state.image_path
                )
                
                # Start recording
                if not state.recording:
                    state.start_recording(frame)
                    state.metrics.recording_status.set(1)
                
                # Broadcast alert
                await manager.broadcast(json.dumps({
                    "type": "alert",
                    "message": ai_result["message"],
                    "timestamp": datetime.now().isoformat()
                }))
                
                # Update alert status metrics
                state.metrics.alert_status.labels(type='fire').set(1)
                
                state.log_event("alert", f"Fire detected: {ai_result['message']}")
                
            elif ai_result.get("status") == "info":
                state.metrics.ai_verification_results.labels(
                    result='no_fire', 
                    type='ai_verification'
                ).inc()
                
            return {
                "status": "alert",
                "detections": detections,
                "ai_result": ai_result
            }
    
    # Update alert status when no fire detected
    state.metrics.alert_status.labels(type='fire').set(0)
    
    # Stop recording if no fire for 10 seconds
    if state.recording and current_time - state.last_sent_time > 10:
        state.stop_recording()
        state.metrics.recording_status.set(0)
    
    # Update recording status
    state.metrics.recording_status.set(1 if state.recording else 0)
    
    # Update frame processing time
    processing_time = time.time() - start_time
    state.metrics.frame_processing_time.observe(processing_time)
    
    return {"status": "clear", "detections": detections}

# Add recording methods to State class
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
    self.metrics.recording_status.set(1)
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
        self.metrics.recording_status.set(0)
        self.metrics.recording_duration.observe(duration)
        self.log_event("recording", 
                      f"Stopped recording (duration: {duration:.2f}s)")

# Add methods to State class
State.start_recording = start_recording
State.add_frame = add_frame
State.stop_recording = stop_recording

async def video_processing(video_source):
    """Main video processing loop."""
    try:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            state.metrics.frame_processing_errors.labels(error_type='video_source').inc()
            raise HTTPException(status_code=400, detail="Could not open video source")
        
        state.metrics.video_source_status.set(1)
        
        while state.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                state.metrics.frame_processing_errors.labels(error_type='frame_read').inc()
                break
            
            frame = cv2.resize(frame, (1020, 500))
            state.frame = frame.copy()
            
            await process_frame(frame)
            await asyncio.sleep(0.03)
            
        cap.release()
        state.running = False
        state.stop_recording()
        state.metrics.video_source_status.set(0)
        return {"info": "Monitoring completed"}
    
    except Exception as e:
        state.running = False
        state.stop_recording()
        state.metrics.frame_processing_errors.labels(error_type='video_processing').inc()
        state.metrics.video_source_status.set(0)
        raise HTTPException(status_code=500, detail=f"Video processing error: {e}")

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
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
async def start_monitoring(
    video_source: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
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
        "frame_available": state.frame is not None,
        "verification_stats": state.verification_stats,
        "detection_threshold": state.detection_threshold
    }

@app.get("/frame")
async def get_latest_frame():
    if state.frame is None:
        raise HTTPException(status_code=404, detail="No frame available")
    
    _, buffer = cv2.imencode('.jpg', state.frame)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

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
        raise HTTPException(status_code=400, detail=f"Failed to send test email: {str(e)}")

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

@app.post("/configure/threshold")
async def configure_threshold(threshold: float = Form(...)):
    """Configure YOLO detection confidence threshold."""
    if 0.1 <= threshold <= 0.9:
        state.detection_threshold = threshold
        return {"message": f"Detection threshold updated to {threshold}"}
    else:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.1 and 0.9")

@app.post("/configure/quantization")
async def configure_quantization(mode: str = Form(...)):
    """Configure model quantization mode (none, fp16, int8)"""
    valid_modes = ['none', 'fp16', 'int8']
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Must be one of {valid_modes}"
        )
    
    if mode == 'none':
        state.yolo_model = YOLO("best.pt")
    else:
        state.yolo_model = QuantizedYOLO("best.pt", quantize_mode=mode)
    
    state.class_names = state.yolo_model.original_model.model.names
    return {"message": f"Quantization mode set to {mode}"}

@app.get("/status")
async def get_status():
    return {
        "running": state.running,
        "alert_sent": state.alert_sent,
        "last_alert": state.last_alert,
        "frame_available": state.frame is not None,
        "verification_stats": state.verification_stats,
        "detection_threshold": state.detection_threshold,
        "quantization_mode": getattr(state.yolo_model, 'quantize_mode', 'none')
    }

import signal
import sys

shutdown_event = asyncio.Event()

@app.on_event("startup")
async def startup_event():
    # Initialize resources if needed
    pass

@app.on_event("shutdown")
async def shutdown_event_handler():
    # Properly close async resources here
    # Wait for gRPC async cleanup if applicable
    try:
        # If you have any grpc aio channels or servers, close them here
        # Example: await grpc_server.stop(0)
        pass
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

def handle_exit(sig, frame):
    logger.info(f"Received exit signal {sig.name}, shutting down...")
    loop = asyncio.get_event_loop()
    loop.create_task(app.shutdown())
    loop.create_task(app.router.shutdown())
    loop.create_task(shutdown_event.set())

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
