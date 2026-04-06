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
from utils.semantic_cache import SemanticFrameCache
from app.core.settings import settings
from app.database.database import DatabaseManager
from app.database.models import Base, Detection, Incident, ModelVersion
from sqlalchemy.future import select
from sqlalchemy import and_, update
import cv2
import numpy as np

frame_cache = SemanticFrameCache()
import torch
from ultralytics import YOLO
import asyncio
from utils.async_cv2 import run_in_cv2_thread, async_detect_fire_yolo
from quantized_yolo import QuantizedYOLO
from torch.quantization import quantize_dynamic
import smtplib
from email.message import EmailMessage
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional, List, Dict, AsyncIterator
from pydantic import BaseModel
import uvicorn
import aiofiles
from cachetools import TTLCache
import shutil
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import filetype

# Configuration constants
MAX_FILE_SIZE_MB = 100  # Maximum file size in MB
ALLOWED_VIDEO_TYPES = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm', 'm4v'}
UPLOAD_DIR = 'uploads'

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

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # STARTUP: Preload resources before accepting traffic
    logger.info("[START] Initializing Fire/Smoke Detection System...")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Preload YOLO model
    try:
        state.yolo_model = QuantizedYOLO("best.pt", quantize_mode='fp16')
        state.class_names = state.yolo_model.names
        logger.info("[OK] YOLO model loaded successfully")
        
        # Initialize database
        await db_manager.create_tables()
        logger.info("[OK] Database tables created")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load YOLO model: {e}")
        raise
        
    yield  # Server runs here
    
    # SHUTDOWN: Graceful cleanup
    logger.info("[STOP] Shutting down system...")
    state.running = False
    state.stop_recording()
    if state.gemini_model:
        del state.gemini_model
    logger.info("[OK] Cleanup complete")

app = FastAPI(title="Fire/Smoke Detection System API", lifespan=lifespan)

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
        self.current_detections = []
        self.current_incident_id = None
        self.current_recording_path = None
        
        # Prometheus metrics
        self.metrics = PrometheusMetrics()

    
    async def add_frame(self, frame):
        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)

    async def stop_recording(self):
        if not self.recording or self.video_writer is None or not self.current_incident_id:
            return
        
        duration = time.time() - self.recording_start
        self.video_writer.release()
        self.video_writer = None
        self.recording = False
        self.recording_start = None
        
        # Calculate incident metrics from detections
        async with db_manager.get_session() as session:
            incident = await session.get(Incident, self.current_incident_id)
            if incident:
                # Get all detections for this incident
                detections = await session.execute(
                    select(Detection).where(Detection.incident_id == self.current_incident_id)
                )
                detections_list = detections.scalars().all()
                
                if detections_list:
                    confidences = [d.confidence for d in detections_list if d.confidence]
                    incident.detection_id = detections_list[0].id  # First detection
                    incident.duration_seconds = duration
                    incident.max_confidence = max(confidences) if confidences else 0.0
                    incident.avg_confidence = sum(confidences)/len(confidences) if confidences else 0.0
                    incident.total_detections = len(detections_list)
                    incident.recording_path = self.current_recording_path
                    incident.end_time = datetime.now()
                    incident.status = "resolved"  # Default, can be updated via API
                    
                    session.add(incident)
                    await session.commit()
                    self.log_event("incident", f"Completed incident {self.current_incident_id}: {len(detections_list)} detections, avg conf {incident.avg_confidence:.2f}")
        
        self.metrics.recording_status.set(0)
        self.metrics.recording_duration.observe(duration)
        self.current_incident_id = None
        self.current_detections.clear()
    
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

db_manager = DatabaseManager(settings.database_url)

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
                    "text": """Analyze the image and determine if fire or smoke is present. If fire or smoke is detected, generate a complete email automatically. - Write a **clear and urgent subject** (avoid long text). - Write a **professional but urgent email body**. - **Include emergency contact numbers** for Fire Department and Ambulance (local numbers for Free State). - If no fire or smoke is detected, simply respond with 'No fire detected'."""
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



async def process_frame(frame):
    """Process a single frame for fire detection."""
    start_time = time.time()
    current_time = start_time
    
    # Update system status metrics
    state.metrics.system_status.labels(component='processing').set(1)
    
    # Run YOLO detection
    fire_detected, detections = await async_detect_fire_yolo(
        state.yolo_model, frame, state.detection_threshold, ['fire', 'smoke']
    )
    
    # Save detections to database
    if detections:
        async with db_manager.get_session() as session:
            for det in detections:
                detection = Detection(
                    camera_id="default",
                    confidence=det['confidence'],
                    detection_type=det['class'],
                    bbox=det['bbox'],
                    image_path=state.image_path,
                    recording_path=getattr(state, 'current_recording_path', None),
                    incident_id=state.current_incident_id
                )
                session.add(detection)
                state.current_detections.append(detection.id)  # Append before flush for ID if needed
            await session.flush()  # Get IDs without commit
            await session.commit()

    
    # Update detection counter
    if fire_detected:
        state.metrics.detection_counter.labels(type='yolo', source='yolo').inc()
    
    # Draw bounding boxes
    annotated_frame = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        await run_in_cv2_thread(cv2.rectangle, annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{detection['class']} {detection['confidence']:.2f}"
        await run_in_cv2_thread(cv2.putText, annotated_frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save frame for AI analysis
    await run_in_cv2_thread(cv2.imwrite, state.image_path, frame)
    
    # Broadcast frame via WebSocket
    _, buffer = await run_in_cv2_thread(cv2.imencode, '.jpg', annotated_frame)
    await manager.broadcast(json.dumps({
        "type": "frame_update",
        "data": base64.b64encode(buffer).decode('utf-8'),
        "timestamp": datetime.now().isoformat(),
        "detections": detections
    }))
    
    # Handle fire detection
    if fire_detected:
        state.verification_stats["yolo_detections"] += 1
        
        # Check cache first
        cached_result = frame_cache.get(state.frame, state.metrics)
        if cached_result:
            await manager.broadcast(json.dumps({
                "type": "alert",
                "message": cached_result["message"],
                "timestamp": datetime.now().isoformat(),
                "cached": True  # For debugging
            }))
            return {"status": "alert", "detections": detections, "ai_result": cached_result, "cached": True}
        
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
                
                # Update latest detection with AI verification
                if state.current_detections:
                    last_det_id = state.current_detections.pop()
                    async with db_manager.get_session() as session:
                        det = await session.get(Detection, last_det_id)
                        if det:
                            det.ai_verified = True
                            det.ai_response = ai_result["message"]
                            session.add(det)
                            await session.commit()
                
                # Calculate false positive rate
                total = state.verification_stats["yolo_detections"]
                confirmed = state.verification_stats["gemini_confirmations"]
                state.verification_stats["false_positive_rate"] = (
                    (total - confirmed) / total * 100 if total > 0 else 0
                )
                
                # Cache the result for future similar frames
                if ai_result.get("status") in ["success", "info"]:
                    frame_cache.set(state.frame, ai_result)
                
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
            
            frame = await run_in_cv2_thread(cv2.resize, frame, (1020, 500))
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
        # Validate file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset pointer
        
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(413, f"File too large. Max {MAX_FILE_SIZE_MB}MB allowed")
        
        # Validate file type
        content = await file.read()
        file_type = filetype.guess_extension(content)
        if file_type not in ALLOWED_VIDEO_TYPES:
            raise HTTPException(415, f"Invalid file type. Allowed: {ALLOWED_VIDEO_TYPES}")
        
        # Save to uploads dir
        upload_path = os.path.join(UPLOAD_DIR, file.filename)
        async with aiofiles.open(upload_path, "wb") as f:
            await f.write(content)
        video_file = upload_path
    else:
        raise HTTPException(status_code=400, detail="Invalid video source or missing file")
    
    asyncio.create_task(video_processing(video_file))
    return {"message": "Monitoring started successfully"}

@app.post("/stop")
async def stop_monitoring():
    state.running = False
    return {"message": "Monitoring stopped successfully"}



@app.get("/frame")
async def get_latest_frame():
    if state.frame is None:
        raise HTTPException(status_code=404, detail="No frame available")
    
    _, buffer = await run_in_cv2_thread(cv2.imencode, '.jpg', state.frame)
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

@app.get("/incidents")
async def list_incidents(status: Optional[str] = None, limit: int = 50):
    """List incidents from DB with optional status filter."""
    async with db_manager.get_session() as session:
        query = select(Incident).order_by(Incident.start_time.desc()).limit(limit)
        if status:
            query = query.where(Incident.status == status)
        result = await session.execute(query)
        incidents = result.scalars().all()
        return {
            "incidents": [
                {
                    "id": i.id,
                    "start_time": i.start_time.isoformat() if i.start_time else None,
                    "end_time": i.end_time.isoformat() if i.end_time else None,
                    "duration_seconds": getattr(i, 'duration_seconds', 0),
                    "max_confidence": getattr(i, 'max_confidence', 0),
                    "avg_confidence": getattr(i, 'avg_confidence', 0),
                    "total_detections": getattr(i, 'total_detections', 0),
                    "recording_path": i.recording_path,
                    "status": i.status,
                    "resolution_notes": i.resolution_notes
                } for i in incidents
            ]
        }

@app.get("/detections")
async def list_detections(incident_id: Optional[int] = None, limit: int = 100):
    """List recent detections."""
    async with db_manager.get_session() as session:
        query = select(Detection).order_by(Detection.timestamp.desc()).limit(limit)
        if incident_id:
            query = query.where(Detection.incident_id == incident_id)
        result = await session.execute(query)
        detections = result.scalars().all()
        return {
            "detections": [
                {
                    "id": d.id,
                    "incident_id": d.incident_id,
                    "timestamp": d.timestamp.isoformat() if d.timestamp else None,
                    "confidence": d.confidence,
                    "detection_type": d.detection_type,
                    "bbox": d.bbox,
                    "ai_verified": d.ai_verified,
                    "false_positive": d.false_positive
                } for d in detections
            ]
        }

@app.get("/model_versions")
async def list_model_versions(active_only: bool = True):
    """List model versions."""
    async with db_manager.get_session() as session:
        query = select(ModelVersion).order_by(ModelVersion.created_at.desc())
        if active_only:
            query = query.where(ModelVersion.is_active == True)
        result = await session.execute(query)
        versions = result.scalars().all()
        return {
            "model_versions": [
                {
                    "id": m.id,
                    "version": m.version,
                    "model_path": m.model_path,
                    "quantization_mode": m.quantization_mode,
                    "is_active": m.is_active,
                    "metrics": m.metrics
                } for m in versions
            ]
        }

@app.post("/incidents/{incident_id}/resolve")
async def resolve_incident(incident_id: int, is_false_positive: bool = Form(False), notes: str = Form("")):
    """Mark incident as resolved/false alarm, label detections."""
    async with db_manager.get_session() as session:
        incident = await session.get(Incident, incident_id)
        if not incident:
            raise HTTPException(404, "Incident not found")
        
        incident.status = "false_alarm" if is_false_positive else "resolved"
        incident.resolution_notes = notes
        
        # Label all detections as false positive
        if is_false_positive:
            await session.execute(
                update(Detection)
                .where(Detection.incident_id == incident_id)
                .values(false_positive=True)
            )
        
        await session.commit()
        return {"message": f"Incident {incident_id} marked as {'false_alarm' if is_false_positive else 'resolved'}"}


@app.get("/recordings")
async def list_recordings():
    """Backward compat: List recordings, enhanced with incident data."""
    return await list_incidents()

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
    
    old_mode = getattr(state.yolo_model, 'quantize_mode', 'none') if state.yolo_model else 'none'
    
    if mode == 'none':
        state.yolo_model = YOLO("best.pt")
    else:
        state.yolo_model = QuantizedYOLO("best.pt", quantize_mode=mode)
    
    state.class_names = state.yolo_model.names
    
    # Log model version
    async with db_manager.get_session() as session:
        version = ModelVersion(
            version=f"best.pt-{mode}",
            model_path="best.pt",
            quantization_mode=mode,
            is_active=True,
            metrics={"old_mode": old_mode}
        )
        # Deactivate previous
        await session.execute(update(ModelVersion).where(ModelVersion.is_active == True).values(is_active=False))
        session.add(version)
        await session.commit()
    
    return {"message": f"Quantization mode set to {mode}, ModelVersion logged"}

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

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker HEALTHCHECK"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "fire-detection-api",
        "version": "1.0.0"
    }

import signal
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

shutdown_event = asyncio.Event()



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
