from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json

Base = declarative_base()

class Detection(Base):
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(Integer, ForeignKey("incidents.id"), index=True)
    camera_id = Column(String, index=True, default="default")
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    confidence = Column(Float)
    detection_type = Column(String)  # 'fire' or 'smoke'
    bbox = Column(JSON)  # Bounding box coordinates [x1, y1, x2, y2]
    ai_verified = Column(Boolean, default=False)
    ai_response = Column(Text)
    image_path = Column(String)
    recording_path = Column(String)
    false_positive = Column(Boolean, default=None)  # None=unlabeled, True/False=labeled
    
    # Relationship
    incident = relationship("Incident", back_populates="detections", foreign_keys=[incident_id])

class Incident(Base):
    __tablename__ = "incidents"
    
    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(Integer, ForeignKey("detections.id"), index=True)  # Links to first detection
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration_seconds = Column(Float)
    max_confidence = Column(Float)
    avg_confidence = Column(Float)
    total_detections = Column(Integer)
    recording_path = Column(String)
    status = Column(String, default="active")  # 'active', 'resolved', 'false_alarm'
    resolution_notes = Column(Text)
    
    # Relationship
    detections = relationship("Detection", foreign_keys=[Detection.incident_id], back_populates="incident")

class ModelVersion(Base):
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True, index=True)
    model_path = Column(String)
    quantization_mode = Column(String, default="none")
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=False)
    metrics = Column(JSON)  # Precision, recall, F1, etc.
    training_data_hash = Column(String)

