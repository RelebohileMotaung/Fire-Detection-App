import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Dedicated thread pool for CPU-bound CV2 operations
_cv2_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix='cv2-worker')

async def run_in_cv2_thread(func: Callable, *args, **kwargs) -> Any:
    """Run blocking cv2 operations in dedicated thread pool (non-blocking)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_cv2_executor, lambda: func(*args, **kwargs))

async def async_detect_fire_yolo(
    yolo_model, 
    frame: np.ndarray, 
    conf_threshold: float = 0.5,
    target_classes: list = None
) -> tuple[bool, list]:
    """
    Async wrapper for YOLO fire/smoke detection.
    
    Args:
        yolo_model: Loaded YOLO model (e.g., state.yolo_model)
        frame: Input frame (np.ndarray)
        conf_threshold: Confidence threshold
        target_classes: List of fire/smoke class names ['fire', 'smoke']
    
    Returns:
        (fire_detected: bool, detections: list[dict])
    """
    try:
        if target_classes is None:
            target_classes = ['fire', 'smoke']
        
        def _sync_yolo_inference():
            results = yolo_model(frame, conf=conf_threshold)
            fire_detected = False
            detections = []
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name - assumes model has .names or pass class_names
                        class_name = yolo_model.names.get(cls_id, f"class_{cls_id}")
                        
                        if class_name.lower() in [c.lower() for c in target_classes]:
                            fire_detected = True
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'class': class_name
                            })
            return fire_detected, detections
        
        return await run_in_cv2_thread(_sync_yolo_inference)
        
    except Exception as e:
        logger.error(f"YOLO async detection failed: {e}", exc_info=True)
        return False, []

