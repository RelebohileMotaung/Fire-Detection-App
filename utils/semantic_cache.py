import cv2
import hashlib
import numpy as np
from typing import Optional, Any
from cachetools import TTLCache

class SemanticFrameCache:
    """Cache AI verification results based on visual similarity"""
    
    def __init__(self, maxsize: int = 100, ttl: int = 300, similarity_threshold: float = 0.92):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.similarity_threshold = similarity_threshold
    
    def _frame_to_hash(self, frame: np.ndarray) -> str:
        """Create perceptual hash for similar frame detection"""
        # Resize to 32x32, convert to grayscale, compute average hash
        small = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        avg = gray.mean()
        bits = (gray > avg).astype(int).flatten()
        return ''.join(str(b) for b in bits)
    
    def _semantic_similarity(self, hash1: str, hash2: str) -> float:
        """Compute similarity between two frame hashes"""
        # Simple Hamming distance for perceptual hashes
        diff = sum(b1 != b2 for b1, b2 in zip(hash1, hash2))
        return 1.0 - (diff / len(hash1))
    
    def get(self, frame: np.ndarray, metrics: Optional[Any] = None) -> Optional[dict]:
        """Return cached result if similar frame found"""
        current_hash = self._frame_to_hash(frame)
        
        for cached_hash, result in self.cache.items():
            if self._semantic_similarity(current_hash, cached_hash) >= self.similarity_threshold:
                if metrics:
                    metrics.ai_verification_results.labels(
                        result='cache_hit', 
                        type='semantic_cache'
                    ).inc()
                return result
        
        return None
    
    def set(self, frame: np.ndarray, result: dict):
        """Store result with frame hash as key"""
        frame_hash = self._frame_to_hash(frame)
        self.cache[frame_hash] = result

