from fastapi import Request, Response
from fastapi.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import time
import logging
from .settings import settings

logger = logging.getLogger(__name__)

# Rate Limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[
        f"{settings.rate_limit_requests}/{settings.rate_limit_window} seconds"
    ]
)

class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Security headers
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Log request
        process_time = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        return response

class InputGuardrailMiddleware(BaseHTTPMiddleware):
    """Validate inputs for prompt injection and malicious content"""
    
    async def dispatch(self, request: Request, call_next):
        # Skip for static files and docs
        if any(path in request.url.path for path in ["/docs", "/redoc", "/openapi.json", "/metrics"]):
            return await call_next(request)
        
        # Check for prompt injection patterns in POST/PUT
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                body_str = body.decode('utf-8', errors='ignore')
                
                # Expanded injection patterns
                injection_patterns = [
                    "ignore previous instructions",
                    "ignore all previous",
                    "system prompt",
                    "you are now",
                    "bypass security",
                    "admin access",
                    "root access",
                    "delete from",
                    "drop table",
                    "union select",
                    "<script",
                    "javascript:",
                    "data:text/html"
                ]
                
                for pattern in injection_patterns:
                    if pattern.lower() in body_str.lower():
                        logger.warning(f"Potential injection attempt blocked: {pattern} in {request.url.path}")
                        return Response(
                            content='{"detail": "Invalid input detected"}',
                            status_code=400,
                            media_type="application/json"
                        )
            except Exception as e:
                logger.error(f"Input validation error: {e}")
        
        return await call_next(request)

def get_middleware():
    return [
        Middleware(SlowAPIMiddleware),
        Middleware(InputGuardrailMiddleware),
        Middleware(SecurityMiddleware),
    ]

