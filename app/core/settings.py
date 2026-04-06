from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    google_api_key: str
    email_password: str
    email_sender: str
    
    model_config = ConfigDict(
        env_file=".env",
        extra="ignore"
    )

settings = Settings()

