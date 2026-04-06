from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field

class Settings(BaseSettings):
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    google_api_key: str = Field(default="")
    email_password: str = Field(default="")
    email_sender: str = Field(default="")
    database_url: str = "sqlite+aiosqlite:///./fire_detection.db"
    
    model_config = ConfigDict(
        env_file=".env",
        extra="ignore"
    )

settings = Settings()

