from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
import os
from .models import Base

class DatabaseManager:
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", 
            "sqlite+aiosqlite:///./fire_detection.db"
        )
        self.async_engine = create_async_engine(
            self.database_url,
            echo=True,
            future=True
        )
        self.async_session = sessionmaker(
            self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    @asynccontextmanager
    async def get_session(self):
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def create_tables(self):
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

