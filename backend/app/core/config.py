"""Application configuration from environment variables."""
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATABASE_URL: Optional[str] = None

    C1_MODEL_WEIGHTS: str = ""
    C1_MODEL_DEVICE: str = "cpu"
    C1_STORAGE_DIR: str = "./storage"
    C1_AUTH_DISABLED: bool = True

    # Checkpoint identity verification (opt-in — see model_identity.py).
    # If set, C1_MODEL_WEIGHTS is hashed and compared before loading;
    # a mismatch blocks loading rather than silently using the file anyway.
    C1_MODEL_FREEZE_MANIFEST: Optional[str] = None
    C1_MODEL_EXPECTED_SHA256: Optional[str] = None


settings = Settings()
