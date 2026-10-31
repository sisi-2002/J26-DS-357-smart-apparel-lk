"""Component configuration with each component's existing environment contract."""
from dataclasses import dataclass
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class ApplicationSettings(BaseSettings):
    """Component 1 and database settings, including its existing .env support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATABASE_URL: str | None = None
    C1_MODEL_WEIGHTS: str = ""
    C1_MODEL_DEVICE: str = "cpu"
    C1_STORAGE_DIR: str = "./storage"
    C1_AUTH_DISABLED: bool = True
    # If both identity checks are configured, their hashes must agree.
    C1_MODEL_FREEZE_MANIFEST: str | None = None
    C1_MODEL_EXPECTED_SHA256: str | None = None


@dataclass(frozen=True)
class Settings:
    """Component 3's immutable, injectable settings; shell variables only.

    Keep this dataclass API for research callers and dataclasses.replace().
    Relative C3 paths always resolve from the repository root.
    """

    c3_model_path: Path | None = None
    c3_data_dir: Path | None = None
    c3_experimental_multihorizon: bool = False
    c3_experimental_model_path: Path | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        def configured_path(name: str) -> Path | None:
            value = os.environ.get(name, "").strip()
            if not value:
                return None
            path = Path(value).expanduser()
            return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()

        return cls(
            c3_model_path=configured_path("C3_MODEL_PATH"),
            c3_data_dir=configured_path("C3_DATA_DIR"),
            c3_experimental_multihorizon=os.environ.get(
                "C3_EXPERIMENTAL_MULTIHORIZON", ""
            ).strip().lower() == "true",
            c3_experimental_model_path=configured_path("C3_EXPERIMENTAL_MODEL_PATH"),
        )


# Keep the singleton used by Component 1 endpoints, storage, and DB sessions.
settings = ApplicationSettings()
