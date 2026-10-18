"""Environment configuration, independent of the server's working directory."""
from dataclasses import dataclass
import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    c3_model_path: Path | None = None
    c3_data_dir: Path | None = None

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
        )
