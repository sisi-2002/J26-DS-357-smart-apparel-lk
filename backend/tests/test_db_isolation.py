"""Proves the test-database isolation guard (conftest.py) actually blocks
unsafe configurations, rather than merely happening not to fire.

Root cause this guards against: `settings` is a module-level singleton
read from `.env` at import time (DATABASE_URL -> shared dev Postgres).
Without isolation, any test using the real `app.main.app` silently writes
into that shared database. These tests exercise the guard directly.
"""
import os
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import app.db.session as db_session_module
from app.core.config import settings
from tests.conftest import UnsafeTestDatabaseError, _assert_safe_test_database_url


DEV_POSTGRES_URL = "postgresql+asyncpg://smart_apparel_dev:1234@localhost:5432/smart_apparel_dev"


class TestResolvedDatabaseIsSafeByDefault:
    """Positive checks: the autouse conftest fixture already ran for this
    test — prove what it actually resolved to."""

    def test_settings_database_url_is_sqlite(self):
        assert settings.DATABASE_URL.startswith("sqlite+aiosqlite:///")

    def test_settings_database_url_is_not_dev_postgres(self):
        assert settings.DATABASE_URL != DEV_POSTGRES_URL
        assert "smart_apparel_dev" not in settings.DATABASE_URL

    def test_settings_database_url_passes_the_guard(self):
        _assert_safe_test_database_url(settings.DATABASE_URL)  # must not raise

    def test_init_engine_is_wrapped_with_guard(self):
        assert db_session_module.init_engine.__name__ == "_guarded_init_engine"


class TestGuardActuallyBlocksUnsafeTargets:
    """Negative checks: directly prove the guard raises for every
    disallowed category, not just that it stayed quiet in the happy path."""

    def test_rejects_real_dev_postgres_url(self):
        with pytest.raises(UnsafeTestDatabaseError, match="postgresql"):
            _assert_safe_test_database_url(DEV_POSTGRES_URL)

    def test_rejects_any_postgres_url_regardless_of_database_name(self):
        """'Do not rely only on a database name prefix': a Postgres URL
        named to *sound* disposable must still be rejected, because the
        backend itself (not the name) is the discriminator."""
        with pytest.raises(UnsafeTestDatabaseError):
            _assert_safe_test_database_url(
                "postgresql+asyncpg://user:pw@localhost:5432/totally_disposable_test_db"
            )

    def test_rejects_in_memory_sqlite(self):
        with pytest.raises(UnsafeTestDatabaseError, match="in-memory|anonymous"):
            _assert_safe_test_database_url("sqlite+aiosqlite://")

    def test_rejects_explicit_memory_marker(self):
        with pytest.raises(UnsafeTestDatabaseError):
            _assert_safe_test_database_url("sqlite+aiosqlite:///:memory:")

    def test_rejects_sqlite_path_outside_temp_dir(self):
        outside_path = os.path.join(
            os.path.dirname(BACKEND_DIR), "not_a_temp_dir", "db.sqlite3"
        )
        with pytest.raises(UnsafeTestDatabaseError, match="temp"):
            _assert_safe_test_database_url(f"sqlite+aiosqlite:///{outside_path}")

    def test_rejects_empty_url(self):
        with pytest.raises(UnsafeTestDatabaseError):
            _assert_safe_test_database_url("")

    def test_rejects_mysql(self):
        with pytest.raises(UnsafeTestDatabaseError, match="mysql"):
            _assert_safe_test_database_url("mysql+aiomysql://user:pw@localhost/db")

    def test_guarded_init_engine_blocks_even_if_settings_reverted(self, monkeypatch):
        """The critical fail-closed property: even if something resets
        settings.DATABASE_URL back to the real dev URL mid-test, the next
        call to init_engine() must still refuse rather than connecting."""
        monkeypatch.setattr(settings, "DATABASE_URL", DEV_POSTGRES_URL)
        with pytest.raises(UnsafeTestDatabaseError):
            db_session_module.init_engine()
        # And it must not have left a real engine bound either.
        assert db_session_module.engine is None or (
            "sqlite" in str(db_session_module.engine.url)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
