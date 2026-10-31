"""Checkpoint identity verification against a frozen model-selection manifest.

Prevents silently loading an unexpected checkpoint (e.g. a stale legacy
model left over from an earlier pipeline). Verification is opt-in: if no
expected hash or freeze manifest is configured, the service behaves as
before (loads whatever checkpoint path is given). When one IS configured,
a hash mismatch blocks loading rather than falling back to any other file.
"""
import json
import logging
from dataclasses import dataclass
from typing import Optional

from app.components.component1.storage import compute_file_hash

logger = logging.getLogger(__name__)


def _normalize_hash(value: str) -> str:
    """Strip an optional 'sha256:' prefix and lowercase for comparison."""
    v = value.strip().lower()
    if v.startswith("sha256:"):
        v = v[len("sha256:"):]
    return v


@dataclass(frozen=True)
class ExpectedCheckpoint:
    """Resolved expected-identity info for a configured checkpoint."""
    expected_sha256: str
    source: str  # "explicit_hash" or "freeze_manifest"
    experiment_id: Optional[str] = None
    manifest_path: Optional[str] = None


class FreezeManifestError(Exception):
    """Raised when a configured freeze manifest cannot be read or parsed."""


def load_freeze_manifest(manifest_path: str) -> dict:
    try:
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise FreezeManifestError(
            f"Could not read freeze manifest at {manifest_path}: {e}"
        ) from e


def resolve_expected_checkpoint(
    expected_sha256: Optional[str] = None,
    freeze_manifest_path: Optional[str] = None,
) -> Optional[ExpectedCheckpoint]:
    """Resolve the expected checkpoint identity from explicit config and/or
    a freeze manifest.

    If only one is configured, that one is used. If BOTH are configured,
    their hashes must agree — a disagreement is a configuration error
    (raises FreezeManifestError) rather than silently preferring one over
    the other, since that could mask a stale or wrong manifest/override.

    Returns None if neither is configured — verification is then skipped
    and the caller loads whatever checkpoint path it was given (unchanged
    legacy behaviour).
    """
    manifest_expected: Optional[ExpectedCheckpoint] = None
    if freeze_manifest_path:
        manifest = load_freeze_manifest(freeze_manifest_path)
        selected = manifest.get("selected_experiment")
        if not selected or "checkpoint_sha256" not in selected:
            raise FreezeManifestError(
                f"Freeze manifest {freeze_manifest_path} missing "
                f"'selected_experiment.checkpoint_sha256'"
            )
        manifest_expected = ExpectedCheckpoint(
            expected_sha256=_normalize_hash(selected["checkpoint_sha256"]),
            source="freeze_manifest",
            experiment_id=selected.get("experiment_id"),
            manifest_path=freeze_manifest_path,
        )

    if expected_sha256 and manifest_expected:
        explicit_norm = _normalize_hash(expected_sha256)
        if explicit_norm != manifest_expected.expected_sha256:
            raise FreezeManifestError(
                "Conflicting checkpoint identity configuration: "
                "C1_MODEL_EXPECTED_SHA256 does not match "
                "selected_experiment.checkpoint_sha256 in the configured "
                f"freeze manifest ({freeze_manifest_path}). Refusing to "
                "silently prefer one over the other — fix the configuration."
            )
        return ExpectedCheckpoint(
            expected_sha256=explicit_norm,
            source="explicit_hash_and_freeze_manifest_agree",
            experiment_id=manifest_expected.experiment_id,
            manifest_path=freeze_manifest_path,
        )

    if expected_sha256:
        return ExpectedCheckpoint(
            expected_sha256=_normalize_hash(expected_sha256),
            source="explicit_hash",
        )

    return manifest_expected


@dataclass(frozen=True)
class VerificationResult:
    verified: bool
    actual_sha256: str
    expected: Optional[ExpectedCheckpoint] = None
    detail: str = ""


def verify_checkpoint(
    weights_path: str,
    expected_sha256: Optional[str] = None,
    freeze_manifest_path: Optional[str] = None,
) -> VerificationResult:
    """Compute the checkpoint's hash and compare against the resolved
    expected identity, if any is configured.

    Never raises on a mismatch — returns verified=False with a detail
    message so the caller can refuse to load without crashing the process.
    Raises FreezeManifestError only if a configured manifest cannot be
    read/parsed at all (a configuration error, not a mismatch).
    """
    actual = _normalize_hash(compute_file_hash(weights_path))
    expected = resolve_expected_checkpoint(expected_sha256, freeze_manifest_path)

    if expected is None:
        return VerificationResult(
            verified=True, actual_sha256=actual, expected=None,
            detail="No expected checkpoint identity configured — verification skipped",
        )

    if actual == expected.expected_sha256:
        return VerificationResult(
            verified=True, actual_sha256=actual, expected=expected,
            detail=f"Checkpoint identity verified against {expected.source}",
        )

    return VerificationResult(
        verified=False, actual_sha256=actual, expected=expected,
        detail=(
            f"Checkpoint hash mismatch: configured checkpoint does not match "
            f"the expected identity from {expected.source}"
            + (f" (experiment_id={expected.experiment_id})" if expected.experiment_id else "")
            + f". Expected sha256 prefix {expected.expected_sha256[:12]}..., "
            f"got {actual[:12]}..."
        ),
    )


def resolve_inspection_experiment_id(
    stored_weights_hash: Optional[str], freeze_manifest_path: Optional[str],
) -> Optional[str]:
    """Resolve identity from a full persisted digest, never the live detector.

    The configured frozen manifest is authoritative. Missing, malformed or
    unmatched evidence fails closed without preventing inspection retrieval.
    """
    import re

    if not isinstance(stored_weights_hash, str) or not freeze_manifest_path:
        return None
    digest = _normalize_hash(stored_weights_hash)
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        return None
    try:
        expected = resolve_expected_checkpoint(freeze_manifest_path=freeze_manifest_path)
    except (FreezeManifestError, TypeError, AttributeError, ValueError):
        logger.warning("Inspection identity unavailable: invalid or unreadable freeze manifest")
        return None
    if expected is None or not re.fullmatch(r"[0-9a-f]{64}", expected.expected_sha256):
        return None
    if digest != expected.expected_sha256:
        return None
    identity = expected.experiment_id
    return identity if isinstance(identity, str) and identity.strip() else None
