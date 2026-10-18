"""Batch the existing strict-past research features; never fit or create labels."""
import numpy as np
import pandas as pd

from .failure_inference import get_research_inference
from .failure_model_loader import InferenceResources


class IncompleteBenchmarkError(Exception):
    """A 100-machine benchmark ranking cannot be produced from this source."""


def score_machines(resources: InferenceResources, prediction_time: str) -> list[tuple[int, float]]:
    expected_ids = frozenset(range(1, 101))
    # Check cached IDs AND current metadata; never silently rank a partial set.
    machine_ids = resources.sources[0].machineID.tolist()
    if (resources.machine_ids != expected_ids or len(machine_ids) != 100
            or set(machine_ids) != expected_ids):
        raise IncompleteBenchmarkError("Ranking requires all 100 synthetic Azure benchmark machines, IDs 1–100")
    inference = get_research_inference()
    ids = sorted(expected_ids)
    features = pd.concat([
        inference.machine_features(*resources.sources, machine_id, prediction_time)
        for machine_id in ids
    ], ignore_index=True)
    # One model call for all machines. The positive class is column 1, exactly
    # as in the existing research predict_from_loaded implementation.
    probabilities = np.asarray(
        resources.saved["model"].predict_proba(features[inference.FEATURE_COLUMNS]), dtype=float,
    )
    if probabilities.shape != (100, 2):
        raise ValueError("Batch model output must contain two probabilities for each of 100 machines")
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Batch model produced an invalid probability")
    return [(machine_id, float(probability)) for machine_id, probability in zip(ids, probabilities[:, 1])]
