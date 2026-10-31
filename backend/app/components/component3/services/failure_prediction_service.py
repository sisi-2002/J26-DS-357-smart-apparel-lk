"""Call existing inference with cached resources; no training or label creation."""
import logging

from ..ml.failure_inference import get_research_inference
from ..ml.failure_model_loader import InferenceResources
from ..ml.failure_ranking import IncompleteBenchmarkError, score_machines


logger = logging.getLogger(__name__)


class MachineNotFound(Exception):
    pass


class PredictionFailed(Exception):
    pass


class FailurePredictionService:
    def __init__(self, resources: InferenceResources):
        self.resources = resources

    def rank_machines(self, prediction_time: str) -> dict:
        try:
            scored = score_machines(self.resources, prediction_time)
        except IncompleteBenchmarkError:
            raise
        except Exception as exc:
            logger.exception("Component 3 batch inference failed")
            raise PredictionFailed("Component 3 ranking failed") from exc
        # Do not rank the six-decimal serialized values: distinct raw values can
        # round to the same number. Exact raw ties use ascending machine ID.
        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))[:10]
        return {
            "prediction_time": prediction_time,
            "horizon_days": 7,
            "scored_machine_count": 100,
            "returned_machine_count": 10,
            "machines": [
                {"rank": rank, "machine_id": machine_id,
                 "predicted_failure_probability": round(probability, 6)}
                for rank, (machine_id, probability) in enumerate(ranked, start=1)
            ],
            "model_schema_version": self.resources.saved["schema_version"],
            "model_version": self.resources.model_version,
            "research_only": True,
            "data_status": "synthetic_azure_historical_demo",
            "note": "Relative ranking of 100 synthetic Azure benchmark machines at one historical 2015 snapshot. "
                    "Sorted using unrounded 7-day model probabilities; displayed probabilities are rounded to six decimals. "
                    "Not a risk threshold, reliability score, maintenance recommendation or apparel-factory validation.",
        }

    def predict(self, machine_id: int, prediction_time: str) -> dict:
        if machine_id not in self.resources.machine_ids:
            raise MachineNotFound("Machine ID not found in the synthetic Azure benchmark")
        inference = get_research_inference()
        try:
            result = inference.predict_from_loaded(
                self.resources.saved, self.resources.sources, machine_id, prediction_time,
            )
        except Exception as exc:
            logger.exception("Component 3 saved-model inference failed")
            raise PredictionFailed("Component 3 prediction failed") from exc
        return {
            **result,
            "model_schema_version": self.resources.saved["schema_version"],
            "model_version": self.resources.model_version,
            "data_status": "synthetic_azure_historical_demo",
        }
