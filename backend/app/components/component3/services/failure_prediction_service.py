"""Call existing inference with cached resources; no training or label creation."""
import logging

from ..ml.failure_inference import get_research_inference
from ..ml.failure_model_loader import InferenceResources


logger = logging.getLogger(__name__)


class MachineNotFound(Exception):
    pass


class PredictionFailed(Exception):
    pass


class FailurePredictionService:
    def __init__(self, resources: InferenceResources):
        self.resources = resources

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
