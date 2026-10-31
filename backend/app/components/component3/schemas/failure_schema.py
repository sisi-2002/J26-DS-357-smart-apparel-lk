"""Historical inputs and risk outputs; evidence is not a reliability score."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, NaiveDatetime, field_validator

from ..ml.failure_inference import get_research_inference


class HistoricalPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_time: NaiveDatetime = Field(
        description="Timezone-naive synthetic benchmark time: 13 Wednesdays, "
        "2015-09-02 through 2015-11-25 at 06:00:00. No current-time predictions.",
        examples=["2015-09-02T06:00:00"],
    )

    @field_validator("prediction_time", mode="before")
    @classmethod
    def require_timestamp_string(cls, value):
        if not isinstance(value, str):
            raise ValueError("prediction_time must be a historical timestamp string")
        return value

    @field_validator("prediction_time")
    @classmethod
    def require_demo_timestamp(cls, value: datetime) -> datetime:
        if value not in get_research_inference().valid_demo_timestamps():
            raise ValueError("Use one of the 13 weekly benchmark timestamps: "
                             "2015-09-02 through 2015-11-25 at 06:00:00")
        return value


class FailureRiskRequest(HistoricalPredictionRequest):
    machine_id: int = Field(strict=True, gt=0, examples=[1])


class EvidenceProfile(BaseModel):
    H_observed_maintenance_span_days: float | None = Field(ge=0)
    M_prior_maintenance_events: int = Field(ge=0)
    F_prior_failure_events: int = Field(ge=0)
    R_days_since_latest_maintenance: float | None = Field(ge=0)
    C_published_field_completeness_pct: Literal[100.0]


class FailureRiskResponse(BaseModel):
    machine_id: int
    prediction_time: NaiveDatetime
    horizon_days: Literal[7]
    predicted_failure_probability: float = Field(ge=0, le=1, allow_inf_nan=False)
    calibration: str
    evidence_profile: EvidenceProfile
    machine_type: str
    research_only: Literal[True]
    note: str
    request_id: str
    model_schema_version: str
    model_version: str = Field(description="SHA-256 identity of the loaded artifact")
    data_status: Literal["synthetic_azure_historical_demo"]


class RankedMachine(BaseModel):
    rank: int = Field(ge=1, le=10)
    machine_id: int = Field(ge=1, le=100)
    predicted_failure_probability: float = Field(ge=0, le=1, allow_inf_nan=False)


class TopRiskMachinesResponse(BaseModel):
    prediction_time: NaiveDatetime
    horizon_days: Literal[7]
    scored_machine_count: Literal[100]
    returned_machine_count: Literal[10]
    machines: list[RankedMachine] = Field(min_length=10, max_length=10)
    request_id: str
    model_schema_version: str
    model_version: str
    research_only: Literal[True]
    data_status: Literal["synthetic_azure_historical_demo"]
    note: str
