"""Shared schemas used across components."""
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    detail: str
