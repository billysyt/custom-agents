"""Strict model verdicts: similarity retrieves evidence, an editor decides novelty."""
from pydantic import BaseModel, Field

class Novelty(BaseModel):
    duplicate: bool
    reason: str = Field(min_length=10, max_length=1500)
    matched_job_id: str | None
    new_facts: list[str]

class Originality(BaseModel):
    original: bool
    reason: str = Field(min_length=10,max_length=1500)
