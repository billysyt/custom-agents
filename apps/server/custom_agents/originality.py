"""Strict model verdicts: similarity retrieves evidence, an editor decides novelty."""
from pydantic import BaseModel, Field

class MatchingFact(BaseModel):
    proposed_quote: str = Field(min_length=12, max_length=300)
    previous_quote: str = Field(min_length=12, max_length=300)

class Novelty(BaseModel):
    duplicate: bool
    reason: str = Field(min_length=10, max_length=1500)
    matched_job_id: str | None
    new_facts: list[str]
    matching_facts: list[MatchingFact] = Field(default_factory=list)

class Originality(BaseModel):
    original: bool
    reason: str = Field(min_length=10,max_length=1500)
