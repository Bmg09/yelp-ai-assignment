from typing import Literal
from pydantic import BaseModel, Field


class StarsOnly(BaseModel):
    stars: int = Field(ge=1, le=5)
    explanation: str = Field(max_length=600)


class StarsDirect(BaseModel):
    stars: int = Field(ge=1, le=5)


class StarsCoT(BaseModel):
    reasoning: str = Field(max_length=1000)
    stars: int = Field(ge=1, le=5)


class KeySignal(BaseModel):
    type: Literal["complaint", "compliment", "mixed"]
    text: str = Field(max_length=400)


class MultiObjective(BaseModel):
    stars: int = Field(ge=1, le=5)
    signal: KeySignal
    business_response: str = Field(max_length=800)


class JudgeScore(BaseModel):
    faithfulness: int = Field(ge=1, le=5)
    politeness: int = Field(ge=1, le=5)
    actionability: int = Field(ge=1, le=5)
    rationale: str = Field(max_length=500)
