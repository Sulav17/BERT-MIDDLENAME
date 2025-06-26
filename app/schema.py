from pydantic import BaseModel
from typing import List

class FullNameRequest(BaseModel):
    full_name: str

# Response item schema
class MiddleNameScore(BaseModel):
    name: str
    score: float

# Response schema
class MiddleNameResponse(BaseModel):
    full_name: str
    middle_name: List[MiddleNameScore]
