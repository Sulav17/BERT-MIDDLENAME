from pydantic import BaseModel

class FullNameRequest(BaseModel):
    full_name: str

class MiddleNameResponse(BaseModel):
    middle_name: str
