from pydantic import BaseModel

# Define request model
class ChatRequest(BaseModel):
    message: str