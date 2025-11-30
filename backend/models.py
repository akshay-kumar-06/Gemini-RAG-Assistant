from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    file_ids: List[str] = []

class ChatResponse(BaseModel):
    response: str
    
class FileUploadResponse(BaseModel):
    filename: str
    file_id: str
    mime_type: str
    uri: str
