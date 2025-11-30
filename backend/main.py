import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from gemini_service import GeminiService

load_dotenv()

app = FastAPI()

# ==========================
#       FIXED CORS
# ==========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5501",
        "http://localhost:5501",
        "http://127.0.0.1",
        "http://localhost",
        "http://localhost:8000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

gemini = GeminiService(api_key=api_key)

class ChatRequest(BaseModel):
    message: str
    history: list
    file_ids: list | None = None

# ==========================
#       FILE UPLOAD
# ==========================
@app.post("/upload")
async def upload(file: UploadFile):
    resp = await gemini.upload_file(file)
    return resp

# ==========================
#       CHAT ENDPOINT
# ==========================
@app.get("/")
async def root():
    return {"status": "alive", "message": "Backend is running"}

@app.post("/chat")
async def chat(req: ChatRequest):
    response = gemini.ask_question(
        message=req.message,
        history=req.history,
        file_ids=req.file_ids
    )

    return {"response": response}

# ==========================
#      FILES ENDPOINTS
# ==========================
@app.get("/files")
async def list_uploaded_files():
    try:
        files = gemini.list_files()
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files/{file_id:path}")
async def delete_file(file_id: str):
    try:
        gemini.delete_file(file_id)
        return {"message": f"File {file_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/view/{filename}")
async def view_file(filename: str):
    file_path = f"uploads/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"message": "File not found locally. It might have been uploaded in a previous session or deleted."}

# ==========================
#          RUN
# ==========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

