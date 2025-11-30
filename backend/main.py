import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from gemini_service import GeminiService

# Load environment variables
load_dotenv()

app = FastAPI()

# ==========================
#        FIXED CORS
# ==========================
# We are allowing ALL origins ("*") to rule out any connection issues.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Validation
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # Print to logs so we can see it in Railway if it fails
    print("CRITICAL ERROR: GEMINI_API_KEY is missing!") 
    raise ValueError("GEMINI_API_KEY not found in environment variables")

gemini = GeminiService(api_key=api_key)

class ChatRequest(BaseModel):
    message: str
    history: list
    file_ids: list | None = None

# ==========================
#   ✅ NEW: HEALTH CHECK
# ==========================
# Railway pings this to ensure the app is not dead.
@app.get("/")
async def health_check():
    return {"status": "alive", "message": "Backend is running smoothly"}

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
    return {"message": "File not found locally."}

# ==========================
#   ✅ NEW: ROBUST STARTUP
# ==========================
if __name__ == "__main__":
    # Get the PORT from Railway (os.environ), or use 8000 if running locally
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
