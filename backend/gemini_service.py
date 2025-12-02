import google.generativeai as genai
from pydantic import BaseModel
from fastapi import UploadFile
import io
import os  # <--- Added this to handle local file saving

class FileUploadResponse(BaseModel):
    filename: str
    file_id: str
    mime_type: str
    uri: str

class GeminiService:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.chat_sessions = {}

    async def upload_file(self, file: UploadFile) -> FileUploadResponse:
        """
        Uploads file to Google Gemini AND saves a local copy for viewing.
        """
        # 1. Read the file content into memory
        content = await file.read()

        # 2. SAVE LOCALLY (The Fix for the "View" button)
        # Check if uploads folder exists, if not create it
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        
        # Write the file to the local disk
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(content)
        print(f"DEBUG: Saved file locally to {file_path}")

        # 3. Prepare for Gemini (using the same content)
        file_bytes = io.BytesIO(content)
        mime_type = file.content_type or "application/octet-stream"

        # 4. Upload to Google
        gemini_file = genai.upload_file(
            file_bytes,
            display_name=file.filename,
            mime_type=mime_type
        )

        # 5. Return the response
        return FileUploadResponse(
            filename=file.filename,
            file_id=gemini_file.uri, 
            mime_type=mime_type,
            uri=gemini_file.uri
        )

    def ask_question(self, message: str, history: list, file_ids=None):
        try:
            print(f"DEBUG: ask_question called with message='{message}', file_ids={file_ids}")
            
            # 1. Initialize model
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash", # Updated to flash for speed/cost
                system_instruction="You are a PDF-based RAG assistant. Answer only from uploaded documents. You have access to metadata. [page x]"
            )

            # 2. Prepare history for Gemini
            gemini_history = []
            for msg in history:
                role = "user" if msg['role'] == "user" else "model"
                gemini_history.append({
                    "role": role,
                    "parts": [msg['content']]
                })

            # 3. Start chat with history
            chat = model.start_chat(history=gemini_history)

            # 4. Prepare current message parts
            message_parts = [message]
            if file_ids:
                print("DEBUG: Processing file_ids...")
                clean_file_ids = []
                for fid in file_ids:
                    # If fid is a URI, extract the 'files/...' part
                    if isinstance(fid, str) and fid.startswith("http") and "/files/" in fid:
                        clean_id = "files/" + fid.split("/files/")[-1]
                        clean_file_ids.append(clean_id)
                        print(f"DEBUG: Extracted ID {clean_id} from URI {fid}")
                    else:
                        clean_file_ids.append(fid)
                        print(f"DEBUG: Using raw ID {fid}")

                valid_files = []
                for fid in clean_file_ids:
                    try:
                        f = genai.get_file(fid)
                        print(f"DEBUG: Retrieved file {f.name}, state={f.state.name}")
                        if f.state.name == "ACTIVE":
                            valid_files.append(f)
                        else:
                            print(f"WARNING: File {f.name} is not ACTIVE (State: {f.state.name}). Skipping.")
                    except Exception as e:
                        print(f"ERROR: Could not get file {fid}: {e}")

                if valid_files:
                    message_parts.extend(valid_files)
                    print(f"DEBUG: Attached {len(valid_files)} files to message.")
                else:
                    print("DEBUG: No valid files to attach.")

            # 5. Send message
            print("DEBUG: Sending message to Gemini...")
            response = chat.send_message(message_parts)
            print("DEBUG: Response received.")
            return response.text
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"CRITICAL ERROR in ask_question: {e}")
            raise e

    def list_files(self):
        """
        List all files uploaded to Gemini.
        """
        try:
            files = []
            # genai.list_files() returns an iterable of File objects
            for f in genai.list_files():
                files.append({
                    "name": f.name,
                    "display_name": f.display_name,
                    "uri": f.uri,
                    "mime_type": f.mime_type,
                    "size_bytes": f.size_bytes,
                    "create_time": f.create_time.isoformat() if f.create_time else None,
                    "state": f.state.name
                })
            return files
        except Exception as e:
            print(f"Error listing files: {e}")
            raise e

    def delete_file(self, file_id: str):
        """
        Delete a file from Gemini.
        """
        try:
            print(f"DEBUG: Deleting file {file_id}")
            genai.delete_file(file_id)
        except Exception as e:
            print(f"Error deleting file {file_id}: {e}")
            raise e
            
