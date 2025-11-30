# Gemini RAG Assistant 🤖📄

A powerful Document Analysis Chatbot powered by **Google Gemini 1.5 Flash**. Upload your documents (PDF, TXT, DOC) and chat with them using an advanced Long-Context RAG (Retrieval-Augmented Generation) approach.

![Gemini RAG Demo](https://via.placeholder.com/800x400?text=Gemini+RAG+Assistant+Screenshot)
*(Add a screenshot of your app here)*

## 🚀 Features

-   **📄 Multi-Format Support**: Upload PDF, TXT, DOC, and DOCX files.
-   **🧠 Long-Context RAG**: Uses Gemini's 1M+ token window to "read" entire documents instead of relying on lossy vector chunks.
-   **💬 Interactive Chat**: Chat with your documents with a history-aware assistant.
-   **📂 File Management**: View, manage, and delete uploaded files.
-   **💾 Persistent History**: Chat history and file lists are saved locally.
-   **🎨 Modern UI**: Clean, responsive interface with Dark/Light themes.

## 🛠️ Tech Stack

-   **Frontend**: HTML5, CSS3, Vanilla JavaScript.
-   **Backend**: Python, FastAPI, Uvicorn.
-   **AI Model**: Google Gemini 1.5 Flash (`google-generativeai`).
-   **Deployment**: Railway (Backend) + Vercel (Frontend).

## 📦 Installation & Setup

### Prerequisites
-   Python 3.9+
-   A Google Cloud API Key for Gemini ([Get it here](https://aistudio.google.com/app/apikey)).

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/gemini-rag-assistant.git
cd gemini-rag-assistant
```

### 2. Backend Setup
Navigate to the backend folder and install dependencies:
```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in the `backend` folder:
```env
GEMINI_API_KEY=your_api_key_here
```

Run the server:
```bash
uvicorn main:app --reload
```
The backend will start at `http://localhost:8000`.

### 3. Frontend Setup
1.  Navigate to the `frontend` folder.
2.  Open `index.html` in your browser.
3.  **Note**: By default, the frontend connects to `http://localhost:8000`.

## 🚀 Deployment

### Backend (Railway)
1.  Push this repo to GitHub.
2.  Create a new project on [Railway](https://railway.app).
3.  Select "Deploy from GitHub".
4.  Set **Root Directory** to `/backend`.
5.  Add `GEMINI_API_KEY` in Variables.
6.  Generate a domain (e.g., `https://your-app.up.railway.app`).

### Frontend (Vercel)
1.  Update `frontend/app.js` and `frontend/files.js` with your Railway URL:
    ```javascript
    const API_URL = "https://your-app.up.railway.app";
    ```
2.  Deploy the `frontend` folder to [Vercel](https://vercel.com).

## 📂 Project Structure

```
├── backend/
│   ├── main.py              # FastAPI application
│   ├── gemini_service.py    # Google Gemini integration logic
│   ├── requirements.txt     # Python dependencies
│   └── uploads/             # Local storage for viewed files
├── frontend/
│   ├── index.html           # Main UI
│   ├── styles.css           # Styling
│   ├── app.js               # Chat & UI logic
│   ├── files.js             # File management logic
│   └── files.html           # File view page
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
