const API_URL = "https://gemini-rag-assistant-production.up.railway.app";

// State
let chatHistory = [];
let uploadedFiles = []; // { id, name, display_name, uri, mime_type }
let currentChatId = null;
let savedChats = JSON.parse(localStorage.getItem('gemini_rag_chats')) || {};
let currentSidebarPanel = 'menu';

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('chat-file-input');
const fileList = document.getElementById('file-list');
const messagesContainer = document.getElementById('messages-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const thinkingIndicator = document.getElementById('thinking-indicator');
const chatHistoryContainer = document.getElementById('chat-history');
const newChatBtn = document.getElementById('new-chat-btn');

// Sidebar
const sidebarWrapper = document.getElementById('sidebar-wrapper');
const menuPanel = document.getElementById('menu-panel');
const historyPanel = document.getElementById('history-panel');
const navBtnMenu = document.getElementById('nav-btn-menu');
const navBtnHistory = document.getElementById('nav-btn-history');

// =============== INITIALIZATION ===============
document.addEventListener('DOMContentLoaded', () => {
    console.log("🚀 App initialized");

    // Initialize UI
    renderChatHistoryList();
    startNewChat();
    initSidebarNavigation();
    loadSavedTheme();

    // Theme toggle
    document.getElementById('theme-toggle')?.addEventListener('click', () => {
        const themes = ['soft-pink', 'cool-gray'];
        const current = localStorage.getItem('gemini_rag_theme') || 'default';
        const next = themes.find(t => t !== current) || themes[0];
        setTheme(next);
    });

    // =============== FILE UPLOAD SETUP ===============
    if (dropZone && fileInput) {
        // Click handler
        dropZone.addEventListener('click', (e) => {
            if (e.target !== fileInput) {
                fileInput.click();
            }
        });

        // Drag handlers
        ['dragover', 'dragenter'].forEach(e =>
            dropZone.addEventListener(e, ev => {
                ev.preventDefault();
                dropZone.classList.add('dragover');
            })
        );

        ['dragleave', 'drop'].forEach(e =>
            dropZone.addEventListener(e, ev => {
                ev.preventDefault();
                dropZone.classList.remove('dragover');
            })
        );

        // Drop handler
        dropZone.addEventListener('drop', e => {
            console.log("Files dropped");
            handleFiles(e.dataTransfer.files);
        });

        // Input change handler
        fileInput.addEventListener('change', e => {
            console.log("File input changed!", e.target.files);
            // alert("File input changed! Starting upload..."); // Debug alert
            if (e.target.files?.length) handleFiles(e.target.files);
        });
    } else {
        console.error("CRITICAL: Drop zone or file input not found!");
    }
});

// =============== SIDEBAR NAVIGATION ===============
function initSidebarNavigation() {
    navBtnMenu.addEventListener('click', () => switchSidebarPanel('menu'));
    navBtnHistory.addEventListener('click', () => switchSidebarPanel('history'));
    switchSidebarPanel('menu');
}

function switchSidebarPanel(panel) {
    currentSidebarPanel = panel;
    navBtnMenu.classList.toggle('active', panel === 'menu');
    navBtnHistory.classList.toggle('active', panel === 'history');
    sidebarWrapper.style.transform = panel === 'menu' ? 'translateX(0)' : 'translateX(-50%)';
}

// =============== THEME HANDLING ===============
function setTheme(themeName) {
    if (themeName === 'default' || !themeName) {
        document.documentElement.removeAttribute('data-theme');
        localStorage.removeItem('gemini_rag_theme');
    } else {
        document.documentElement.setAttribute('data-theme', themeName);
        localStorage.setItem('gemini_rag_theme', themeName);
    }
}

function loadSavedTheme() {
    const saved = localStorage.getItem('gemini_rag_theme');
    if (saved) setTheme(saved);
}

// =============== FILE MANAGEMENT ===============
function updateFileCount(count) {
    const el = document.getElementById('file-count');
    if (el) el.textContent = count;
}

function renderActiveFilesList() {
    console.log("Rendering active files list. Count:", uploadedFiles.length);
    fileList.innerHTML = '';

    if (uploadedFiles.length === 0) {
        fileList.innerHTML = `
            <div class="empty-files-state">
                <i class="fa-solid fa-folder-open"></i>
                <p>No files attached</p>
                <span>Drop PDFs here to analyze</span>
            </div>
        `;
        updateFileCount(0);
        return;
    }

    uploadedFiles.forEach((file, index) => {
        const item = document.createElement('div');
        item.className = 'file-stack-item';

        // Create inner HTML structure
        item.innerHTML = `
            <div class="file-stack" style="cursor: pointer;">
                <div class="layer"></div>
                <div class="layer"></div>
                <div class="layer"></div>
                <div class="content">
                    <i class="fa-solid fa-file-pdf"></i>
                    <span class="file-name"></span>
                </div>
            </div>
            <button class="remove-file-btn" title="Remove from this chat">
                <i class="fa-solid fa-xmark"></i>
            </button>
        `;

        // Set text content safely
        const nameSpan = item.querySelector('.file-name');
        nameSpan.textContent = file.display_name || file.name;
        nameSpan.title = file.display_name || file.name;

        // Add click handler for viewing
        const fileStack = item.querySelector('.file-stack');
        fileStack.addEventListener('click', () => viewFile(file.name));

        // Add click handler for removal
        const removeBtn = item.querySelector('.remove-file-btn');
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            uploadedFiles.splice(index, 1);
            saveChat();
            renderActiveFilesList();
            showToast('File removed from chat');
        });

        fileList.appendChild(item);
    });

    updateFileCount(uploadedFiles.length);
}

// Toast
function showToast(msg) {
    const toast = document.createElement('div');
    toast.className = 'toast-notification';
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 2000);
}

async function handleFiles(files) {
    console.log("Handling files:", files);
    for (const file of files) await uploadFile(file);
}

async function uploadFile(file) {
    const tempItem = document.createElement('div');
    tempItem.className = 'file-item uploading';
    tempItem.innerHTML = `
        <div class="file-item-content">
            <i class="fa-solid fa-spinner fa-spin"></i>
            <span>${file.name}</span>
        </div>
    `;
    fileList.appendChild(tempItem);

    const formData = new FormData();
    formData.append('file', file);

    try {
        console.warn("Starting upload for:", file.name);
        const res = await fetch(`${API_URL}/upload`, { method: 'POST', body: formData });
        if (!res.ok) throw new Error(`Upload failed: ${res.status} ${res.statusText}`);

        const data = await res.json();
        console.warn("SERVER RESPONSE:", data);

        if (!data || !data.file_id) {
            console.error("INVALID DATA FROM SERVER:", data);
            throw new Error("Invalid response from server: missing file_id");
        }

        const newFile = {
            id: data.file_id,
            name: data.filename || file.name,
            display_name: data.filename || file.name,
            mime_type: data.mime_type || file.type,
            uri: data.uri || ""
        };

        uploadedFiles.push(newFile);
        console.warn("UPLOADED FILES ARRAY NOW:", uploadedFiles);

        saveChat();

        tempItem.classList.replace('uploading', 'success');
        tempItem.innerHTML = `
            <div class="file-item-content">
                <i class="fa-solid fa-check"></i>
                <span>${file.name}</span>
            </div>
        `;

        setTimeout(() => {
            tempItem.remove();
            renderActiveFilesList();
        }, 1200);

    } catch (err) {
        console.error("Upload error details:", err);
        tempItem.classList.replace('uploading', 'error');
        tempItem.innerHTML = `
            <div class="file-item-content">
                <i class="fa-solid fa-triangle-exclamation"></i>
                <span>${file.name}</span>
            </div>
        `;
        // Force remove after error so user can try again
        setTimeout(() => tempItem.remove(), 3000);
    }
}

// =============== CHAT MANAGEMENT ===============
newChatBtn.addEventListener('click', startNewChat);

function startNewChat() {
    console.log("Starting new chat... Called from:", new Error().stack);
    currentChatId = Date.now().toString();
    chatHistory = [];
    uploadedFiles = [];

    messagesContainer.innerHTML = `
        <div class="message system welcome-message">
            <div class="avatar"><i class="fa-solid fa-robot"></i></div>
            <div class="content">
                <h2>How can I help you today?</h2>
                <p>I've read your documents. Ask me anything about them!</p>
            </div>
        </div>
    `;

    renderActiveFilesList();
    updateActiveHistoryItem();
    userInput.focus();
}

function saveChat() {
    if (!chatHistory.length && !uploadedFiles.length) return;

    let title = "New Chat";
    const firstMsg = chatHistory.find(m => m.role === 'user');
    if (firstMsg) {
        title = firstMsg.content.slice(0, 35) + (firstMsg.content.length > 35 ? '...' : '');
    } else if (uploadedFiles.length) {
        title = `Chat with ${uploadedFiles.length} file${uploadedFiles.length > 1 ? 's' : ''}`;
    }

    savedChats[currentChatId] = {
        id: currentChatId,
        title,
        timestamp: Date.now(),
        messages: chatHistory,
        files: uploadedFiles
    };

    localStorage.setItem('gemini_rag_chats', JSON.stringify(savedChats));
    renderChatHistoryList();
}

function loadChat(id) {
    const chat = savedChats[id];
    if (!chat) return;

    currentChatId = id;
    chatHistory = chat.messages || [];
    uploadedFiles = chat.files || [];

    messagesContainer.innerHTML = chatHistory.length === 0 ? `
        <div class="message system welcome-message">
            <div class="avatar"><i class="fa-solid fa-robot"></i></div>
            <div class="content">
                <h2>How can I help you today?</h2>
                <p>I've read your documents. Ask me anything about them!</p>
            </div>
        </div>
    ` : '';

    chatHistory.forEach(m => addMessageToUI(m.content, m.role));
    renderActiveFilesList();
    updateActiveHistoryItem();
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function deleteChat(id, e) {
    e.stopPropagation();
    delete savedChats[id];
    localStorage.setItem('gemini_rag_chats', JSON.stringify(savedChats));
    if (currentChatId === id) startNewChat();
    renderChatHistoryList();
}

function renderChatHistoryList() {
    chatHistoryContainer.innerHTML = '';
    const sorted = Object.values(savedChats).sort((a, b) => b.timestamp - a.timestamp);

    if (!sorted.length) {
        chatHistoryContainer.innerHTML = `
            <div class="empty-history">
                <i class="fa-solid fa-comments"></i>
                <p>No chat history yet</p>
                <span>Start a new conversation!</span>
            </div>`;
        return;
    }

    sorted.forEach(chat => {
        const item = document.createElement('div');
        item.className = `history-item ${chat.id === currentChatId ? 'active' : ''}`;
        item.innerHTML = `
            <div class="history-item-content">
                <i class="fa-solid fa-message"></i>
                <div class="history-item-text">
                    <span class="history-title">${chat.title}</span>
                    <span class="history-time">${formatTimeAgo(new Date(chat.timestamp))}</span>
                </div>
            </div>
            <button class="delete-chat-btn" title="Delete">
                <i class="fa-solid fa-trash"></i>
            </button>
        `;
        item.querySelector('.history-item-content').onclick = () => loadChat(chat.id);
        item.querySelector('.delete-chat-btn').onclick = (e) => deleteChat(chat.id, e);
        chatHistoryContainer.appendChild(item);
    });
}

function formatTimeAgo(date) {
    const sec = Math.floor((Date.now() - date) / 1000);
    if (sec < 60) return 'Just now';
    const min = Math.floor(sec / 60);
    if (min < 60) return `${min}m ago`;
    const hr = Math.floor(min / 60);
    if (hr < 24) return `${hr}h ago`;
    const day = Math.floor(hr / 24);
    if (day < 7) return `${day}d ago`;
    return date.toLocaleDateString();
}

function updateActiveHistoryItem() {
    renderChatHistoryList();
}

// =============== CHAT INPUT & SEND ===============
userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = userInput.scrollHeight + 'px';
    sendBtn.disabled = !userInput.value.trim();
});

userInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!sendBtn.disabled) sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    addMessageToUI(text, 'user');
    chatHistory.push({ role: 'user', content: text });
    saveChat();

    userInput.value = '';
    userInput.style.height = 'auto';
    sendBtn.disabled = true;
    thinkingIndicator.classList.add('active');

    try {
        const fileIds = uploadedFiles.map(f => f.id);

        const res = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                history: chatHistory.slice(0, -1),
                file_ids: fileIds
            })
        });

        if (!res.ok) throw new Error('Request failed');

        const data = await res.json();
        addMessageToUI(data.response, 'model');
        chatHistory.push({ role: 'model', content: data.response });
        saveChat();

    } catch (err) {
        console.error(err);
        addMessageToUI("Sorry, something went wrong. Try again.", 'system');
    } finally {
        thinkingIndicator.classList.remove('active');
        sendBtn.disabled = false;
    }
}

function addMessageToUI(text, role) {
    const div = document.createElement('div');
    div.className = `message ${role}`;

    const icon = role === 'user' ? 'fa-user' : 'fa-robot';

    const formatted = text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\[Page (\d+)\]/g, '<span class="citation">[Page $1]</span>')
        .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
        .replace(/\n/g, '<br>');

    div.innerHTML = `
        <div class="avatar"><i class="fa-solid ${icon}"></i></div>
        <div class="content">${formatted}</div>
    `;

    messagesContainer.appendChild(div);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// =============== FILE PAGE LINK ===============
document.getElementById('open-files-page')?.addEventListener('click', () => {
    window.location.href = "files.html";
});

function viewFile(fileName) {
    if (fileName) {
        window.open(`${API_URL}/files/view/${encodeURIComponent(fileName)}`, '_blank');
    }
}

