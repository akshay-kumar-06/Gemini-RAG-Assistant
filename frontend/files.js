const API_URL = "https://gemini-rag-assistant-production.up.railway.app";

// DOM Elements
const filesGrid = document.getElementById('files-grid');
const searchInput = document.getElementById('file-search');
const uploadBtn = document.getElementById('upload-btn');
const fileInput = document.getElementById('library-file-input');

// State
let allFiles = [];

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    loadSavedTheme();
    loadFiles();

    // Event Listeners
    searchInput.addEventListener('input', (e) => filterFiles(e.target.value));

    uploadBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleUpload(e.target.files);
        }
    });
});

// Theme Handling
function loadSavedTheme() {
    const savedTheme = localStorage.getItem('gemini_rag_theme');
    if (savedTheme && savedTheme !== 'default') {
        document.documentElement.setAttribute('data-theme', savedTheme);
    }
}

// File Management
async function loadFiles() {
    try {
        const response = await fetch(`${API_URL}/files`);
        if (response.ok) {
            allFiles = await response.json();
            renderFiles(allFiles);
        } else {
            throw new Error('Failed to fetch files');
        }
    } catch (error) {
        console.error("Error loading files:", error);
        filesGrid.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-triangle-exclamation" style="color: var(--danger-color)"></i>
                <p>Failed to load files. Please check if the backend is running.</p>
            </div>
        `;
    }
}

function renderFiles(files) {
    if (files.length === 0) {
        filesGrid.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-folder-open"></i>
                <p>No files uploaded yet</p>
            </div>
        `;
        return;
    }

    filesGrid.innerHTML = files.map(file => {
        const name = file.display_name || file.name;
        const safeName = name.replace(/'/g, "\\'");
        // Mock date since API might not return it yet, or use current date if new
        const date = new Date().toLocaleDateString();
        const type = name.split('.').pop().toUpperCase();

        return `
            <div class="file-card">
                <div class="file-icon">
                    <i class="fa-solid fa-file-pdf"></i>
                </div>
                <div class="file-info">
                    <div class="file-name" title="${name}">${name}</div>
                    <div class="file-meta">
                        <span>${type}</span>
                        <span>•</span>
                        <span>Uploaded</span>
                    </div>
                </div>
                <div class="file-actions">
                    <button class="file-action-btn" onclick="viewFile('${safeName}')" title="View File">
                        <i class="fa-solid fa-eye"></i> View
                    </button>
                    <button class="file-action-btn delete" onclick="deleteFile('${file.name}')" title="Delete File">
                        <i class="fa-solid fa-trash"></i> Delete
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

function filterFiles(searchTerm) {
    const term = searchTerm.toLowerCase();
    const filtered = allFiles.filter(file =>
        (file.display_name || file.name).toLowerCase().includes(term)
    );
    renderFiles(filtered);
}

// Upload Handling
async function handleUpload(files) {
    // Show loading state
    const originalBtnContent = uploadBtn.innerHTML;
    uploadBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Uploading...`;
    uploadBtn.disabled = true;

    try {
        for (const file of files) {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${API_URL}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error(`Failed to upload ${file.name}`);
        }

        // Reload files after successful upload
        await loadFiles();

    } catch (error) {
        console.error("Upload error:", error);
        alert("Failed to upload one or more files.");
    } finally {
        // Reset button
        uploadBtn.innerHTML = originalBtnContent;
        uploadBtn.disabled = false;
        fileInput.value = ''; // Reset input
    }
}

// Actions
function viewFile(fileName) {
    if (fileName) {
        // Open the local file view endpoint
        window.open(`${API_URL}/files/view/${encodeURIComponent(fileName)}`, '_blank');
    } else {
        alert("File name not available");
    }
}

async function deleteFile(fileId) {
    if (confirm(`Are you sure you want to delete this file?`)) {
        try {
            const response = await fetch(`${API_URL}/files/${fileId}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                // Refresh list
                await loadFiles();
            } else {
                const error = await response.json();
                alert(`Failed to delete file: ${error.detail || 'Unknown error'}`);
            }
        } catch (error) {
            console.error("Delete error:", error);
            alert("Failed to delete file. Check console for details.");
        }
    }
}

