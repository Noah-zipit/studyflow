// StudyFlow - Frontend JavaScript Application

class StudyFlowApp {
    constructor() {
        this.currentCollectionId = null;
        this.isUploading = false;
        this.isChatting = false;
        this.apiBase = '';

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupDragAndDrop();
        this.checkHealth();
    }

    setupEventListeners() {
        // File input change
        const fileInput = document.getElementById('file-input');
        fileInput?.addEventListener('change', (e) => this.handleFileSelect(e));

        // Chat input
        const chatInput = document.getElementById('chat-input');
        chatInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Upload area click
        const uploadArea = document.getElementById('upload-area');
        uploadArea?.addEventListener('click', () => {
            document.getElementById('file-input')?.click();
        });
    }

    setupDragAndDrop() {
        const uploadArea = document.getElementById('upload-area');
        if (!uploadArea) return;

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');

            const files = Array.from(e.dataTransfer.files).filter(file =>
                file.type === 'application/pdf'
            );

            if (files.length > 0) {
                this.handleFiles(files);
            } else {
                this.showNotification('Please drop only PDF files', 'error');
            }
        });
    }

    handleFileSelect(event) {
        const files = Array.from(event.target.files);
        this.handleFiles(files);
    }

    async handleFiles(files) {
        if (this.isUploading) {
            this.showNotification('Upload already in progress', 'error');
            return;
        }

        if (files.length === 0) {
            this.showNotification('No files selected', 'error');
            return;
        }

        if (files.length > 5) {
            this.showNotification('Maximum 5 files allowed', 'error');
            return;
        }

        // Validate file types and sizes
        const maxSize = 10 * 1024 * 1024; // 10MB
        for (const file of files) {
            if (file.type !== 'application/pdf') {
                this.showNotification(`${file.name} is not a PDF file`, 'error');
                return;
            }
            if (file.size > maxSize) {
                this.showNotification(`${file.name} is too large (max 10MB)`, 'error');
                return;
            }
        }

        await this.uploadFiles(files);
    }

    async uploadFiles(files) {
        this.isUploading = true;
        this.showUploadProgress(true);
        this.updateUploadStatus('Preparing files...');

        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file);
        });

        try {
            this.updateUploadProgress(10);
            this.updateUploadStatus('Uploading files...');

            const response = await fetch(`${this.apiBase}/api/upload`, {
                method: 'POST',
                body: formData
            });

            this.updateUploadProgress(50);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || `Upload failed: ${response.status}`);
            }

            this.updateUploadProgress(80);
            this.updateUploadStatus('Processing documents...');

            const result = await response.json();

            this.updateUploadProgress(100);
            this.updateUploadStatus('Upload complete!');

            setTimeout(() => {
                this.showUploadProgress(false);
                this.displayUploadResults(result);

                if (result.collection_id) {
                    this.currentCollectionId = result.collection_id;
                    this.showChatSection();
                }
            }, 1000);

            this.showNotification('Documents uploaded successfully!', 'success');

        } catch (error) {
            console.error('Upload error:', error);
            this.showUploadProgress(false);
            this.showNotification(`Upload failed: ${error.message}`, 'error');
        } finally {
            this.isUploading = false;
            // Reset file input
            const fileInput = document.getElementById('file-input');
            if (fileInput) fileInput.value = '';
        }
    }

    showUploadProgress(show) {
        const progressDiv = document.getElementById('upload-progress');
        if (progressDiv) {
            progressDiv.style.display = show ? 'block' : 'none';
        }
    }

    updateUploadProgress(percentage) {
        const progressFill = document.getElementById('progress-fill');
        if (progressFill) {
            progressFill.style.width = `${percentage}%`;
        }
    }

    updateUploadStatus(message) {
        const statusElement = document.getElementById('upload-status');
        if (statusElement) {
            statusElement.textContent = message;
        }
    }

    displayUploadResults(result) {
        const resultsDiv = document.getElementById('upload-results');
        if (!resultsDiv) return;

        resultsDiv.style.display = 'block';

        let html = `
            <div class="result-summary">
                <h3>Upload Results</h3>
                <p>Processed ${result.document_count} document${result.document_count !== 1 ? 's' : ''}</p>
                ${result.collection_id ? `<p><strong>Collection ID:</strong> <code>${result.collection_id}</code></p>` : ''}
            </div>
        `;

        if (result.documents && result.documents.length > 0) {
            html += '<div class="documents-list">';
            result.documents.forEach(doc => {
                html += `
                    <div class="result-card result-success">
                        <div class="result-header">
                            <span class="result-filename">${doc.filename}</span>
                            <span class="result-status status-success">✅ Success</span>
                        </div>
                        <div class="result-details">
                            <div><strong>Pages:</strong> ${doc.page_count}</div>
                            <div><strong>Chunks:</strong> ${doc.chunk_count}</div>
                            <div><strong>Size:</strong> ${this.formatFileSize(doc.file_size)}</div>
                            <div><strong>Time:</strong> ${doc.processing_time.toFixed(2)}s</div>
                        </div>
                    </div>
                `;
            });
            html += '</div>';
        }

        if (result.error) {
            html += `
                <div class="result-card result-error">
                    <div class="result-header">
                        <span class="result-filename">Error</span>
                        <span class="result-status status-error">❌ Failed</span>
                    </div>
                    <p style="margin-top: 0.5rem; color: #dc2626;">${result.error}</p>
                </div>
            `;
        }

        resultsDiv.innerHTML = html;
    }

    showChatSection() {
        document.getElementById('chat-section').style.display = 'block';
        document.getElementById('features-section').style.display = 'none';

        // Update collection info
        const collectionElement = document.getElementById('current-collection');
        if (collectionElement && this.currentCollectionId) {
            collectionElement.textContent = this.currentCollectionId;
        }

        // Scroll to chat section
        document.getElementById('chat-section').scrollIntoView({
            behavior: 'smooth'
        });
    }

    async sendMessage() {
        const chatInput = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');

        if (!chatInput || !sendBtn) return;

        const message = chatInput.value.trim();
        if (!message) return;

        if (this.isChatting) {
            this.showNotification('Please wait for the previous response', 'error');
            return;
        }

        this.isChatting = true;
        sendBtn.disabled = true;
        chatInput.disabled = true;

        // Add user message to chat
        this.addMessageToChat(message, 'user');
        chatInput.value = '';

        try {
            // Show typing indicator
            this.addTypingIndicator();

            const includeSourcesInput = document.getElementById('include-sources');
            const includeSources = includeSourcesInput ? includeSourcesInput.checked : true;

            const requestBody = {
                message: message,
                collection_id: this.currentCollectionId,
                include_sources: includeSources,
                max_tokens: 1000,
                temperature: 0.7
            };

            const response = await fetch(`${this.apiBase}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            this.removeTypingIndicator();

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || `Chat failed: ${response.status}`);
            }

            const result = await response.json();

            // Add assistant response
            this.addMessageToChat(result.answer, 'assistant', result.sources);

            this.showNotification(`Response generated (${result.tokens_used} tokens)`, 'success');

        } catch (error) {
            console.error('Chat error:', error);
            this.removeTypingIndicator();
            this.addMessageToChat('Sorry, I encountered an error. Please try again.', 'assistant');
            this.showNotification(`Chat failed: ${error.message}`, 'error');
        } finally {
            this.isChatting = false;
            sendBtn.disabled = false;
            chatInput.disabled = false;
            chatInput.focus();
        }
    }

    addMessageToChat(content, role, sources = null) {
        const chatMessages = document.getElementById('chat-messages');
        if (!chatMessages) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;

        let html = `
            <div class="message-content">
                ${role === 'assistant' ? '<strong>StudyFlow Assistant:</strong> ' : ''}
                ${content}
            </div>
        `;

        if (sources && sources.length > 0) {
            html += `
                <div class="message-sources">
                    <h4>📖 Sources:</h4>
                    <div class="source-list">
                        ${sources.map(source => `
                            <div class="source-item">
                                <div class="source-header">
                                    <span class="source-filename">${source.filename}</span>
                                    ${source.page ? `<span class="source-page">Page ${source.page}</span>` : ''}
                                </div>
                                <div class="source-preview">${source.content_preview}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        messageDiv.innerHTML = html;
        chatMessages.appendChild(messageDiv);

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    addTypingIndicator() {
        const chatMessages = document.getElementById('chat-messages');
        if (!chatMessages) return;

        const typingDiv = document.createElement('div');
        typingDiv.id = 'typing-indicator';
        typingDiv.className = 'message assistant-message';
        typingDiv.innerHTML = `
            <div class="message-content">
                <strong>StudyFlow Assistant:</strong> <em>Thinking...</em>
                <div class="typing-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;

        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Add typing animation CSS if not present
        if (!document.querySelector('#typing-animation-css')) {
            const style = document.createElement('style');
            style.id = 'typing-animation-css';
            style.textContent = `
                .typing-dots {
                    display: inline-block;
                    margin-left: 0.5rem;
                }
                .typing-dots span {
                    display: inline-block;
                    width: 4px;
                    height: 4px;
                    background: #666;
                    border-radius: 50%;
                    margin: 0 1px;
                    animation: typing 1.4s infinite;
                }
                .typing-dots span:nth-child(2) {
                    animation-delay: 0.2s;
                }
                .typing-dots span:nth-child(3) {
                    animation-delay: 0.4s;
                }
                @keyframes typing {
                    0%, 60%, 100% { transform: translateY(0); }
                    30% { transform: translateY(-10px); }
                }
            `;
            document.head.appendChild(style);
        }
    }

    removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    async generateSummary() {
        if (!this.currentCollectionId) {
            this.showNotification('Please upload documents first', 'error');
            return;
        }

        try {
            this.showLoadingOverlay(true, 'Generating summary...');

            const response = await fetch(`${this.apiBase}/api/summarize?collection_id=${this.currentCollectionId}&max_length=500`, {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error('Summary generation failed');
            }

            const result = await response.json();

            this.addMessageToChat(
                `📝 **Document Summary**\n\n${result.summary}`,
                'assistant'
            );

            this.showNotification('Summary generated successfully', 'success');

        } catch (error) {
            console.error('Summary error:', error);
            this.showNotification(`Summary failed: ${error.message}`, 'error');
        } finally {
            this.showLoadingOverlay(false);
        }
    }

    async extractKeyPoints() {
        if (!this.currentCollectionId) {
            this.showNotification('Please upload documents first', 'error');
            return;
        }

        try {
            this.showLoadingOverlay(true, 'Extracting key points...');

            const response = await fetch(`${this.apiBase}/api/key-points?collection_id=${this.currentCollectionId}&num_points=5`, {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error('Key point extraction failed');
            }

            const result = await response.json();

            const keyPointsList = result.key_points.map((point, index) =>
                `${index + 1}. ${point}`
            ).join('\n');

            this.addMessageToChat(
                `🎯 **Key Points**\n\n${keyPointsList}`,
                'assistant'
            );

            this.showNotification('Key points extracted successfully', 'success');

        } catch (error) {
            console.error('Key points error:', error);
            this.showNotification(`Key point extraction failed: ${error.message}`, 'error');
        } finally {
            this.showLoadingOverlay(false);
        }
    }

    askSuggestedQuestion() {
        const suggestedQuestions = [
            "What are the main topics covered in these documents?",
            "Can you explain the key concepts in simple terms?",
            "What are the most important points I should remember?",
            "How do the different sections relate to each other?",
            "What practical applications are discussed?",
            "Are there any examples or case studies mentioned?"
        ];

        const randomQuestion = suggestedQuestions[Math.floor(Math.random() * suggestedQuestions.length)];

        const chatInput = document.getElementById('chat-input');
        if (chatInput) {
            chatInput.value = randomQuestion;
            chatInput.focus();
        }
    }

    showUploadSection() {
        document.getElementById('chat-section').style.display = 'none';
        document.getElementById('features-section').style.display = 'block';
        document.getElementById('upload-section').scrollIntoView({
            behavior: 'smooth'
        });
    }

    showLoadingOverlay(show, message = 'Processing...') {
        const overlay = document.getElementById('loading-overlay');
        if (!overlay) return;

        overlay.style.display = show ? 'flex' : 'none';

        const messageElement = overlay.querySelector('.loading-content p');
        if (messageElement) {
            messageElement.textContent = message;
        }
    }

    showNotification(message, type = 'info') {
        const notification = document.getElementById('notification');
        const messageElement = document.getElementById('notification-message');

        if (!notification || !messageElement) return;

        messageElement.textContent = message;
        notification.className = `notification ${type}`;
        notification.style.display = 'block';

        // Auto hide after 4 seconds
        setTimeout(() => {
            notification.style.display = 'none';
        }, 4000);
    }

    async checkHealth() {
        try {
            const response = await fetch(`${this.apiBase}/api/health`);
            if (response.ok) {
                console.log('✅ StudyFlow API is healthy');
            } else {
                console.warn('⚠️ StudyFlow API health check failed');
            }
        } catch (error) {
            console.error('❌ StudyFlow API connection failed:', error);
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Global functions for onclick handlers
function sendMessage() {
    window.studyFlowApp?.sendMessage();
}

function generateSummary() {
    window.studyFlowApp?.generateSummary();
}

function extractKeyPoints() {
    window.studyFlowApp?.extractKeyPoints();
}

function askSuggestedQuestion() {
    window.studyFlowApp?.askSuggestedQuestion();
}

function showUploadSection() {
    window.studyFlowApp?.showUploadSection();
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.studyFlowApp = new StudyFlowApp();
});