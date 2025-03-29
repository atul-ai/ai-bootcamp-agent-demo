// API configuration
const API_BASE_URL = 'http://localhost:8002';

// DOM elements
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
const searchInput = document.getElementById('search-input');
const searchLimit = document.getElementById('search-limit');
const searchModel = document.getElementById('search-model');
const searchBtn = document.getElementById('search-btn');
const searchResults = document.getElementById('search-results');
const queryInput = document.getElementById('query-input');
const queryTask = document.getElementById('query-task');
const queryModel = document.getElementById('query-model');
const queryBtn = document.getElementById('query-btn');
const conversation = document.getElementById('conversation');
const paperIdInput = document.getElementById('paper-id-input');
const summaryModel = document.getElementById('summary-model');
const getPaperBtn = document.getElementById('get-paper-btn');
const getSummaryBtn = document.getElementById('get-summary-btn');
const paperDetails = document.getElementById('paper-details');
const paperSummary = document.getElementById('paper-summary');
const paperTitleInput = document.getElementById('paper-title-input');
const searchTitleBtn = document.getElementById('search-title-btn');
const searchSummarizeBtn = document.getElementById('search-summarize-btn');

// Check API status on page load
document.addEventListener('DOMContentLoaded', checkApiStatus);

// Set up event listeners
tabButtons.forEach(button => {
    button.addEventListener('click', () => switchTab(button.dataset.tab));
});

searchBtn.addEventListener('click', handleSearch);
queryBtn.addEventListener('click', handleQuery);
getPaperBtn.addEventListener('click', handleGetPaper);
getSummaryBtn.addEventListener('click', handleGetSummary);
searchTitleBtn.addEventListener('click', handleSearchByTitle);
searchSummarizeBtn.addEventListener('click', handleSearchAndSummarize);

// Connect paper title clicks to paper details tab
document.addEventListener('click', function(e) {
    if (e.target && e.target.classList.contains('paper-title')) {
        const paperId = e.target.closest('.paper-item').dataset.paperId;
        if (paperId) {
            paperIdInput.value = paperId;
            switchTab('paper');
            handleGetPaper();
        }
    }
});

// Function to check API status
async function checkApiStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/`);
        const data = await response.json();
        
        if (data.message && data.message.includes('running')) {
            statusDot.classList.add('connected');
            statusDot.classList.remove('disconnected');
            statusText.textContent = 'API Connected';
        } else {
            throw new Error('API response unexpected');
        }
    } catch (error) {
        statusDot.classList.add('disconnected');
        statusDot.classList.remove('connected');
        statusText.textContent = 'API Disconnected';
        console.error('API Status Error:', error);
    }
}

// Function to switch tabs
function switchTab(tabName) {
    tabButtons.forEach(btn => {
        if (btn.dataset.tab === tabName) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    tabContents.forEach(content => {
        if (content.id === `${tabName}-tab`) {
            content.classList.add('active');
        } else {
            content.classList.remove('active');
        }
    });
}

// Function to handle paper search
async function handleSearch() {
    const query = searchInput.value.trim();
    if (!query) {
        showError(searchResults, 'Please enter a search query');
        return;
    }
    
    // Clear previous results and show loading
    searchResults.innerHTML = '<div class="loading-container">Searching papers...<div class="loading"></div></div>';
    
    try {
        const response = await fetch(`${API_BASE_URL}/search_papers`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                limit: parseInt(searchLimit.value),
                model: searchModel.value
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success' && data.papers) {
            displaySearchResults(data.papers);
        } else if (data.detail) {
            showError(searchResults, `Error: ${data.detail}`);
        } else {
            showError(searchResults, 'No papers found matching your query');
        }
    } catch (error) {
        showError(searchResults, `Error: ${error.message}`);
        console.error('Search Error:', error);
    }
}

// Function to display search results
function displaySearchResults(papers) {
    if (!papers || papers.length === 0) {
        searchResults.innerHTML = '<p>No papers found matching your query.</p>';
        return;
    }
    
    let resultsHtml = '<h3>Search Results</h3>';
    
    papers.forEach(paper => {
        // Format authors as a comma-separated list
        const authors = Array.isArray(paper.authors) 
            ? paper.authors.join(', ') 
            : paper.authors || 'Unknown';
        
        // Calculate similarity score as percentage if available
        const similarityScore = paper.similarity_score 
            ? `<span class="similarity-score">${Math.round(paper.similarity_score * 100)}%</span>` 
            : '';
        
        resultsHtml += `
            <div class="paper-item" data-paper-id="${paper.id || ''}">
                <h4 class="paper-title">${paper.title || 'Untitled'}</h4>
                <p class="paper-authors">${authors}</p>
                <p class="paper-abstract">${paper.abstract || 'No abstract available'}</p>
                <div class="paper-meta">
                    <span>ID: ${paper.id || 'Unknown'}</span>
                    ${paper.categories ? `<span>Categories: ${paper.categories}</span>` : ''}
                    ${similarityScore}
                </div>
            </div>
        `;
    });
    
    searchResults.innerHTML = resultsHtml;
}

// Function to handle query submission
async function handleQuery() {
    const query = queryInput.value.trim();
    if (!query) {
        showError(conversation, 'Please enter a query');
        return;
    }
    
    // Add user message to conversation
    addMessage(query, 'user');
    
    // Add assistant thinking message
    const thinkingMessage = addMessage('Thinking...', 'assistant');
    
    try {
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                task: queryTask.value,
                model: queryModel.value
            })
        });
        
        const data = await response.json();
        
        // Remove thinking message
        conversation.removeChild(thinkingMessage);
        
        if (data.status === 'success' && data.result) {
            addMessage(data.result, 'assistant');
        } else if (data.detail) {
            showError(conversation, `Error: ${data.detail}`);
        } else {
            addMessage("I'm sorry, I couldn't find an answer to your query.", 'assistant');
        }
    } catch (error) {
        // Remove thinking message
        conversation.removeChild(thinkingMessage);
        showError(conversation, `Error: ${error.message}`);
        console.error('Query Error:', error);
    }
    
    // Clear input
    queryInput.value = '';
}

// Function to add a message to the conversation
function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `${sender}-message`);
    messageDiv.textContent = text;
    conversation.appendChild(messageDiv);
    
    // Scroll to bottom
    conversation.scrollTop = conversation.scrollHeight;
    
    return messageDiv;
}

// Function to display error messages
function showError(container, message) {
    const errorDiv = document.createElement('div');
    errorDiv.classList.add('error-message');
    errorDiv.textContent = message;
    
    // Clear container first if it's the search results
    if (container === searchResults) {
        container.innerHTML = '';
    }
    
    container.appendChild(errorDiv);
}

// Function to handle paper retrieval
async function handleGetPaper() {
    const paperId = paperIdInput.value.trim();
    if (!paperId) {
        showError(paperDetails, 'Please enter a paper ID');
        return;
    }
    
    // Clear previous results and show loading
    paperDetails.innerHTML = '<div class="loading-container">Retrieving paper details...<div class="loading"></div></div>';
    paperSummary.innerHTML = ''; // Clear any previous summary
    
    try {
        const response = await fetch(`${API_BASE_URL}/paper/${paperId}`);
        const data = await response.json();
        
        if (data.status === 'success' && data.paper) {
            displayPaperDetails(data.paper);
        } else if (data.detail) {
            showError(paperDetails, `Error: ${data.detail}`);
        } else {
            showError(paperDetails, 'No paper found with the provided ID');
        }
    } catch (error) {
        showError(paperDetails, `Error: ${error.message}`);
        console.error('Paper retrieval error:', error);
    }
}

// Function to handle paper summarization
async function handleGetSummary() {
    const paperId = paperIdInput.value.trim();
    if (!paperId) {
        showError(paperSummary, 'Please enter a paper ID');
        return;
    }
    
    // Clear previous summary and show loading
    paperSummary.innerHTML = '<div class="loading-container">Generating summary...<div class="loading"></div></div>';
    
    try {
        const response = await fetch(`${API_BASE_URL}/summarize_paper`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                paper_id: paperId,
                model: summaryModel.value
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success' && data.summary) {
            displayPaperSummary(data.summary, summaryModel.value);
            
            // If we don't have paper details yet, display them
            if (paperDetails.innerHTML === '' || paperDetails.querySelector('.error-message')) {
                displayPaperDetails(data.paper);
            }
        } else if (data.detail) {
            showError(paperSummary, `Error: ${data.detail}`);
        } else if (data.error) {
            showError(paperSummary, `Error: ${data.error}`);
        } else {
            showError(paperSummary, 'Failed to generate summary');
        }
    } catch (error) {
        showError(paperSummary, `Error: ${error.message}`);
        console.error('Summary generation error:', error);
    }
}

// Function to display paper details
function displayPaperDetails(paper) {
    let detailsHtml = `
        <div class="paper-metadata">
            <h3>${paper.title || 'Untitled'}</h3>
            <div class="metadata-row">
                <span class="metadata-label">Authors:</span>
                <span>${paper.authors || 'Unknown'}</span>
            </div>
            <div class="metadata-row">
                <span class="metadata-label">ID:</span>
                <span>${paper.id || 'Unknown'}</span>
            </div>
            ${paper.categories ? `
            <div class="metadata-row">
                <span class="metadata-label">Categories:</span>
                <span>${paper.categories}</span>
            </div>` : ''}
            ${paper.journal_ref ? `
            <div class="metadata-row">
                <span class="metadata-label">Journal:</span>
                <span>${paper.journal_ref}</span>
            </div>` : ''}
            ${paper.doi ? `
            <div class="metadata-row">
                <span class="metadata-label">DOI:</span>
                <span>${paper.doi}</span>
            </div>` : ''}
            ${paper.comments ? `
            <div class="metadata-row">
                <span class="metadata-label">Comments:</span>
                <span>${paper.comments}</span>
            </div>` : ''}
        </div>
        <div class="paper-abstract">
            <h4>Abstract</h4>
            <p>${paper.abstract || 'No abstract available'}</p>
        </div>
    `;
    
    paperDetails.innerHTML = detailsHtml;
}

// Function to display paper summary
function displayPaperSummary(summary, model) {
    const modelName = model === 'groq' ? 'Groq (Llama-3)' : 'Sambanova (DeepSeek-R1)';
    
    let summaryHtml = `
        <div class="summary-header">
            <h3>Paper Summary</h3>
            <span class="summary-model-badge">Generated by ${modelName}</span>
        </div>
        <div class="summary-content">
            ${summary}
        </div>
    `;
    
    paperSummary.innerHTML = summaryHtml;
}

// Function to handle paper search by title
async function handleSearchByTitle() {
    const title = paperTitleInput.value.trim();
    if (!title) {
        showError(paperDetails, 'Please enter a paper title');
        return;
    }
    
    // Clear previous results and show loading
    paperDetails.innerHTML = '<div class="loading-container">Searching for paper by title...<div class="loading"></div></div>';
    paperSummary.innerHTML = ''; // Clear any previous summary
    
    try {
        const response = await fetch(`${API_BASE_URL}/search_by_title`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                title: title
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success' && data.paper) {
            // Fill in the paper ID input with the found paper's ID
            if (data.paper.id) {
                paperIdInput.value = data.paper.id;
            }
            
            // Display the paper details
            displayPaperDetails(data.paper);
        } else if (data.detail) {
            showError(paperDetails, `Error: ${data.detail}`);
        } else {
            showError(paperDetails, 'No paper found with the provided title');
        }
    } catch (error) {
        showError(paperDetails, `Error: ${error.message}`);
        console.error('Paper title search error:', error);
    }
}

// Function to handle search and summarize in one go
async function handleSearchAndSummarize() {
    const title = paperTitleInput.value.trim();
    if (!title) {
        showError(paperDetails, 'Please enter a paper title');
        return;
    }
    
    // Clear previous results and show loading
    paperDetails.innerHTML = '<div class="loading-container">Searching and summarizing paper...</div>';
    paperSummary.innerHTML = '<div class="loading-container">This may take a minute...<div class="loading"></div></div>';
    
    try {
        const response = await fetch(`${API_BASE_URL}/search_and_summarize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                title: title,
                model: summaryModel.value
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Fill in the paper ID input with the found paper's ID
            if (data.paper && data.paper.id) {
                paperIdInput.value = data.paper.id;
            }
            
            // Display the paper details and summary
            if (data.paper) {
                displayPaperDetails(data.paper);
            }
            
            if (data.summary) {
                displayPaperSummary(data.summary, summaryModel.value);
            }
        } else if (data.detail) {
            showError(paperSummary, `Error: ${data.detail}`);
            if (!paperDetails.innerHTML || paperDetails.querySelector('.loading-container')) {
                showError(paperDetails, 'Could not find or summarize the paper');
            }
        } else if (data.error) {
            showError(paperSummary, `Error: ${data.error}`);
            if (!paperDetails.innerHTML || paperDetails.querySelector('.loading-container')) {
                showError(paperDetails, 'Could not find or summarize the paper');
            }
        } else {
            showError(paperSummary, 'Failed to search and summarize the paper');
            if (!paperDetails.innerHTML || paperDetails.querySelector('.loading-container')) {
                showError(paperDetails, 'Could not find the paper');
            }
        }
    } catch (error) {
        showError(paperSummary, `Error: ${error.message}`);
        showError(paperDetails, 'Search and summarize request failed');
        console.error('Search and summarize error:', error);
    }
} 