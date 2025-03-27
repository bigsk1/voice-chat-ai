/**
 * Debug Panel for Voice Chat AI
 * Provides a simple interface for debugging connections and data flow
 */

document.addEventListener('DOMContentLoaded', function() {
    // Create the debug panel container
    const debugPanel = document.createElement('div');
    debugPanel.id = 'debug-panel';
    debugPanel.style.cssText = `
        position: fixed;
        bottom: 10px;
        right: 10px;
        width: 400px;
        max-height: 300px;
        background-color: rgba(0, 0, 0, 0.9);
        color: #00ff00;
        font-family: monospace;
        font-size: 12px;
        padding: 10px;
        border-radius: 5px;
        z-index: 9999;
        overflow-y: auto;
        display: none;
        border: 1px solid #00ff00;
        box-shadow: 0 0 10px rgba(0, 255, 0, 0.3);
    `;
    
    // Create the header
    const header = document.createElement('div');
    header.style.cssText = `
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
        padding-bottom: 5px;
        border-bottom: 1px solid #00ff00;
    `;
    
    const title = document.createElement('h3');
    title.textContent = 'Debug Panel';
    title.style.margin = '0';
    
    const closeBtn = document.createElement('button');
    closeBtn.textContent = 'X';
    closeBtn.style.cssText = `
        background: none;
        border: none;
        color: #ff5555;
        cursor: pointer;
        font-weight: bold;
    `;
    closeBtn.onclick = function() {
        debugPanel.style.display = 'none';
        toggleButton.style.display = 'block';
    };
    
    header.appendChild(title);
    header.appendChild(closeBtn);
    debugPanel.appendChild(header);
    
    // Add intro message
    const introMsg = document.createElement('div');
    introMsg.style.cssText = `
        color: #aaffaa;
        margin-bottom: 10px;
        font-style: italic;
    `;
    introMsg.textContent = 'Debug information will appear here. Click "Send Text" or speak to see data flow.';
    debugPanel.appendChild(introMsg);
    
    // Create the log area
    const logArea = document.createElement('div');
    logArea.id = 'debug-log';
    logArea.style.cssText = `
        max-height: 220px;
        overflow-y: auto;
        margin-bottom: 10px;
    `;
    debugPanel.appendChild(logArea);
    
    // Create footer controls
    const footerControls = document.createElement('div');
    footerControls.style.cssText = `
        display: flex;
        justify-content: space-between;
    `;
    
    const clearBtn = document.createElement('button');
    clearBtn.textContent = 'Clear Log';
    clearBtn.style.cssText = `
        background-color: #333;
        color: #fff;
        border: 1px solid #00aa00;
        padding: 5px 10px;
        border-radius: 3px;
        cursor: pointer;
    `;
    clearBtn.onclick = function() {
        document.getElementById('debug-log').innerHTML = '';
    };
    
    footerControls.appendChild(clearBtn);
    debugPanel.appendChild(footerControls);
    
    // Add to the document
    document.body.appendChild(debugPanel);
    
    // Create toggle button
    const toggleButton = document.createElement('button');
    toggleButton.textContent = 'Debug Panel';
    toggleButton.style.cssText = `
        position: fixed;
        bottom: 10px;
        right: 10px;
        background-color: rgba(0, 0, 0, 0.7);
        color: #00ff00;
        border: 1px solid #00ff00;
        border-radius: 3px;
        padding: 5px 10px;
        font-family: monospace;
        font-size: 12px;
        cursor: pointer;
        z-index: 9998;
    `;
    
    toggleButton.onclick = function() {
        const panel = document.getElementById('debug-panel');
        if (panel.style.display === 'none') {
            panel.style.display = 'block';
            toggleButton.style.display = 'none';
        }
    };
    
    document.body.appendChild(toggleButton);
    
    // Flag to prevent recursion
    let isLogging = false;
    
    // Internal log function to prevent recursion
    function internalLog(message, type = 'info') {
        if (isLogging) return; // Prevent recursion
        
        const log = document.getElementById('debug-log');
        if (!log) return;
        
        const entry = document.createElement('div');
        
        const timestamp = new Date().toLocaleTimeString();
        
        let color = '#aaffaa'; // Default color for info
        switch(type.toLowerCase()) {
            case 'error':
                color = '#ff5555';
                break;
            case 'warning':
                color = '#ffaa55';
                break;
            case 'success':
                color = '#55ff55';
                break;
            case 'event':
                color = '#55aaff';
                break;
            case 'data':
                color = '#aa55ff';
                break;
        }
        
        entry.style.color = color;
        entry.style.marginBottom = '3px';
        
        // Handle objects and arrays
        let formattedMessage;
        if (typeof message === 'object' && message !== null) {
            try {
                formattedMessage = JSON.stringify(message, null, 2);
                // Add pre formatting for objects
                entry.innerHTML = `<span style="color:#777">[${timestamp}]</span> <pre style="margin: 0; white-space: pre-wrap;">${formattedMessage}</pre>`;
            } catch (e) {
                formattedMessage = `[Object: ${e.message}]`;
                entry.innerHTML = `<span style="color:#777">[${timestamp}]</span> ${formattedMessage}`;
            }
        } else {
            formattedMessage = String(message);
            entry.innerHTML = `<span style="color:#777">[${timestamp}]</span> ${formattedMessage}`;
        }
        
        log.appendChild(entry);
        log.scrollTop = log.scrollHeight; // Scroll to bottom
        
        // If debug panel is hidden, show a notification on the toggle button
        if (document.getElementById('debug-panel').style.display === 'none') {
            toggleButton.style.backgroundColor = color;
            setTimeout(() => {
                toggleButton.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
            }, 500);
        }
    }
    
    // Create global debug logging function
    window.debugLog = function(message, type = 'info') {
        isLogging = true;
        try {
            internalLog(message, type);
            
            // Show the debug panel automatically on error
            if (type === 'error' && document.getElementById('debug-panel').style.display === 'none') {
                document.getElementById('debug-panel').style.display = 'block';
                toggleButton.style.display = 'none';
            }
        } finally {
            isLogging = false;
        }
    };
    
    // Intercept console.log and other console methods
    const originalConsoleLog = console.log;
    const originalConsoleError = console.error;
    const originalConsoleWarn = console.warn;
    const originalConsoleInfo = console.info;
    
    console.log = function() {
        originalConsoleLog.apply(console, arguments);
        isLogging = true;
        try {
            internalLog(Array.from(arguments).join(' '), 'info');
        } finally {
            isLogging = false;
        }
    };
    
    console.error = function() {
        originalConsoleError.apply(console, arguments);
        isLogging = true;
        try {
            internalLog(Array.from(arguments).join(' '), 'error');
            
            // Auto-show panel on error
            if (document.getElementById('debug-panel').style.display === 'none') {
                document.getElementById('debug-panel').style.display = 'block';
                toggleButton.style.display = 'none';
            }
        } finally {
            isLogging = false;
        }
    };
    
    console.warn = function() {
        originalConsoleWarn.apply(console, arguments);
        isLogging = true;
        try {
            internalLog(Array.from(arguments).join(' '), 'warning');
        } finally {
            isLogging = false;
        }
    };
    
    console.info = function() {
        originalConsoleInfo.apply(console, arguments);
        isLogging = true;
        try {
            internalLog(Array.from(arguments).join(' '), 'info');
        } finally {
            isLogging = false;
        }
    };
    
    // Add a debug info method for WebRTC-specific events
    window.debugEvent = function(event, data) {
        isLogging = true;
        try {
            internalLog(`Event: ${event}`, 'event');
            if (data) {
                internalLog(data, 'data');
            }
        } finally {
            isLogging = false;
        }
    };
    
    // Add an explicit debug message
    internalLog('Debug panel initialized and ready', 'success');
}); 