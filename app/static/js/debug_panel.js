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
        width: 800px;
        max-height: 600px;
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
        align-items: center;
        margin-bottom: 10px;
        padding-bottom: 5px;
        border-bottom: 1px solid #00ff00;
        position: relative;
        padding-right: 30px; /* Make room for X button */
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
        position: absolute;
        top: 0;
        right: 0;
        font-size: 16px;
        padding: 0;
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
    introMsg.textContent = 'Debug information will appear here. Interact with the app to see messages.';
    debugPanel.appendChild(introMsg);
    
    // Create the log area
    const logArea = document.createElement('div');
    logArea.id = 'debug-log';
    logArea.style.cssText = `
        max-height: 500px;
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
    
    // Add autoshow checkbox
    const autoShowContainer = document.createElement('div');
    autoShowContainer.style.cssText = `
        display: flex;
        align-items: center;
    `;
    
    const autoShowCheckbox = document.createElement('input');
    autoShowCheckbox.type = 'checkbox';
    autoShowCheckbox.id = 'auto-show-debug';
    autoShowCheckbox.checked = false;
    autoShowCheckbox.style.marginRight = '5px';
    
    const autoShowLabel = document.createElement('label');
    autoShowLabel.htmlFor = 'auto-show-debug';
    autoShowLabel.textContent = 'Auto-show on messages';
    autoShowLabel.style.fontSize = '11px';
    autoShowLabel.style.color = '#aaffaa';
    
    autoShowContainer.appendChild(autoShowCheckbox);
    autoShowContainer.appendChild(autoShowLabel);
    
    footerControls.appendChild(clearBtn);
    footerControls.appendChild(autoShowContainer);
    debugPanel.appendChild(footerControls);
    
    // Add to the document
    document.body.appendChild(debugPanel);
    
    // Create toggle button
    const toggleButton = document.createElement('button');
    toggleButton.textContent = 'Debug Panel';
    toggleButton.id = 'debug-panel-toggle';
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
        display: block;
    `;
    
    toggleButton.onclick = function() {
        const panel = document.getElementById('debug-panel');
        if (panel.style.display === 'none') {
            panel.style.display = 'block';
            toggleButton.style.display = 'none';
        }
    };
    
    document.body.appendChild(toggleButton);
    
    // Store references to elements globally for easier access
    window.debugPanel = debugPanel;
    window.debugPanelToggle = toggleButton;
    
    // Flag to prevent recursion
    let isLogging = false;
    
    // Internal log function to prevent recursion
    function internalLog(message, type = 'info') {
        if (isLogging) return; // Prevent recursion
        
        const log = document.getElementById('debug-log');
        if (!log) return;
        
        // Never show the panel automatically - just log the message
                
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
    }
    
    // Create global debug logging function
    window.debugLog = function(message, type = 'info') {
        isLogging = true;
        try {
            // Never force the debug panel to be visible
            // Just log the message silently
            internalLog(message, type);
            
            // Only flash the toggle button if debug panel is hidden
            if (toggleButton && document.getElementById('debug-panel').style.display === 'none') {
                let color = '#aaffaa';
                switch(type.toLowerCase()) {
                    case 'error': color = '#ff5555'; break;
                    case 'warning': color = '#ffaa55'; break;
                    case 'success': color = '#55ff55'; break;
                    case 'event': color = '#55aaff'; break;
                    case 'data': color = '#aa55ff'; break;
                }
                
                toggleButton.style.backgroundColor = color;
                setTimeout(() => {
                    toggleButton.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
                }, 500);
            }
        } finally {
            isLogging = false;
        }
    };
    
    // Log a startup message to verify the panel is working
    setTimeout(() => {
        // Direct approach to show messages
        const logElement = document.getElementById('debug-log');
        if (logElement) {
            const testEntry = document.createElement('div');
            testEntry.style.color = '#55ff55';
            testEntry.style.marginBottom = '3px';
            testEntry.innerHTML = `<span style="color:#777">[${new Date().toLocaleTimeString()}]</span> Debug panel initialized and ready`;
            logElement.appendChild(testEntry);
            logElement.scrollTop = logElement.scrollHeight;
        } else {
            console.error("Could not find debug-log element");
        }
        
        // Log startup without showing panel
        window.debugLog('Debug panel initialized and ready', 'success');
        
        // Test the panel is working
        window.forceDebugMessage = function(message, type = 'info', show = false) {
            try {
                const logElement = document.getElementById('debug-log');
                if (!logElement) return;
                
                // Only show the panel if explicitly requested with show=true
                // This should be used very sparingly
                if (show === true) {
                    const panel = document.getElementById('debug-panel');
                    if (panel) {
                        panel.style.display = 'block';
                        if (window.debugPanelToggle) {
                            window.debugPanelToggle.style.display = 'none';
                        }
                    }
                }
                
                // Create entry
                const entry = document.createElement('div');
                const timestamp = new Date().toLocaleTimeString();
                
                // Set color based on type
                let color = '#aaffaa';
                switch(type.toLowerCase()) {
                    case 'error': color = '#ff5555'; break;
                    case 'warning': color = '#ffaa55'; break;
                    case 'success': color = '#55ff55'; break;
                    case 'event': color = '#55aaff'; break;
                    case 'data': color = '#aa55ff'; break;
                }
                
                entry.style.color = color;
                entry.style.marginBottom = '3px';
                entry.innerHTML = `<span style="color:#777">[${timestamp}]</span> ${message}`;
                
                logElement.appendChild(entry);
                logElement.scrollTop = logElement.scrollHeight;
                
                // If debug panel is hidden, flash the toggle button without showing panel
                if (!show && toggleButton && document.getElementById('debug-panel').style.display === 'none') {
                    toggleButton.style.backgroundColor = color;
                    setTimeout(() => {
                        toggleButton.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
                    }, 500);
                }
                
                return true;
            } catch (error) {
                console.error("Force debug message error:", error);
                return false;
            }
        };
        
        // Add initial messages without showing the panel
        window.forceDebugMessage('Debug panel is now active', 'success', false);
        window.forceDebugMessage('Test the microphone by clicking the Test Microphone button', 'info', false);
        window.forceDebugMessage('Start a session to begin talking with the AI', 'info', false);
    }, 500);
    
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
            
            // Don't auto-show, just flash the button to indicate errors
            if (toggleButton && document.getElementById('debug-panel').style.display === 'none') {
                toggleButton.style.backgroundColor = '#ff5555';
                setTimeout(() => {
                    toggleButton.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
                }, 500);
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
            // Never auto-show the panel, just flash button if needed
            if (toggleButton && document.getElementById('debug-panel').style.display === 'none') {
                toggleButton.style.backgroundColor = '#ffaa55';
                setTimeout(() => {
                    toggleButton.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
                }, 500);
            }
        } finally {
            isLogging = false;
        }
    };
    
    console.info = function() {
        originalConsoleInfo.apply(console, arguments);
        isLogging = true;
        try {
            internalLog(Array.from(arguments).join(' '), 'info');
            // Never auto-show the panel
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
}); 