"""
Logger configuration for the voice chat application
"""

import logging
import os
from typing import Optional

class StatusEndpointFilter(logging.Filter):
    """
    Filter that removes all logs containing '/audio-bridge/status'
    """
    def filter(self, record):
        # Skip any log entries that contain the status endpoint path
        if hasattr(record, 'args') and isinstance(record.args, tuple) and len(record.args) > 0:
            if isinstance(record.args[0], str) and '/audio-bridge/status' in record.args[0]:
                return False
        
        # Also check the message itself
        if hasattr(record, 'msg') and isinstance(record.msg, str) and '/audio-bridge/status' in record.msg:
            return False
            
        return True

def configure_logging():
    """Configure the application logging with appropriate filters"""
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Add filter to hide audio bridge status requests
    status_filter = StatusEndpointFilter()
    
    # Apply filter to uvicorn and app loggers
    loggers_to_filter = [
        'uvicorn', 
        'uvicorn.access', 
        'uvicorn.error',
        'fastapi',
        'app.main'
    ]
    
    for logger_name in loggers_to_filter:
        logger = logging.getLogger(logger_name)
        logger.addFilter(status_filter)
    
    # Debug mode check
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    
    # Set level for our app logger
    app_logger = logging.getLogger('app')
    app_logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    
    return app_logger

# Export the main app logger
logger = configure_logging() 