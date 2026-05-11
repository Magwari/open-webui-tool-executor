"""
Configuration for the Tool Executor Server.

Loads environment variables for OpenWebUI connection,
code execution settings, and server configuration.
"""

import os
from typing import Optional


class Settings:
    """Application settings loaded from environment variables."""

    # OpenWebUI Connection
    OPENWEBUI_BASE_URL: str = os.getenv('OPENWEBUI_BASE_URL', 'http://localhost:8080')
    OPENWEBUI_API_KEY: str = os.getenv('OPENWEBUI_API_KEY', '')

    # Tool Calling
    MAX_TOOL_ITERATIONS: int = int(os.getenv('MAX_TOOL_ITERATIONS', '10'))

    # Code Execution
    CODE_EXEC_TIMEOUT: int = int(os.getenv('CODE_EXEC_TIMEOUT', '60'))
    CODE_EXEC_MAX_WORKERS: int = int(os.getenv('CODE_EXEC_MAX_WORKERS', '5'))

    # Server
    SERVER_HOST: str = os.getenv('SERVER_HOST', '0.0.0.0')
    SERVER_PORT: int = int(os.getenv('SERVER_PORT', '8000'))

    # Request timeout for OpenWebUI API calls (seconds)
    OPENWEBUI_TIMEOUT: int = int(os.getenv('OPENWEBUI_TIMEOUT', '300'))

    def validate(self) -> None:
        """Validate required settings."""
        if not self.OPENWEBUI_API_KEY:
            raise ValueError(
                'OPENWEBUI_API_KEY is required. '
                'Set it via environment variable or create a .env file.'
            )
        if not self.OPENWEBUI_BASE_URL:
            raise ValueError('OPENWEBUI_BASE_URL is required.')


settings = Settings()