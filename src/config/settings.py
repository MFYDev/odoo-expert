from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Settings
    API_VERSION: str = "0.0.1"
    API_TITLE: str = "Odoo Expert API"
    API_DESCRIPTION: str = "API for querying Odoo documentation with RAG-powered responses"

    # The provider for the AI models to use.
    MODEL_PROVIDER: str = "openai"

    # Ollama Settings
    OLLAMA_BASE_URL: str
    OLLAMA_REQUEST_TIMEOUT: int

    # The name of LLM model to use.
    MODEL: str = "gpt-4o"

    # Name of the embedding model to use.
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Dimension of the embedding model to use.
    EMBEDDING_DIM: int = 1024

    # OpenAI Settings
    OPENAI_API_KEY: str = ""
    OPENAI_API_BASE: str = ""

    # PostgreSQL Settings
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "odoo_expert"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    
    # Security
    BEARER_TOKEN: str = ""
    CORS_ORIGINS: str = "*"
    
    # Odoo Settings
    ODOO_VERSIONS: str = "16.0,17.0,18.0"
    
    # Chat Settings
    SYSTEM_PROMPT: str
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    RAW_DATA_DIR: str = "raw_data"
    MARKDOWN_DATA_DIR: str = "markdown"
    
    @property
    def bearer_tokens_list(self) -> List[str]:
        if not self.BEARER_TOKEN:
            return []
        return [x.strip() for x in self.BEARER_TOKEN.split(',') if x.strip()]
    
    @property
    def cors_origins_list(self) -> List[str]:
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [x.strip() for x in self.CORS_ORIGINS.split(',') if x.strip()]
    
    @property
    def odoo_versions_list(self) -> List[str]:
        return [x.strip() for x in self.ODOO_VERSIONS.split(',') if x.strip()]
    
    class Config:
        env_file = ".env"

settings = Settings()
