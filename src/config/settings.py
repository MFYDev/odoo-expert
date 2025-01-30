from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Settings
    API_VERSION: str = "0.0.1"
    API_TITLE: str = "Odoo Expert API"
    API_DESCRIPTION: str = "API for querying Odoo documentation with RAG-powered responses"
    
    # OpenAI Settings
    OPENAI_API_KEY: str
    OPENAI_API_BASE: str
    LLM_MODEL: str = "gpt-4o"
    EMBEDDING_MODEL: str

    # PostgreSQL Settings
    POSTGRES_USER: str = "postgres"  # Changed default to match your config
    POSTGRES_PASSWORD: str = "postgres"  # Changed default to match your config
    POSTGRES_DB: str = "odoo_expert"  # Changed default to match your config
    POSTGRES_HOST: str = "localhost"  # Changed default to localhost
    POSTGRES_PORT: int = 5433  # Changed default to match your config
    
    # Security
    BEARER_TOKEN: str = ""
    CORS_ORIGINS: str = "*"
    
    # Chat Settings
    SYSTEM_PROMPT: str = """You are an expert in Odoo development and architecture.
    Answer the question using the provided documentation chunks and conversation history.
    In your answer:
    1. Start with a clear, direct response to the question
    2. Support your answer with specific references to the source documents
    3. Use markdown formatting for readability
    4. When citing information, mention which Source (1, 2, etc.) it came from
    5. If different sources provide complementary information, explain how they connect
    6. Consider the conversation history for context
    
    Format your response like this:
    
    **Answer:**
    [Your main answer here]
    
    **Sources Used:**
    - Source 1: Title chunk['url']
    - Source 2: Title chunk['url']
    - etc if needed
    """
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    
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
    
    class Config:
        env_file = ".env"

settings = Settings()