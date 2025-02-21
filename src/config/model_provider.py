from typing import Dict
from src.config.settings import settings
from src.utils.logging import logger

class ModelProvider:
    def __init__(self):
        model_provider = settings.MODEL_PROVIDER
        logger.info(f"Model provider: {model_provider}")
        logger.info(f"Embedding model: {settings.EMBEDDING_MODEL}")
        logger.info(f"LLM: {settings.MODEL}")
        match model_provider:
            case "openai":
                self.init_openai()
            case "groq":
                self.init_groq()
            case "ollama":
                self.init_ollama()
            case "anthropic":
                self.init_anthropic()
            case "gemini":
                self.init_gemini()
            case "mistral":
                self.init_mistral()
            case "azure-openai":
                self.init_azure_openai()
            case "huggingface":
                self.init_huggingface()
            case _:
                raise ValueError(f"Invalid model provider: {model_provider}")

    def init_ollama(self):
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding
            from llama_index.llms.ollama.base import DEFAULT_REQUEST_TIMEOUT, Ollama
        except ImportError:
            raise ImportError(
                "Ollama support is not installed. Please install it with `poetry add llama-index-llms-ollama` and `poetry add llama-index-embeddings-ollama`"
            )

        base_url = settings.OLLAMA_BASE_URL or "http://127.0.0.1:11434"
        request_timeout = float(
            settings.OLLAMA_REQUEST_TIMEOUT or DEFAULT_REQUEST_TIMEOUT
        )
        self.embed_model = OllamaEmbedding(
            base_url=base_url,
            model_name=settings.EMBEDDING_MODEL,
        )
        self.llm = Ollama(
            base_url=base_url, model=settings.MODEL, request_timeout=request_timeout
        )

    def init_openai(self):
        from llama_index.core.constants import DEFAULT_TEMPERATURE
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI

        max_tokens = settings.LLM_MAX_TOKENS
        model_name = settings.MODEL or "gpt-4o"
        self.llm = OpenAI(
            model=model_name,
            temperature=float(settings.LLM_TEMPERATURE or DEFAULT_TEMPERATURE),
            max_tokens=int(max_tokens) if max_tokens is not None else None,
        )

        dimensions = settings.EMBEDDING_DIM
        self.embed_model = OpenAIEmbedding(
            model=settings.EMBEDDING_MODEL or "text-embedding-3-small",
            dimensions=int(dimensions) if dimensions is not None else None,
        )

    def init_azure_openai(self):
        from llama_index.core.constants import DEFAULT_TEMPERATURE

        try:
            from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
            from llama_index.llms.azure_openai import AzureOpenAI
        except ImportError:
            raise ImportError(
                "Azure OpenAI support is not installed. Please install it with `poetry add llama-index-llms-azure-openai` and `poetry add llama-index-embeddings-azure-openai`"
            )

        llm_deployment = settings.AZURE_OPENAI_LLM_DEPLOYMENT
        embedding_deployment = settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        max_tokens = settings.LLM_MAX_TOKENS
        temperature = settings.LLM_TEMPERATURE or DEFAULT_TEMPERATURE
        dimensions = settings.EMBEDDING_DIM

        azure_config = {
            "api_key": settings.AZURE_OPENAI_API_KEY,
            "azure_endpoint": settings.AZURE_OPENAI_ENDPOINT,
            "api_version": settings.AZURE_OPENAI_API_VERSION or settings.OPENAI_API_VERSION,
        }

        self.llm = AzureOpenAI(
            model=settings.MODEL,
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            temperature=float(temperature),
            deployment_name=llm_deployment,
            **azure_config,
        )

        self.embed_model = AzureOpenAIEmbedding(
            model=settings.EMBEDDING_MODEL,
            dimensions=int(dimensions) if dimensions is not None else None,
            deployment_name=embedding_deployment,
            **azure_config,
        )

    def init_fastembed(self):
        try:
            from llama_index.embeddings.fastembed import FastEmbedEmbedding
        except ImportError:
            raise ImportError(
                "FastEmbed support is not installed. Please install it with `poetry add llama-index-embeddings-fastembed`"
            )

        embed_model_map: Dict[str, str] = {
            # Small and multilingual
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
            # Large and multilingual
            "paraphrase-multilingual-mpnet-base-v2": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        }

        embedding_model = settings.EMBEDDING_MODEL
        if embedding_model is None:
            raise ValueError("EMBEDDING_MODEL environment variable is not set")

        # This will download the model automatically if it is not already downloaded
        self.embed_model = FastEmbedEmbedding(
            model_name=embed_model_map[embedding_model]
        )

    def init_huggingface_embedding(self):
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        except ImportError:
            raise ImportError(
                "Hugging Face support is not installed. Please install it with `poetry add llama-index-embeddings-huggingface`"
            )

        embedding_model = settings.EMBEDDING_MODEL or "all-MiniLM-L6-v2"
        backend = settings.EMBEDDING_BACKEND or "onnx"  # "torch", "onnx", or "openvino"
        trust_remote_code = (
                (settings.EMBEDDING_TRUST_REMOTE_CODE or "false").lower() == "true"
        )

        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            trust_remote_code=trust_remote_code,
            backend=backend,
        )

    def init_huggingface(self):
        try:
            from llama_index.llms.huggingface import HuggingFaceLLM
        except ImportError:
            raise ImportError(
                "Hugging Face support is not installed. Please install it with `poetry add llama-index-llms-huggingface` and `poetry add llama-index-embeddings-huggingface`"
            )

        self.llm = HuggingFaceLLM(
            model_name=settings.MODEL,
            tokenizer_name=settings.MODEL,
        )
        self.init_huggingface_embedding()

    def init_groq(self):
        try:
            from llama_index.llms.groq import Groq
        except ImportError:
            raise ImportError(
                "Groq support is not installed. Please install it with `poetry add llama-index-llms-groq`"
            )

        self.llm = Groq(model=settings.MODEL)
        # Groq does not provide embeddings, so we use FastEmbed instead
        self.init_fastembed()

    def init_anthropic(self):
        try:
            from llama_index.llms.anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "Anthropic support is not installed. Please install it with `poetry add llama-index-llms-anthropic`"
            )

        model_map: Dict[str, str] = {
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-2.1": "claude-2.1",
            "claude-instant-1.2": "claude-instant-1.2",
        }

        self.llm = Anthropic(model=model_map[settings.MODEL])
        # Anthropic does not provide embeddings, so we use FastEmbed instead
        self.init_fastembed()

    def init_gemini(self):
        try:
            from llama_index.embeddings.gemini import GeminiEmbedding
            from llama_index.llms.gemini import Gemini
        except ImportError:
            raise ImportError(
                "Gemini support is not installed. Please install it with `poetry add llama-index-llms-gemini` and `poetry add llama-index-embeddings-gemini`"
            )

        model_name = f"models/{settings.MODEL}"
        embed_model_name = f"models/{settings.EMBEDDING_MODEL}"

        self.llm = Gemini(model=model_name)
        self.embed_model = GeminiEmbedding(model_name=embed_model_name)

    def init_mistral(self):
        from llama_index.embeddings.mistralai import MistralAIEmbedding
        from llama_index.llms.mistralai import MistralAI

        self.llm = MistralAI(model=settings.MODEL)
        self.embed_model = MistralAIEmbedding(model_name=settings.EMBEDDING_MODEL)