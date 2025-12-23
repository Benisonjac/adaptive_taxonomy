import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Config:
    VECTOR_MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    CONFIDENCE_THRESHOLD: float = 0.35
    AMBIGUITY_GAP: float = 0.05
    
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
    
    HF_API_KEY: Optional[str] = os.getenv("HF_API_KEY")
    HUGGINGFACE_MODEL: str = "meta-llama/Llama-3.2-3B-Instruct"
    
    OLLAMA_MODEL: str = "llama3.2:3b"
    OLLAMA_URL: str = "http://localhost:11434"
    
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    
    LLAMACPP_MODEL_PATH: Optional[str] = os.getenv("LLAMACPP_MODEL_PATH")
    
    LLM_MAX_TOKENS: int = 20
    LLM_TEMPERATURE: float = 0.0
    
    ENABLE_MONITORING: bool = True
    LOG_PREDICTIONS: bool = False
    
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TAXONOMY_FILE: str = os.path.join(BASE_DIR, "data", "taxonomy.json")
    TEST_CASES_FILE: str = os.path.join(BASE_DIR, "data", "test_cases.json")
    
    @classmethod
    def get_llm_kwargs(cls) -> dict:
        if cls.LLM_PROVIDER == "huggingface":
            return {
                "api_key": cls.HF_API_KEY,
                "model": cls.HUGGINGFACE_MODEL
            }
        elif cls.LLM_PROVIDER == "ollama":
            return {
                "model": cls.OLLAMA_MODEL,
                "base_url": cls.OLLAMA_URL
            }
        elif cls.LLM_PROVIDER == "groq":
            return {
                "api_key": cls.GROQ_API_KEY,
                "model": cls.GROQ_MODEL
            }
        elif cls.LLM_PROVIDER == "llamacpp":
            return {
                "model_path": cls.LLAMACPP_MODEL_PATH
            }
        return {}
    
    @classmethod
    def print_config(cls):
        print("Current Configuration:")
        print(f"  Vector Model: {cls.VECTOR_MODEL_NAME}")
        print(f"  LLM Provider: {cls.LLM_PROVIDER}")
        
        if cls.LLM_PROVIDER == "huggingface":
            print(f"  HuggingFace Model: {cls.HUGGINGFACE_MODEL}")
            print(f"  HF API Key: {'Set' if cls.HF_API_KEY else 'Not Set'}")
        elif cls.LLM_PROVIDER == "ollama":
            print(f"  Ollama Model: {cls.OLLAMA_MODEL}")
            print(f"  Ollama URL: {cls.OLLAMA_URL}")
        elif cls.LLM_PROVIDER == "groq":
            print(f"  Groq Model: {cls.GROQ_MODEL}")
            print(f"  Groq API Key: {'Set' if cls.GROQ_API_KEY else 'Not Set'}")
        
        print(f"  Confidence Threshold: {cls.CONFIDENCE_THRESHOLD}")
        print(f"  Ambiguity Gap: {cls.AMBIGUITY_GAP}")
