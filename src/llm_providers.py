import os
import json
from typing import Optional, List
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        pass


class OllamaProvider(LLMProvider):
    
    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("Install requests: pip install requests")
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        try:
            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.0
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")


class HuggingFaceProvider(LLMProvider):
    
    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.2", api_key: Optional[str] = None):
        self.model_name = model
        self.api_key = api_key or os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Set HF_API_KEY environment variable or pass api_key")
        
        try:
            import requests
            self.requests = requests
            self.api_url = "https://router.huggingface.co/v1/chat/completions"
            print(f"✓ Hugging Face API ready with {model}")
        except ImportError:
            raise ImportError("Install requests: pip install requests")
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        response = self.requests.post(self.api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
        return str(result).strip()


class LlamaCppProvider(LLMProvider):
    """
    LlamaCpp - Run GGUF models locally
    Installation: pip install llama-cpp-python
    Download models from: https://huggingface.co/TheBloke
    """
    
    def __init__(self, model_path: str, n_ctx: int = 2048):
        self.model_path = model_path
        
        try:
            from llama_cpp import Llama
            
            print(f"Loading GGUF model: {model_path}...")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=4,
                verbose=False
            )
            print("Model loaded!")
            
        except ImportError:
            raise ImportError("Install llama-cpp-python: pip install llama-cpp-python")
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate using LlamaCpp."""
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            echo=False,
            stop=["</s>", "\n\n"]
        )
        
        return output["choices"][0]["text"].strip()


class GroqProvider(LLMProvider):
    """
    Groq Cloud - FREE API with fast inference
    Models: llama-3.1-8b-instant, mixtral-8x7b-32768
    Get free API key: https://console.groq.com
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("Set GROQ_API_KEY environment variable or pass api_key")
        
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
        except ImportError:
            raise ImportError("Install groq: pip install groq")
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Call Groq API (FREE and fast!)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content.strip()


def get_llm_provider(provider: str = "ollama", **kwargs) -> LLMProvider:
    """
    Factory function to get LLM provider.
    
    Args:
        provider: "ollama", "huggingface", "llamacpp", "groq"
        **kwargs: Provider-specific arguments
    
    Examples:
        >>> llm = get_llm_provider("ollama", model="llama3.2:3b")
        >>> llm = get_llm_provider("huggingface", model="google/flan-t5-base")
        >>> llm = get_llm_provider("groq", api_key="your-key")
    """
    providers = {
        "ollama": OllamaProvider,
        "huggingface": HuggingFaceProvider,
        "llamacpp": LlamaCppProvider,
        "groq": GroqProvider
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(providers.keys())}")
    
    return providers[provider](**kwargs)


def classify_with_llm(
    llm: LLMProvider,
    blurb: str,
    tags: List[str],
    valid_categories: List[str]
) -> str:
    """
    Use LLM to classify story with constrained output.
    
    Args:
        llm: LLM provider instance
        blurb: Story description
        tags: User tags
        valid_categories: List of valid category names
    
    Returns:
        Category name from valid_categories or "[UNMAPPED]"
    """
    
    # Build constrained prompt
    categories_str = ", ".join(valid_categories[:20])  # Limit to avoid token limits
    if len(valid_categories) > 20:
        categories_str += f" ... ({len(valid_categories)} total)"
    
    prompt = f"""Task: Classify a story into ONE category from the list below.

VALID CATEGORIES:
{categories_str}

STORY DESCRIPTION: {blurb}

USER TAGS: {', '.join(tags)}

RULES:
1. Output ONLY the category name (no explanation)
2. Choose from the valid categories list only
3. If the story is non-fiction (how-to, recipe, guide, tutorial), output: [UNMAPPED]
4. If uncertain, choose the closest match

OUTPUT (one category only):"""
    
    try:
        response = llm.generate(prompt, max_tokens=20)
        
        # Clean up response
        response = response.strip().split('\n')[0]  # Take first line only
        
        # Validate against categories (fuzzy match)
        if response in valid_categories:
            return response
        
        # Try case-insensitive match
        for cat in valid_categories:
            if cat.lower() == response.lower():
                return cat
        
        # Fuzzy match for typos
        from difflib import get_close_matches
        matches = get_close_matches(response, valid_categories, n=1, cutoff=0.8)
        if matches:
            return matches[0]
        
        # Default to unmapped if no match
        return "[UNMAPPED]"
        
    except Exception as e:
        print(f"LLM error: {e}")
        return "[UNMAPPED]"


if __name__ == "__main__":
    print("Testing Free LLM Providers...\n")
    
    # Test prompt
    test_prompt = "What is 2+2? Answer with just the number:"
    
    # Test Ollama (if running)
    try:
        print("1. Testing Ollama...")
        ollama = OllamaProvider(model="llama3.2:3b")
        result = ollama.generate(test_prompt, max_tokens=10)
        print(f"   Result: {result}\n")
    except Exception as e:
        print(f"   Skipped (not running): {e}\n")
    
    # Test Groq (if API key available)
    try:
        print("2. Testing Groq (FREE API)...")
        groq = GroqProvider()
        result = groq.generate(test_prompt, max_tokens=10)
        print(f"   Result: {result}\n")
    except Exception as e:
        print(f"   Skipped (no API key): {e}\n")
    
    print("Setup instructions:")
    print("- Ollama: Install from https://ollama.ai, then run: ollama pull llama3.2:3b")
    print("- Groq: Get free API key from https://console.groq.com")
    print("- Hugging Face: Automatically downloads models on first use")
