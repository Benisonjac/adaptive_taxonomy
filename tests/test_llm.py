import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from hybrid_mapper_v2 import HybridMapper
from config import Config

def test_with_huggingface():
    print("=" * 60)
    print("Testing with HuggingFace API (FREE Cloud API)")
    print("=" * 60)
    
    try:
        mapper = HybridMapper(llm_provider="huggingface")
        
        result = mapper.map_story(
            blurb="In a neon-lit dystopian city, a hacker falls in love with an AI.",
            user_tags=["Love", "Technology"]
        )
        
        print(f"\n✓ Result: {result.subgenre}")
        print(f"  Source: {result.source}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Reasoning: {result.reasoning}\n")
        
    except Exception as e:
        print(f"\n⚠ HuggingFace test failed: {e}")
        print("  Make sure you've set HF_API_KEY environment variable\n")


def test_with_groq():
    print("=" * 60)
    print("Testing with Groq (FREE Cloud API)")
    print("=" * 60)
    
    try:
        mapper = HybridMapper(llm_provider="groq")
        
        # Test ambiguous case (should use Tier 2)
        result = mapper.map_story(
            blurb="In a neon-lit dystopian city, a hacker falls in love with an AI.",
            user_tags=["Love", "Technology"]
        )
        
        print(f"\n✓ Result: {result.subgenre}")
        print(f"  Source: {result.source}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Reasoning: {result.reasoning}\n")
        
    except Exception as e:
        print(f"\n⚠ Groq test failed: {e}")
        print("  Make sure you've set GROQ_API_KEY environment variable\n")


def test_with_ollama():
    print("=" * 60)
    print("Testing with Ollama (FREE Local)")
    print("=" * 60)
    
    try:
        mapper = HybridMapper(llm_provider="ollama")
        
        # Test ambiguous case
        result = mapper.map_story(
            blurb="Learn how to bake perfect sourdough bread with fermentation tips.",
            user_tags=["Cooking", "Tutorial"]
        )
        
        print(f"\n✓ Result: {result.subgenre}")
        print(f"  Source: {result.source}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Reasoning: {result.reasoning}\n")
        
    except Exception as e:
        print(f"\n⚠ Ollama test failed: {e}")
        print("  Make sure Ollama is running: ollama serve")
        print("  And you have a model: ollama pull llama3.2:3b\n")


def test_vector_only():
    print("=" * 60)
    print("Testing with Vector Search Only (No LLM)")
    print("=" * 60)
    
    mapper = HybridMapper(llm_provider=None)
    
    result = mapper.map_story(
        blurb="Agent Smith infiltrates the enemy base to steal classified documents.",
        user_tags=["Spy", "Action"]
    )
    
    print(f"\n✓ Result: {result.subgenre}")
    print(f"  Source: {result.source}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Reasoning: {result.reasoning}\n")


if __name__ == "__main__":
    print("\n🚀 Testing LLM Providers\n")
    
    test_vector_only()
    test_with_huggingface()
    test_with_groq()
    test_with_ollama()
    
    print("\n" + "=" * 60)
    print("Setup Guide:")
    print("  - HuggingFace: Get FREE API key at https://huggingface.co/settings/tokens")
    print("  - Groq: Get FREE API key at https://console.groq.com")
    print("  - Ollama: Install from https://ollama.ai")
    print("  - See SETUP_LLM.md for detailed instructions")
    print("=" * 60)
