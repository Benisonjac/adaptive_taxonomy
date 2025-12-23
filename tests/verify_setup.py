"""
Project Setup Verification Script
Run this to check if everything is configured correctly
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (need 3.8+)")
        return False

def check_files():
    """Check if all core files exist"""
    os.chdir(os.path.dirname(os.path.dirname(__file__)))  # Go to project root
    
    files = {
        'src/hybrid_mapper_v2.py': 'Main mapper',
        'src/llm_providers.py': 'LLM providers',
        'src/config.py': 'Configuration',
        'src/api.py': 'API server',
        'data/taxonomy.json': 'Taxonomy data',
        'data/test_cases.json': 'Test cases',
        'tests/test_llm.py': 'LLM tests',
        'tests/examples.py': 'Examples',
        'requirements.txt': 'Dependencies',
        'README.md': 'Documentation',
        'docs/SETUP_LLM.md': 'LLM setup guide'
    }
    
    all_present = True
    for file, desc in files.items():
        if os.path.exists(file):
            print(f"✅ {file:<30} ({desc})")
        else:
            print(f"❌ {file:<30} MISSING")
            all_present = False
    
    return all_present

def check_dependencies():
    deps = {
        'numpy': 'numpy',
        'torch': 'torch',
        'sentence_transformers': 'sentence-transformers',
        'requests': 'requests'
    }
    
    missing = []
    for module, package in deps.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"⚠️  {package} (install with: pip install {package})")
            missing.append(package)
    
    return missing

def check_optional_dependencies():
    optional = {
        'groq': 'groq',
        'transformers': 'transformers',
        'llama_cpp': 'llama-cpp-python'
    }
    
    available = []
    for module, package in optional.items():
        try:
            __import__(module)
            print(f"✅ {package}")
            available.append(package)
        except ImportError:
            print(f"   {package} (optional)")
    
    return available

def check_environment():
    vars_to_check = ['LLM_PROVIDER', 'HF_API_KEY', 'GROQ_API_KEY', 'OLLAMA_MODEL']
    
    for var in vars_to_check:
        value = os.getenv(var)
        if value:
            if 'KEY' in var and len(value) > 10:
                display_value = value[:10] + "..." 
            else:
                display_value = value
            print(f"✅ {var}={display_value}")
        else:
            print(f"   {var} (not set)")

def main():
    print("="*60)
    print("🔍 Adaptive Taxonomy Mapper - Setup Verification")
    print("="*60)
    
    # Check Python
    print("\n📌 Python Version:")
    python_ok = check_python_version()
    
    # Check files
    print("\n📁 Core Files:")
    files_ok = check_files()
    
    # Check dependencies
    print("\n📦 Required Dependencies:")
    missing = check_dependencies()
    
    # Check optional
    print("\n🎁 Optional LLM Providers:")
    available = check_optional_dependencies()
    
    # Check environment
    print("\n⚙️  Environment Variables:")
    check_environment()
    
    # Summary
    print("\n" + "="*60)
    print("📊 Summary:")
    print("="*60)
    
    if python_ok and files_ok and not missing:
        print("✅ Core system ready!")
        print("\n🚀 Quick Start:")
        print("   python -m src.hybrid_mapper_v2      # Run tests")
        print("   python tests/test_llm.py            # Test LLM")
        print("   python src/api.py                   # Start API")
    elif missing:
        print(f"⚠️  Missing dependencies: {', '.join(missing)}")
        print("\n📥 Install them with:")
        print(f"   pip install {' '.join(missing)}")
        print("\nOr install everything:")
        print("   pip install -r requirements.txt")
    else:
        print("❌ Some issues found. Please fix them above.")
    
    if not available:
        print("\n💡 No LLM providers installed yet.")
        print("   See SETUP_LLM.md for setup instructions.")
        print("   Recommended: Groq (FREE cloud API)")
    
    print("\n📚 Documentation:")
    print("   README.md              - Main documentation")
    print("   docs/SETUP_LLM.md      - LLM setup guide")
    print("   docs/QUICKSTART.md     - Quick start")
    print("="*60)

if __name__ == "__main__":
    main()
