"""
Simple API server using FastAPI
Deploy this for production use
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    print("FastAPI not installed. Run: pip install fastapi uvicorn")
    FASTAPI_AVAILABLE = False
    exit(1)

from hybrid_mapper_v2 import HybridMapper, MappingResult
from typing import List, Optional

# Initialize app
app = FastAPI(
    title="Adaptive Taxonomy Mapper API",
    description="Classify stories into taxonomy categories using hybrid AI (vector + LLM)",
    version="2.0.0"
)

print("Initializing mapper...")
mapper = HybridMapper(llm_provider=os.getenv("LLM_PROVIDER", "huggingface"))
print("✓ Mapper ready!")


# Request/Response models
class StoryRequest(BaseModel):
    blurb: str
    user_tags: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "blurb": "Agent Smith must recover the stolen drive without being detected.",
                "user_tags": ["Action", "Spies"]
            }
        }


class StoryResponse(BaseModel):
    parent_genre: str
    subgenre: str
    confidence: float
    source: str
    reasoning: str


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "Adaptive Taxonomy Mapper",
        "version": "2.0.0",
        "llm_provider": os.getenv("LLM_PROVIDER", "huggingface")
    }


@app.post("/classify", response_model=StoryResponse)
async def classify_story(request: StoryRequest):
    try:
        result = mapper.map_story(request.blurb, request.user_tags)
        
        return StoryResponse(
            parent_genre=result.parent_genre,
            subgenre=result.subgenre,
            confidence=result.confidence,
            source=result.source,
            reasoning=result.reasoning
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/batch")
async def classify_batch(stories: List[StoryRequest]):
    results = []
    
    for story in stories:
        try:
            result = mapper.map_story(story.blurb, story.user_tags)
            results.append(StoryResponse(
                parent_genre=result.parent_genre,
                subgenre=result.subgenre,
                confidence=result.confidence,
                source=result.source,
                reasoning=result.reasoning
            ))
        except Exception as e:
            results.append(StoryResponse(
                parent_genre="",
                subgenre="[ERROR]",
                confidence=0.0,
                source="Error",
                reasoning=str(e)
            ))
    
    return results


@app.get("/taxonomy")
async def get_taxonomy():
    return mapper.taxonomy


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_model": "loaded" if mapper.vector_model else "not available",
        "llm_provider": os.getenv("LLM_PROVIDER", "huggingface"),
        "llm_status": "ready" if mapper.llm else "simulated",
        "taxonomy_categories": len([
            sg for subgenres in mapper.taxonomy["Fiction"].values() 
            for sg in subgenres
        ])
    }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("Starting Adaptive Taxonomy Mapper API")
    print("="*60)
    print(f"LLM Provider: {os.getenv('LLM_PROVIDER', 'simulated')}")
    print("Docs: http://localhost:8000/docs")
    print("Health: http://localhost:8000/health")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
