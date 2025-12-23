import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from hybrid_mapper_v2 import HybridMapper, MappingResult


def example_single_story():
    print("=" * 60)
    print("Example 1: Single Story Classification")
    print("=" * 60)
    
    mapper = HybridMapper()
    
    result = mapper.map_story(
        blurb="A detective hunts a serial killer who leaves cryptic messages at crime scenes.",
        user_tags=["Mystery", "Dark"]
    )
    
    print(f"\nCategory: {result.parent_genre} > {result.subgenre}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Source: {result.source}")
    print(f"Reasoning: {result.reasoning}\n")


def example_batch_processing():
    print("=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)
    
    mapper = HybridMapper()
    
    stories = [
        {
            "title": "Digital Hearts",
            "blurb": "A programmer falls for an AI in a futuristic city.",
            "tags": ["Romance", "Tech"]
        },
        {
            "title": "Courtroom Justice",
            "blurb": "A defense attorney must prove her client's innocence.",
            "tags": ["Legal", "Drama"]
        }
    ]
    
    for story in stories:
        result = mapper.map_story(story["blurb"], story["tags"])
        print(f"\n{story['title']}: {result.subgenre} ({result.confidence:.2%})")


def example_confidence_analysis():
    print("=" * 60)
    print("Example 3: Confidence Analysis")
    print("=" * 60)
    
    mapper = HybridMapper()
    
    test_stories = [
        ("Agent infiltrates enemy base with classified intel.", ["Spy", "Action"]),
        ("A story about mixed themes and unclear direction.", ["General", "Fiction"]),
    ]
    
    for blurb, tags in test_stories:
        result = mapper.map_story(blurb, tags)
        
        confidence_level = "HIGH" if result.confidence > 0.7 else "MEDIUM" if result.confidence > 0.4 else "LOW"
        
        print(f"\nStory: {blurb[:50]}...")
        print(f"Predicted: {result.subgenre}")
        print(f"Confidence: {confidence_level} ({result.confidence:.2%})")
        print(f"Used: {result.source}")


if __name__ == "__main__":
    example_single_story()
    print("\n\n")
    example_batch_processing()
    print("\n\n")
    example_confidence_analysis()
