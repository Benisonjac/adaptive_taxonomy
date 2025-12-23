"""
Legacy Taxonomy Mapper - Rule-Based Implementation
This is the original keyword-matching approach (kept for comparison).
"""

import json
from typing import Dict, List, Tuple


class RuleBasedMapper:
    """Simple rule-based taxonomy mapper using keyword matching."""
    
    def __init__(self, taxonomy_file: str = "taxonomy.json"):
        with open(taxonomy_file, 'r') as f:
            self.taxonomy = json.load(f)
        
        # Define keyword rules for each subgenre
        self.rules = self._build_keyword_rules()
    
    def _build_keyword_rules(self) -> Dict[str, List[str]]:
        """Define keyword patterns for each subgenre."""
        return {
            # Romance
            "Slow-burn": ["gradual", "slowly", "friends", "building", "develop"],
            "Enemies-to-Lovers": ["rival", "enemy", "enemies", "compete", "hate", "adversaries"],
            "Second Chance": ["reunite", "childhood", "former", "years apart", "rekindle"],
            
            # Thriller
            "Espionage": ["spy", "agent", "intelligence", "covert", "classified", "operative"],
            "Psychological": ["mind", "manipulation", "sanity", "paranoia", "psychological"],
            "Legal Thriller": ["lawyer", "courtroom", "trial", "legal", "judge", "attorney"],
            
            # Sci-Fi
            "Hard Sci-Fi": ["physics", "scientific", "faster-than-light", "causality", "quantum"],
            "Space Opera": ["space battle", "starship", "galaxy", "empire", "alien civilization"],
            "Cyberpunk": ["dystopian", "hacker", "neon", "cybernetic", "corporate control"],
            
            # Horror
            "Psychological Horror": ["mental", "dread", "psychological", "sanity", "fear"],
            "Gothic": ["mansion", "haunted", "ancestral", "gothic", "victorian", "decay"],
            "Slasher": ["killer", "masked", "murder", "brutal", "stalking", "camp"]
        }
    
    def _score_subgenre(self, text: str, keywords: List[str]) -> int:
        """Count how many keywords appear in the text."""
        text_lower = text.lower()
        return sum(1 for keyword in keywords if keyword in text_lower)
    
    def map_story(self, blurb: str, user_tags: List[str]) -> Tuple[str, str]:
        """Map story to taxonomy using keyword rules."""
        combined_text = f"{blurb} {' '.join(user_tags)}"
        
        # Non-fiction detection
        non_fiction_keywords = ["how to", "learn", "guide", "tutorial", "recipe"]
        if any(kw in combined_text.lower() for kw in non_fiction_keywords):
            return "", "[UNMAPPED]"
        
        # Score all subgenres
        scores = {}
        for subgenre, keywords in self.rules.items():
            scores[subgenre] = self._score_subgenre(combined_text, keywords)
        
        # Get best match
        if max(scores.values()) == 0:
            return "", "[UNMAPPED]"
        
        best_subgenre = max(scores, key=scores.get)
        
        # Find parent genre
        for parent, children in self.taxonomy["Fiction"].items():
            if best_subgenre in children:
                return parent, best_subgenre
        
        return "", "[UNMAPPED]"


def compare_approaches():
    """Compare rule-based vs hybrid approach."""
    print("Rule-based mapper loaded. Use hybrid_mapper_v2.py for better results.")


if __name__ == "__main__":
    compare_approaches()
