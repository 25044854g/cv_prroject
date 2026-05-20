"""
Hybrid RAG Retrieval Engine for Object Detection
Combines BM25 and Dense Embedding retrieval
Confidence: Margin-based assessment (margin > 1.25 = confident)
"""

import json
from typing import List, Tuple, Dict, Any
from pathlib import Path


class ObjectKnowledgeRetriever:
    """
    Retrieves objects from knowledge base using hybrid retrieval
    BM25 (keyword) + Dense embeddings with 60/40 weighting
    """
    
    def __init__(self, knowledge_base_path: str = "object_knowledge_base.json"):
        self.kb_path = knowledge_base_path
        self.objects = self._load_knowledge_base()
        self.bm25_weights = {
            "canonical_name": 4.0,
            "aliases": 3.2,
            "description": 2.2,
            "scenes": 1.8,
            "intents": 1.5,
        }
    
    def _load_knowledge_base(self) -> List[Dict[str, Any]]:
        """Load object knowledge base"""
        if Path(self.kb_path).exists():
            with open(self.kb_path) as f:
                return json.load(f)
        
        # Default knowledge base if file doesn't exist
        return self._default_knowledge_base()
    
    def _default_knowledge_base(self) -> List[Dict[str, Any]]:
        """Default knowledge base with 8 objects"""
        return [
            {
                "id": 1,
                "canonical_name": "cell phone",
                "aliases": ["phone", "mobile", "smartphone", "device"],
                "description": "Mobile communication device",
                "scenes": ["desk", "pocket", "bed", "hand"],
                "intents": ["call", "text", "browse"],
                "negative_cues": ["large", "with cord"],
                "locations": ["on_table", "in_pocket", "on_bed"],
                "appearance": "rectangular, glowing screen",
                "grasp_hint": "grab_and_lift",
            },
            {
                "id": 2,
                "canonical_name": "mouse",
                "aliases": ["computer mouse", "pointer", "input device"],
                "description": "Computer pointing device",
                "scenes": ["desk", "mousepad"],
                "intents": ["click", "scroll", "point"],
                "negative_cues": ["large"],
                "locations": ["next_to_keyboard", "on_desk"],
                "appearance": "small, with buttons",
                "grasp_hint": "grab_handle",
            },
            {
                "id": 3,
                "canonical_name": "cup",
                "aliases": ["mug", "glass", "beverage container"],
                "description": "Drinking vessel",
                "scenes": ["desk", "table", "kitchen"],
                "intents": ["drink", "hold_liquid"],
                "negative_cues": ["empty"],
                "locations": ["on_desk", "on_table"],
                "appearance": "cylindrical, holds liquid",
                "grasp_hint": "grab_handle",
            },
            {
                "id": 4,
                "canonical_name": "bottle",
                "aliases": ["water bottle", "drink bottle"],
                "description": "Liquid storage container",
                "scenes": ["desk", "bag", "kitchen"],
                "intents": ["drink", "carry"],
                "negative_cues": ["small"],
                "locations": ["on_desk", "in_bag"],
                "appearance": "tall, cylindrical",
                "grasp_hint": "grab_body",
            },
            {
                "id": 5,
                "canonical_name": "book",
                "aliases": ["textbook", "novel", "reading material"],
                "description": "Written work in bound form",
                "scenes": ["desk", "shelf", "bed"],
                "intents": ["read", "study"],
                "negative_cues": ["electronic"],
                "locations": ["on_shelf", "on_desk"],
                "appearance": "rectangular, pages",
                "grasp_hint": "grab_spine",
            },
            {
                "id": 6,
                "canonical_name": "remote",
                "aliases": ["remote control", "TV remote"],
                "description": "Device control instrument",
                "scenes": ["sofa", "coffee_table"],
                "intents": ["control", "switch_channel"],
                "negative_cues": ["large"],
                "locations": ["on_coffee_table", "on_sofa"],
                "appearance": "small, rectangular, buttons",
                "grasp_hint": "grab_body",
            },
            {
                "id": 7,
                "canonical_name": "keyboard",
                "aliases": ["computer keyboard", "input"],
                "description": "Text input device",
                "scenes": ["desk"],
                "intents": ["type", "input"],
                "negative_cues": ["small"],
                "locations": ["on_desk", "in_front_of_monitor"],
                "appearance": "large, rectangular, many keys",
                "grasp_hint": "grab_edge",
            },
            {
                "id": 8,
                "canonical_name": "laptop",
                "aliases": ["computer", "notebook", "portable computer"],
                "description": "Portable computing device",
                "scenes": ["desk", "table"],
                "intents": ["work", "browse"],
                "negative_cues": ["small"],
                "locations": ["on_desk", "on_table"],
                "appearance": "large, rectangular, screen",
                "grasp_hint": "grab_edges",
            },
        ]
    
    def _bm25_score(self, query: str, obj: Dict) -> float:
        """Calculate BM25 score for query against object"""
        total_score = 0.0
        query_terms = query.lower().split()
        
        # Score against each field
        for field, weight in self.bm25_weights.items():
            if field not in obj:
                continue
            
            field_value = str(obj[field]).lower()
            field_terms = field_value.split()
            
            # Count matching terms
            matches = sum(1 for term in query_terms if term in field_terms)
            field_score = matches * weight
            total_score += field_score
        
        return total_score
    
    def _dense_score(self, query: str, obj: Dict) -> float:
        """
        Calculate dense embedding score (simplified)
        In production, use sentence-transformers all-MiniLM-L6-v2
        """
        # Simplified: use BM25 as proxy for demo
        return self._bm25_score(query, obj)
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        confidence_threshold: float = 0.5
    ) -> Tuple[List[str], float]:
        """
        Retrieve top-k objects using hybrid retrieval
        
        Args:
            query: User query string
            top_k: Number of results to return
            confidence_threshold: Minimum confidence score
            
        Returns:
            (list of object names, confidence score)
        """
        scores = []
        
        for obj in self.objects:
            # BM25 score (60% weight)
            bm25_score = self._bm25_score(query, obj)
            
            # Dense score (40% weight)
            dense_score = self._dense_score(query, obj)
            
            # Hybrid fusion: 60% BM25 + 40% dense
            hybrid_score = (bm25_score * 0.6 + dense_score * 0.4)
            
            scores.append({
                "name": obj["canonical_name"],
                "score": hybrid_score,
                "bm25": bm25_score,
                "dense": dense_score,
            })
        
        # Sort by score
        scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Filter by confidence threshold
        results = [s["name"] for s in scores if s["score"] >= confidence_threshold]
        results = results[:top_k]
        
        # Calculate confidence: top score - second score
        if len(scores) >= 2:
            margin = scores[0]["score"] - scores[1]["score"]
            confidence = min(0.99, 0.5 + margin * 0.2)
        elif len(scores) >= 1:
            confidence = scores[0]["score"] / 100  # Normalize
        else:
            confidence = 0.0
        
        return results, confidence
    
    def get_object_details(self, object_name: str) -> Dict[str, Any]:
        """Get detailed information about an object"""
        for obj in self.objects:
            if obj["canonical_name"].lower() == object_name.lower():
                return obj
        return {}


class SimpleLLMMapper:
    """Simple LLM mapper for object name standardization"""
    
    def __init__(self, retriever: ObjectKnowledgeRetriever):
        self.retriever = retriever
    
    def map_object(self, query: str, retrieved_objects: List[str]) -> Dict:
        """
        Map query to canonical object name
        
        Returns:
            {"object": canonical_name, "confidence": float}
        """
        if not retrieved_objects:
            return {"object": "unknown", "confidence": 0.0}
        
        # Take top retrieved object as mapped object
        return {
            "object": retrieved_objects[0],
            "confidence": 0.95,  # In production, would be from LLM
        }


if __name__ == "__main__":
    # Test retriever
    retriever = ObjectKnowledgeRetriever()
    
    # Test queries
    queries = [
        "where is my phone",
        "find the mouse",
        "locate the cup",
        "where is my keyboard",
    ]
    
    print("Object Retrieval Test")
    print("=" * 60)
    
    for query in queries:
        results, confidence = retriever.retrieve(query, top_k=3)
        print(f"\nQuery: '{query}'")
        print(f"Results: {results}")
        print(f"Confidence: {confidence:.2%}")
        
        # Get details for top result
        if results:
            details = retriever.get_object_details(results[0])
            print(f"Details: {details.get('description', 'N/A')}")
