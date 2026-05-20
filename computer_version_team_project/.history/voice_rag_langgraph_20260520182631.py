"""
VoiceRAG LangGraph Integration
7-node state machine for voice-guided object detection
Integrates retrieval, LLM mapping, and confirmation
"""

from typing import Any, Dict, Optional, List
from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence classification"""
    HIGH = "high"       # margin > 1.25
    MEDIUM = "medium"   # 0.75 < margin <= 1.25
    LOW = "low"         # margin <= 0.75


class ObjectDetectionState:
    """State machine state with 12 tracked fields"""
    
    def __init__(self, user_input: str = ""):
        # Input
        self.user_input: str = user_input
        
        # Retrieval results
        self.retrieved_objects: List[str] = []
        self.retrieval_confidence: float = 0.0
        
        # Mapping results
        self.mapped_object: str = ""
        self.mapping_confidence: float = 0.0
        self.confidence_margin: float = 0.0
        self.confidence_level: str = ConfidenceLevel.MEDIUM.value
        
        # User confirmation
        self.requires_confirmation: bool = False
        self.user_confirmed: bool = False
        self.confirmation_attempts: int = 0
        
        # Final result
        self.target_object: Optional[str] = None
        self.final_confidence: float = 0.0
        
        # Fallback
        self.fallback_used: bool = False
        self.fallback_strategy: str = ""


class VoiceRAGLangGraph:
    """7-node LangGraph state machine for object detection"""
    
    def __init__(self, object_retriever, llm_mapper, voice_detector):
        self.object_retriever = object_retriever
        self.llm_mapper = llm_mapper
        self.voice_detector = voice_detector
    
    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full workflow"""
        user_input = input_dict.get("user_input", "")
        state = ObjectDetectionState(user_input)
        
        # Node 1: Retrieve
        state = self._node_retrieve(state)
        
        # Node 2: Assess
        state = self._node_assess(state)
        
        if state.requires_confirmation:
            # Node 3: Confirm
            state = self._node_map_object(state)
            
            # Node 4: Process confirmation
            state = self._node_confirm(state)
        else:
            # Direct mapping
            state = self._node_map_object(state)
        
        # Node 5: Process result
        state = self._node_process_result(state)
        
        # Node 6: Fallback (if needed)
        if state.target_object is None:
            state = self._node_fallback(state)
        
        # Node 7: Finish
        state = self._node_finish(state)
        
        return {
            "target_object": state.target_object,
            "confidence": state.final_confidence,
            "requires_confirmation": state.requires_confirmation,
            "fallback_used": state.fallback_used,
        }
    
    def _node_retrieve(self, state: ObjectDetectionState) -> ObjectDetectionState:
        """Node 1: Retrieve candidate objects from knowledge base"""
        query = state.user_input
        retrieved, confidence = self.object_retriever.retrieve(query)
        
        state.retrieved_objects = retrieved
        state.retrieval_confidence = confidence
        
        return state
    
    def _node_assess(self, state: ObjectDetectionState) -> ObjectDetectionState:
        """Node 2: Assess confidence and decide if confirmation needed"""
        # Assess confidence margin
        if len(state.retrieved_objects) >= 2:
            # Margin between top 2 candidates
            top_candidate = state.retrieved_objects[0]
            second_candidate = state.retrieved_objects[1]
            
            # Simplified margin calculation
            state.confidence_margin = state.retrieval_confidence - 0.2
        else:
            state.confidence_margin = state.retrieval_confidence
        
        # Classify confidence
        if state.confidence_margin > 1.25:
            state.confidence_level = ConfidenceLevel.HIGH.value
            state.requires_confirmation = False
        elif state.confidence_margin > 0.75:
            state.confidence_level = ConfidenceLevel.MEDIUM.value
            state.requires_confirmation = True
        else:
            state.confidence_level = ConfidenceLevel.LOW.value
            state.requires_confirmation = True
        
        return state
    
    def _node_map_object(self, state: ObjectDetectionState) -> ObjectDetectionState:
        """Node 3: Map query to canonical object name using LLM"""
        query = state.user_input
        context = state.retrieved_objects
        
        result = self.llm_mapper.map_object(query, context)
        
        state.mapped_object = result.get("object", "")
        state.mapping_confidence = result.get("confidence", 0.0)
        
        return state
    
    def _node_confirm(self, state: ObjectDetectionState) -> ObjectDetectionState:
        """Node 4: Get user confirmation via voice"""
        if state.requires_confirmation:
            # Ask user for confirmation
            confirmed = self.voice_detector.request_confirmation(
                state.mapped_object
            )
            state.user_confirmed = confirmed
            state.confirmation_attempts += 1
        
        return state
    
    def _node_process_result(self, state: ObjectDetectionState) -> ObjectDetectionState:
        """Node 5: Process result based on confidence and confirmation"""
        if state.requires_confirmation and not state.user_confirmed:
            state.target_object = None
        else:
            state.target_object = state.mapped_object
            state.final_confidence = state.mapping_confidence
        
        return state
    
    def _node_fallback(self, state: ObjectDetectionState) -> ObjectDetectionState:
        """Node 6: Apply fallback if primary failed"""
        # Fallback strategies:
        # 1. Try YOLO-Pose detection if available
        # 2. Return top retrieved object
        # 3. Ask user to repeat
        
        if state.retrieved_objects:
            state.target_object = state.retrieved_objects[0]
            state.final_confidence = state.retrieval_confidence * 0.9
            state.fallback_used = True
            state.fallback_strategy = "top_candidate"
        
        return state
    
    def _node_finish(self, state: ObjectDetectionState) -> ObjectDetectionState:
        """Node 7: Complete workflow"""
        # Log result, update metrics, etc.
        return state


class ObjectRetrieverMock:
    """Mock object retriever for testing"""
    
    def retrieve(self, query: str) -> tuple:
        """Retrieve candidate objects"""
        candidates = ["cell phone", "mobile device", "smartphone"]
        confidence = 0.92
        return candidates, confidence


class LLMMapperMock:
    """Mock LLM mapper for testing"""
    
    def map_object(self, query: str, context: List[str]) -> Dict:
        """Map query to object"""
        return {
            "object": "cell phone",
            "confidence": 0.95,
        }


class VoiceDetectorMock:
    """Mock voice detector for testing"""
    
    def request_confirmation(self, object_name: str) -> bool:
        """Request user confirmation"""
        return True


def create_voice_rag_langgraph():
    """Factory function"""
    retriever = ObjectRetrieverMock()
    mapper = LLMMapperMock()
    detector = VoiceDetectorMock()
    
    return VoiceRAGLangGraph(retriever, mapper, detector)


if __name__ == "__main__":
    # Test the LangGraph
    rag = create_voice_rag_langgraph()
    
    result = rag.invoke({"user_input": "where is my phone"})
    
    print("Result:")
    print(f"  Target object: {result['target_object']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Requires confirmation: {result['requires_confirmation']}")
    print(f"  Fallback used: {result['fallback_used']}")
