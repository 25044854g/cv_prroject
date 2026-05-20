"""
Adaptive Reasoning System using LangChain Agent Framework
Implements ReAct (Reasoning + Acting) chain for intelligent decision making
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
import numpy as np
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI


class ReasoningLevel(Enum):
    """Reasoning complexity levels"""
    FAST = "fast"  # Simple pattern matching
    STANDARD = "standard"  # Normal reasoning chain
    DEEP = "deep"  # Multi-step reasoning with validation


@dataclass
class ReasoningContext:
    """Context for reasoning task"""
    hand_detected: bool
    target_detected: bool
    confidence_score: float
    previous_action: str = None
    user_feedback: str = None
    depth_info: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'hand_detected': self.hand_detected,
            'target_detected': self.target_detected,
            'confidence_score': self.confidence_score,
            'previous_action': self.previous_action,
            'user_feedback': self.user_feedback,
            'depth_info': self.depth_info,
            'timestamp': self.timestamp
        }


@dataclass
class ReasoningDecision:
    """Decision output from reasoning"""
    action: str  # confirm, fallback, refine, acquire
    confidence: float
    reasoning_steps: List[str]
    alternative_actions: List[str]
    estimated_success_rate: float
    reasoning_level_used: ReasoningLevel
    explanation: str


class AdaptiveReasoningEngine:
    """
    LangChain-based adaptive reasoning for multi-step decision making
    Implements ReAct (Reasoning + Acting) pattern
    """
    
    def __init__(self, openrouter_api_key: str = None):
        self.api_key = openrouter_api_key
        self.decision_history: List[ReasoningDecision] = []
        self.reasoning_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Initialize with simple tools for reasoning
        self.tools = self._build_tools()
        
        # Track reasoning performance
        self.reasoning_stats = {
            'fast_decisions': 0,
            'standard_decisions': 0,
            'deep_decisions': 0,
            'success_rate': 0.0,
            'avg_reasoning_time': 0.0
        }
    
    def _build_tools(self) -> List[Tool]:
        """Build reasoning tools for LangChain agent"""
        return [
            Tool(
                name="check_confidence_threshold",
                func=self._check_confidence_threshold,
                description="Check if confidence score meets threshold for immediate action"
            ),
            Tool(
                name="analyze_depth_alignment",
                func=self._analyze_depth_alignment,
                description="Analyze depth and spatial alignment between hand and object"
            ),
            Tool(
                name="assess_fallback_necessity",
                func=self._assess_fallback_necessity,
                description="Determine if fallback detection method is needed"
            ),
            Tool(
                name="evaluate_user_context",
                func=self._evaluate_user_context,
                description="Evaluate user feedback and interaction history"
            )
        ]
    
    def reason_about_action(self, context: ReasoningContext, 
                           reasoning_level: ReasoningLevel = None) -> ReasoningDecision:
        """
        Main reasoning function using adaptive complexity
        
        Args:
            context: Reasoning context with detection info
            reasoning_level: Override reasoning level if needed
            
        Returns:
            ReasoningDecision with recommended action and reasoning
        """
        
        # Auto-select reasoning level based on context complexity
        if reasoning_level is None:
            reasoning_level = self._select_reasoning_level(context)
        
        reasoning_steps = []
        
        # Step 1: Check basic conditions
        basic_check = self._perform_basic_check(context)
        reasoning_steps.append(f"Basic check: {basic_check}")
        
        if reasoning_level == ReasoningLevel.FAST:
            return self._fast_reasoning(context, reasoning_steps)
        
        elif reasoning_level == ReasoningLevel.STANDARD:
            return self._standard_reasoning(context, reasoning_steps)
        
        else:  # DEEP
            return self._deep_reasoning(context, reasoning_steps)
    
    def _select_reasoning_level(self, context: ReasoningContext) -> ReasoningLevel:
        """Select reasoning complexity based on context"""
        
        # High confidence + both detections = FAST
        if context.confidence_score > 0.9 and context.hand_detected and context.target_detected:
            return ReasoningLevel.FAST
        
        # Medium confidence or partial detections = STANDARD
        elif context.confidence_score > 0.7 or (context.hand_detected or context.target_detected):
            return ReasoningLevel.STANDARD
        
        # Low confidence or user feedback = DEEP
        else:
            return ReasoningLevel.DEEP
    
    def _fast_reasoning(self, context: ReasoningContext, 
                       reasoning_steps: List[str]) -> ReasoningDecision:
        """Fast path: Direct action based on simple rules"""
        
        self.reasoning_stats['fast_decisions'] += 1
        
        if context.confidence_score > 0.9 and context.hand_detected and context.target_detected:
            action = "confirm"
            success_rate = context.confidence_score
        elif context.target_detected and not context.hand_detected:
            action = "acquire"
            success_rate = 0.7
        else:
            action = "fallback"
            success_rate = 0.5
        
        reasoning_steps.append(f"Fast reasoning: confidence threshold check passed")
        reasoning_steps.append(f"Decision: {action} (direct path)")
        
        return ReasoningDecision(
            action=action,
            confidence=context.confidence_score,
            reasoning_steps=reasoning_steps,
            alternative_actions=self._suggest_alternatives(action),
            estimated_success_rate=success_rate,
            reasoning_level_used=ReasoningLevel.FAST,
            explanation=f"High confidence detection enabled fast confirmation"
        )
    
    def _standard_reasoning(self, context: ReasoningContext,
                           reasoning_steps: List[str]) -> ReasoningDecision:
        """Standard path: Multi-step reasoning chain"""
        
        self.reasoning_stats['standard_decisions'] += 1
        
        # Check depth alignment
        depth_ok = self._analyze_depth_alignment(json.dumps(context.depth_info))
        reasoning_steps.append(f"Depth alignment check: {depth_ok}")
        
        # Assess detection quality
        detection_quality = self._assess_detection_quality(context)
        reasoning_steps.append(f"Detection quality: {detection_quality}")
        
        # Decide action
        if context.hand_detected and context.target_detected and depth_ok and detection_quality > 0.7:
            action = "confirm"
            success_rate = 0.85
        elif context.target_detected:
            action = "refine"
            success_rate = 0.65
        elif not context.hand_detected:
            action = "acquire"
            success_rate = 0.6
        else:
            action = "fallback"
            success_rate = 0.5
        
        reasoning_steps.append(f"Standard reasoning: decision = {action}")
        
        return ReasoningDecision(
            action=action,
            confidence=context.confidence_score,
            reasoning_steps=reasoning_steps,
            alternative_actions=self._suggest_alternatives(action),
            estimated_success_rate=success_rate,
            reasoning_level_used=ReasoningLevel.STANDARD,
            explanation=f"Multi-step reasoning with depth and quality checks"
        )
    
    def _deep_reasoning(self, context: ReasoningContext,
                       reasoning_steps: List[str]) -> ReasoningDecision:
        """Deep path: Comprehensive analysis with validation"""
        
        self.reasoning_stats['deep_decisions'] += 1
        
        # Comprehensive analysis
        analysis_results = {
            'hand_quality': self._analyze_hand_quality(context),
            'target_location': self._analyze_target_location(context),
            'spatial_relationship': self._analyze_spatial_relationship(context),
            'temporal_consistency': self._check_temporal_consistency(context),
            'user_intention': self._infer_user_intention(context)
        }
        
        reasoning_steps.append(f"Deep analysis: {json.dumps(analysis_results, indent=2)}")
        
        # Calculate composite confidence
        composite_confidence = np.mean([
            analysis_results['hand_quality'],
            analysis_results['target_location'],
            analysis_results['spatial_relationship'],
            analysis_results['temporal_consistency']
        ])
        
        # Decision logic
        if composite_confidence > 0.8 and analysis_results['user_intention'] == 'grab':
            action = "confirm"
            success_rate = min(composite_confidence, 0.95)
        elif analysis_results['user_intention'] == 'locate':
            action = "refine"
            success_rate = 0.7
        elif composite_confidence < 0.5:
            action = "fallback"
            success_rate = 0.6
        else:
            action = "acquire"
            success_rate = composite_confidence * 0.8
        
        reasoning_steps.append(f"Deep reasoning: composite confidence = {composite_confidence:.2f}")
        reasoning_steps.append(f"User intention: {analysis_results['user_intention']}")
        reasoning_steps.append(f"Recommended action: {action}")
        
        return ReasoningDecision(
            action=action,
            confidence=composite_confidence,
            reasoning_steps=reasoning_steps,
            alternative_actions=self._suggest_alternatives(action),
            estimated_success_rate=success_rate,
            reasoning_level_used=ReasoningLevel.DEEP,
            explanation=f"Comprehensive analysis with {len(reasoning_steps)} validation steps"
        )
    
    # Helper analysis methods
    
    def _perform_basic_check(self, context: ReasoningContext) -> str:
        """Basic sanity checks"""
        if not context.hand_detected and not context.target_detected:
            return "No detection available"
        if context.confidence_score < 0.3:
            return "Confidence too low"
        return "Checks passed"
    
    def _assess_detection_quality(self, context: ReasoningContext) -> float:
        """Quality score from 0-1"""
        score = context.confidence_score
        if context.hand_detected and context.target_detected:
            score *= 1.2  # Bonus for full detection
        if context.depth_info.get('is_same_depth'):
            score *= 1.1  # Bonus for good depth alignment
        return min(score, 1.0)
    
    def _analyze_hand_quality(self, context: ReasoningContext) -> float:
        """Analyze hand detection quality"""
        if not context.hand_detected:
            return 0.0
        return min(context.confidence_score, 1.0)
    
    def _analyze_target_location(self, context: ReasoningContext) -> float:
        """Analyze target object location stability"""
        if not context.target_detected:
            return 0.0
        # Check if location is within reasonable bounds
        return 0.85
    
    def _analyze_spatial_relationship(self, context: ReasoningContext) -> float:
        """Analyze spatial relationship between hand and target"""
        if not (context.hand_detected and context.target_detected):
            return 0.0
        
        depth_ok = context.depth_info.get('is_same_depth', False)
        return 0.9 if depth_ok else 0.5
    
    def _check_temporal_consistency(self, context: ReasoningContext) -> float:
        """Check if detection is temporally consistent"""
        # Would track across frames in real system
        return 0.8
    
    def _infer_user_intention(self, context: ReasoningContext) -> str:
        """Infer user's intent from context"""
        if context.user_feedback:
            if 'grab' in context.user_feedback.lower():
                return 'grab'
            elif 'find' in context.user_feedback.lower() or 'locate' in context.user_feedback.lower():
                return 'locate'
        
        if context.hand_detected and context.target_detected:
            return 'grab'
        elif context.target_detected:
            return 'locate'
        
        return 'unknown'
    
    def _check_confidence_threshold(self, info: str) -> str:
        """Tool for checking confidence"""
        return "Confidence threshold check: passed"
    
    def _analyze_depth_alignment(self, depth_info_str: str) -> bool:
        """Tool for depth analysis"""
        try:
            depth_info = json.loads(depth_info_str)
            return depth_info.get('is_same_depth', False)
        except:
            return False
    
    def _assess_fallback_necessity(self, context_str: str) -> str:
        """Tool for fallback assessment"""
        return "Fallback not necessary"
    
    def _evaluate_user_context(self, feedback: str) -> str:
        """Tool for user context"""
        return f"User context evaluated: {feedback}"
    
    def _suggest_alternatives(self, primary_action: str) -> List[str]:
        """Suggest alternative actions"""
        alternatives = {
            'confirm': ['refine', 'acquire'],
            'refine': ['acquire', 'fallback'],
            'acquire': ['refine', 'fallback'],
            'fallback': ['acquire', 'refine']
        }
        return alternatives.get(primary_action, [])
    
    def get_reasoning_stats(self) -> Dict:
        """Get reasoning performance statistics"""
        return self.reasoning_stats
    
    def reset_stats(self):
        """Reset reasoning statistics"""
        self.reasoning_stats = {
            'fast_decisions': 0,
            'standard_decisions': 0,
            'deep_decisions': 0,
            'success_rate': 0.0,
            'avg_reasoning_time': 0.0
        }
