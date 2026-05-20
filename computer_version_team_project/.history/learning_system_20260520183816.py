"""
Learning System: AI Training Module for VisionRAG
Tracks performance, learns patterns, and improves decision-making over time
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import numpy as np
from pathlib import Path
from enum import Enum


class ActionType(Enum):
    """Types of actions taken"""
    CONFIRM = "confirm"
    REFINE = "refine"
    ACQUIRE = "acquire"
    FALLBACK = "fallback"


class OutcomeType(Enum):
    """Outcome types for learning"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    SKIP = "skip"


@dataclass
class DetectionRecord:
    """Single detection event with outcome"""
    timestamp: str
    action_taken: str
    target_object: str
    hand_detected: bool
    target_detected: bool
    confidence_score: float
    depth_aligned: bool
    outcome: str
    user_satisfaction: float = 0.5  # 0.0-1.0
    reasoning_level: str = "standard"
    time_to_completion: float = 0.0  # seconds
    fallback_used: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LearningPattern:
    """Learned pattern for decision optimization"""
    pattern_id: str
    trigger_conditions: Dict
    recommended_action: str
    success_rate: float
    pattern_count: int
    last_updated: str
    confidence: float


class AILearningSystem:
    """
    Core learning module for VisionRAG AI training
    Implements reinforcement learning patterns and behavioral adaptation
    """
    
    def __init__(self, max_history: int = 10000, persistence_path: str = None):
        self.max_history = max_history
        self.persistence_path = Path(persistence_path) if persistence_path else Path("learning_data")
        self.persistence_path.mkdir(exist_ok=True)
        
        # Core learning data structures
        self.detection_history: deque = deque(maxlen=max_history)
        self.learned_patterns: Dict[str, LearningPattern] = {}
        self.object_profiles: Dict[str, Dict] = defaultdict(lambda: {
            'detection_count': 0,
            'success_count': 0,
            'avg_confidence': 0.0,
            'preferred_action': ActionType.CONFIRM,
            'fallback_frequency': 0.0
        })
        
        # Performance metrics
        self.performance_metrics = {
            'total_actions': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'avg_success_rate': 0.0,
            'avg_confidence': 0.0,
            'learning_efficiency': 0.0  # Improvement over time
        }
        
        # Learning rate and parameters
        self.learning_rate = 0.1
        self.pattern_threshold = 0.7  # Minimum success rate to extract pattern
        self.min_pattern_occurrences = 5
        
        self._load_persistent_data()
    
    def record_interaction(self, record: DetectionRecord) -> None:
        """
        Record a single interaction for learning
        
        Args:
            record: DetectionRecord with all interaction details
        """
        self.detection_history.append(record)
        
        # Update object profile
        self._update_object_profile(record)
        
        # Update performance metrics
        self._update_metrics(record)
        
        # Potentially extract new patterns
        if len(self.detection_history) % 20 == 0:
            self._extract_patterns()
        
        # Auto-save periodically
        if len(self.detection_history) % 100 == 0:
            self.save_learning_data()
    
    def _update_object_profile(self, record: DetectionRecord) -> None:
        """Update learning profile for specific object"""
        profile = self.object_profiles[record.target_object]
        
        profile['detection_count'] += 1
        
        if record.outcome == OutcomeType.SUCCESS.value:
            profile['success_count'] += 1
        
        # Update average confidence with exponential moving average
        alpha = self.learning_rate
        profile['avg_confidence'] = (
            alpha * record.confidence_score + 
            (1 - alpha) * profile['avg_confidence']
        )
        
        # Update fallback frequency
        if record.fallback_used:
            profile['fallback_frequency'] = (
                alpha * 1.0 + 
                (1 - alpha) * profile['fallback_frequency']
            )
        
        # Recommend action based on success
        if profile['success_count'] > profile['detection_count'] * 0.7:
            profile['preferred_action'] = record.action_taken
    
    def _update_metrics(self, record: DetectionRecord) -> None:
        """Update global performance metrics"""
        self.performance_metrics['total_actions'] += 1
        
        if record.outcome == OutcomeType.SUCCESS.value:
            self.performance_metrics['successful_actions'] += 1
        elif record.outcome == OutcomeType.FAILURE.value:
            self.performance_metrics['failed_actions'] += 1
        
        # Exponential moving average for success rate
        total = self.performance_metrics['total_actions']
        current_rate = self.performance_metrics['successful_actions'] / max(total, 1)
        
        alpha = self.learning_rate
        self.performance_metrics['avg_success_rate'] = (
            alpha * current_rate +
            (1 - alpha) * self.performance_metrics['avg_success_rate']
        )
        
        # Update average confidence
        self.performance_metrics['avg_confidence'] = (
            alpha * record.confidence_score +
            (1 - alpha) * self.performance_metrics['avg_confidence']
        )
    
    def _extract_patterns(self) -> None:
        """Extract learnable patterns from recent history"""
        
        if len(self.detection_history) < self.min_pattern_occurrences:
            return
        
        # Look for recurring successful patterns
        recent_history = list(self.detection_history)[-100:]
        
        # Group by action type
        action_groups = defaultdict(list)
        for record in recent_history:
            action_groups[record.action_taken].append(record)
        
        # Analyze each action type
        for action, records in action_groups.items():
            success_count = sum(1 for r in records if r.outcome == OutcomeType.SUCCESS.value)
            success_rate = success_count / len(records) if records else 0
            
            if success_rate >= self.pattern_threshold and len(records) >= self.min_pattern_occurrences:
                pattern = self._create_pattern(action, records, success_rate)
                self.learned_patterns[pattern.pattern_id] = pattern
    
    def _create_pattern(self, action: str, records: List[DetectionRecord], 
                       success_rate: float) -> LearningPattern:
        """Create a learnable pattern from successful records"""
        
        # Extract common trigger conditions
        conditions = {
            'avg_confidence': np.mean([r.confidence_score for r in records]),
            'hand_detection_rate': sum(1 for r in records if r.hand_detected) / len(records),
            'target_detection_rate': sum(1 for r in records if r.target_detected) / len(records),
            'depth_alignment_rate': sum(1 for r in records if r.depth_aligned) / len(records),
            'typical_target': max(set([r.target_object for r in records]), 
                                 key=[r.target_object for r in records].count)
        }
        
        pattern_id = f"pattern_{len(self.learned_patterns)}_{datetime.now().timestamp()}"
        
        return LearningPattern(
            pattern_id=pattern_id,
            trigger_conditions=conditions,
            recommended_action=action,
            success_rate=success_rate,
            pattern_count=len(records),
            last_updated=datetime.now().isoformat(),
            confidence=min(success_rate, 0.95)
        )
    
    def get_recommendation(self, context: Dict) -> Tuple[str, float]:
        """
        Get action recommendation based on learned patterns
        
        Args:
            context: Current detection context
            
        Returns:
            Tuple of (recommended_action, confidence)
        """
        
        best_match = None
        best_score = 0
        
        for pattern in self.learned_patterns.values():
            match_score = self._pattern_match_score(context, pattern.trigger_conditions)
            
            if match_score > best_score:
                best_score = match_score
                best_match = pattern
        
        if best_match and best_score > 0.7:
            return best_match.recommended_action, best_match.confidence
        
        # Fallback to object-specific recommendation
        target = context.get('target_object', 'unknown')
        if target in self.object_profiles:
            profile = self.object_profiles[target]
            if profile['detection_count'] > 5:
                return profile['preferred_action'], profile['avg_confidence']
        
        # Default recommendation
        return ActionType.CONFIRM.value, 0.5
    
    def _pattern_match_score(self, context: Dict, conditions: Dict) -> float:
        """Calculate how well context matches pattern conditions"""
        
        scores = []
        
        # Check each condition
        if 'avg_confidence' in conditions:
            conf_diff = abs(context.get('confidence_score', 0) - conditions['avg_confidence'])
            scores.append(max(0, 1 - conf_diff))
        
        if 'hand_detection_rate' in conditions:
            hand_match = float(context.get('hand_detected', False)) == conditions['hand_detection_rate']
            scores.append(0.8 if hand_match else 0.3)
        
        if 'target_detection_rate' in conditions:
            target_match = float(context.get('target_detected', False)) == conditions['target_detection_rate']
            scores.append(0.8 if target_match else 0.3)
        
        return np.mean(scores) if scores else 0
    
    def get_learning_progress(self) -> Dict:
        """Get overall learning progress metrics"""
        
        total_objects = len(self.object_profiles)
        proficient_objects = sum(
            1 for p in self.object_profiles.values() 
            if p['success_count'] / max(p['detection_count'], 1) > 0.8
        )
        
        total_patterns = len(self.learned_patterns)
        strong_patterns = sum(
            1 for p in self.learned_patterns.values()
            if p.success_rate > 0.85
        )
        
        return {
            'total_interactions': len(self.detection_history),
            'success_rate': self.performance_metrics['avg_success_rate'],
            'avg_confidence': self.performance_metrics['avg_confidence'],
            'objects_mastered': {
                'total': total_objects,
                'proficient': proficient_objects
            },
            'patterns_learned': {
                'total': total_patterns,
                'strong': strong_patterns
            },
            'learning_efficiency': self._calculate_learning_efficiency(),
            'estimated_improvement': self._estimate_improvement()
        }
    
    def _calculate_learning_efficiency(self) -> float:
        """Calculate learning efficiency (improvement per interaction)"""
        if len(self.detection_history) < 20:
            return 0.0
        
        recent = list(self.detection_history)[-20:]
        older = list(self.detection_history)[-40:-20]
        
        recent_success = sum(1 for r in recent if r.outcome == OutcomeType.SUCCESS.value) / len(recent)
        older_success = sum(1 for r in older if r.outcome == OutcomeType.SUCCESS.value) / len(older)
        
        return (recent_success - older_success) / 20
    
    def _estimate_improvement(self) -> float:
        """Estimate potential improvement by applying learned patterns"""
        
        if not self.learned_patterns:
            return 0.0
        
        avg_pattern_confidence = np.mean([p.confidence for p in self.learned_patterns.values()])
        current_success = self.performance_metrics['avg_success_rate']
        
        potential_improvement = (avg_pattern_confidence - current_success) * 0.5
        return max(0, min(potential_improvement, 0.3))
    
    def save_learning_data(self) -> None:
        """Persist learning data to disk"""
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': self.performance_metrics,
            'patterns': {
                pid: asdict(p) for pid, p in self.learned_patterns.items()
            },
            'object_profiles': dict(self.object_profiles),
            'history_sample': [r.to_dict() for r in list(self.detection_history)[-100:]]
        }
        
        save_path = self.persistence_path / 'learning_state.json'
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_persistent_data(self) -> None:
        """Load previously learned data"""
        
        load_path = self.persistence_path / 'learning_state.json'
        if not load_path.exists():
            return
        
        try:
            with open(load_path, 'r') as f:
                data = json.load(f)
            
            # Restore metrics
            self.performance_metrics = data.get('performance_metrics', self.performance_metrics)
            
            # Restore patterns
            for pid, pattern_data in data.get('patterns', {}).items():
                self.learned_patterns[pid] = LearningPattern(**pattern_data)
            
            # Restore object profiles
            self.object_profiles = defaultdict(
                lambda: {
                    'detection_count': 0,
                    'success_count': 0,
                    'avg_confidence': 0.0,
                    'preferred_action': ActionType.CONFIRM.value,
                    'fallback_frequency': 0.0
                },
                {k: v for k, v in data.get('object_profiles', {}).items()}
            )
        except Exception as e:
            print(f"Warning: Could not load learning data: {e}")
    
    def export_training_report(self) -> str:
        """Generate a learning report"""
        
        progress = self.get_learning_progress()
        
        report = f"""
=== VisionRAG AI Learning Report ===
Generated: {datetime.now().isoformat()}

Performance Summary:
- Total Interactions: {progress['total_interactions']}
- Success Rate: {progress['success_rate']:.1%}
- Average Confidence: {progress['avg_confidence']:.2f}

Object Mastery:
- Objects Encountered: {progress['objects_mastered']['total']}
- Objects Proficient (>80%): {progress['objects_mastered']['proficient']}

Pattern Learning:
- Patterns Discovered: {progress['patterns_learned']['total']}
- Strong Patterns (>85%): {progress['patterns_learned']['strong']}

Learning Efficiency:
- Improvement Rate: {progress['learning_efficiency']:.4f} per interaction
- Estimated Further Improvement: {progress['estimated_improvement']:.1%}
"""
        return report
