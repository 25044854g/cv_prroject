"""
Model Selector: Intelligent Model Selection based on Context and Performance
Dynamically chooses between different detection models and inference strategies
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta


class ModelName(Enum):
    """Available models"""
    YOLO_FAST = "yolov8n"  # Nano: fast, low accuracy
    YOLO_BALANCED = "yolov8m"  # Medium: balanced
    YOLO_ACCURATE = "yolov8l"  # Large: accurate, slow
    MEDIAPIPE_FAST = "mediapipe_fast"
    MEDIAPIPE_FULL = "mediapipe_full"
    YOLO_POSE = "yolov8n-pose"


class ScenarioType(Enum):
    """Detection scenarios"""
    FAST_RESPONSE = "fast_response"  # Real-time, latency critical
    HIGH_ACCURACY = "high_accuracy"  # Accuracy critical
    BALANCED = "balanced"
    LOW_RESOURCE = "low_resource"  # CPU/memory limited
    BATCH_PROCESSING = "batch"


@dataclass
class ModelProfile:
    """Profile of model characteristics"""
    name: ModelName
    accuracy: float  # mAP or similar 0-1
    latency_ms: float
    memory_mb: float
    throughput_fps: float
    cost_per_1k_requests: float = 0.0  # For API models
    reliability: float = 0.95  # Success rate
    best_for: List[str] = None  # ["small_objects", "occluded", etc]
    worst_for: List[str] = None
    
    def cost_score(self) -> float:
        """Calculate cost-effectiveness score"""
        return self.accuracy / (self.latency_ms * self.memory_mb / 10000)


@dataclass
class SelectionContext:
    """Context for model selection"""
    scenario: ScenarioType
    target_object: str
    frame_resolution: Tuple[int, int]
    available_resources: Dict  # CPU%, Memory%, GPU available
    latency_budget_ms: float
    accuracy_requirement: float  # 0-1
    user_priority: str = "balanced"  # "speed" | "accuracy" | "cost"
    historical_success_rate: float = 0.0
    ambient_conditions: str = "normal"  # "low_light", "occluded", etc


class ModelSelector:
    """
    Intelligent model selection system for VisionRAG
    Chooses optimal detection models based on context and constraints
    """
    
    def __init__(self):
        self.models = self._initialize_models()
        self.selection_history: List[Dict] = []
        self.selection_performance: Dict = {}
        self.user_preferences: Dict = {}
    
    def _initialize_models(self) -> Dict[ModelName, ModelProfile]:
        """Initialize all available model profiles"""
        
        return {
            ModelName.YOLO_FAST: ModelProfile(
                name=ModelName.YOLO_FAST,
                accuracy=0.75,
                latency_ms=15,
                memory_mb=50,
                throughput_fps=60,
                reliability=0.92,
                best_for=["real-time", "small_objects"],
                worst_for=["occlusion", "low_light"]
            ),
            ModelName.YOLO_BALANCED: ModelProfile(
                name=ModelName.YOLO_BALANCED,
                accuracy=0.82,
                latency_ms=30,
                memory_mb=150,
                throughput_fps=30,
                reliability=0.95,
                best_for=["general", "balanced"],
                worst_for=["very_small_objects"]
            ),
            ModelName.YOLO_ACCURATE: ModelProfile(
                name=ModelName.YOLO_ACCURATE,
                accuracy=0.88,
                latency_ms=60,
                memory_mb=400,
                throughput_fps=15,
                reliability=0.96,
                best_for=["high-accuracy", "complex_scenes"],
                worst_for=["real-time", "low_resource"]
            ),
            ModelName.MEDIAPIPE_FAST: ModelProfile(
                name=ModelName.MEDIAPIPE_FAST,
                accuracy=0.80,
                latency_ms=10,
                memory_mb=30,
                throughput_fps=100,
                reliability=0.88,
                best_for=["hands", "fast"],
                worst_for=["small_hands"]
            ),
            ModelName.MEDIAPIPE_FULL: ModelProfile(
                name=ModelName.MEDIAPIPE_FULL,
                accuracy=0.90,
                latency_ms=25,
                memory_mb=80,
                throughput_fps=40,
                reliability=0.93,
                best_for=["hands", "accurate_hand_tracking"],
                worst_for=["very_crowded"]
            ),
            ModelName.YOLO_POSE: ModelProfile(
                name=ModelName.YOLO_POSE,
                accuracy=0.85,
                latency_ms=35,
                memory_mb=200,
                throughput_fps=25,
                reliability=0.94,
                best_for=["hand_pose", "fallback"],
                worst_for=["small_subjects"]
            ),
        }
    
    def select_model(self, context: SelectionContext) -> Tuple[ModelName, float]:
        """
        Main model selection function
        
        Args:
            context: Selection context with constraints
            
        Returns:
            Tuple of (selected_model, confidence_score)
        """
        
        # Get candidate models
        candidates = self._get_candidate_models(context)
        
        if not candidates:
            # Fallback to balanced model
            return ModelName.YOLO_BALANCED, 0.5
        
        # Score each candidate
        scores = {}
        for model_name in candidates:
            score = self._score_model(model_name, context)
            scores[model_name] = score
        
        # Select best
        selected = max(scores.items(), key=lambda x: x[1])
        
        # Record selection
        self._record_selection(selected[0], context, selected[1])
        
        return selected
    
    def _get_candidate_models(self, context: SelectionContext) -> List[ModelName]:
        """Filter models based on hard constraints"""
        
        candidates = []
        available_memory = context.available_resources.get('memory_mb', 2000)
        available_cpu = context.available_resources.get('cpu_percent', 100)
        has_gpu = context.available_resources.get('gpu_available', False)
        
        for model_name, profile in self.models.items():
            # Hard constraints
            if profile.memory_mb > available_memory * 0.7:
                continue  # Skip if would use >70% of available memory
            
            if profile.latency_ms > context.latency_budget_ms:
                continue  # Skip if too slow
            
            if profile.accuracy < context.accuracy_requirement:
                continue  # Skip if not accurate enough
            
            candidates.append(model_name)
        
        return candidates
    
    def _score_model(self, model_name: ModelName, context: SelectionContext) -> float:
        """Score a model for the given context"""
        
        profile = self.models[model_name]
        
        # Base score from accuracy
        accuracy_score = profile.accuracy * 100
        
        # Penalty for latency
        latency_penalty = (profile.latency_ms / context.latency_budget_ms) * 20
        
        # Bonus for resource efficiency
        resource_efficiency = (context.available_resources.get('memory_mb', 2000) / 
                             profile.memory_mb) * 5
        
        # Bonus for scenario fit
        scenario_bonus = 0
        if context.scenario == ScenarioType.FAST_RESPONSE:
            scenario_bonus = 30 * (1 / (profile.latency_ms + 1))
        elif context.scenario == ScenarioType.HIGH_ACCURACY:
            scenario_bonus = 20 * profile.accuracy
        elif context.scenario == ScenarioType.LOW_RESOURCE:
            scenario_bonus = 15 * (500 / (profile.memory_mb + 100))
        elif context.scenario == ScenarioType.BALANCED:
            scenario_bonus = 10
        
        # User priority adjustment
        if context.user_priority == "speed":
            latency_penalty *= 0.5
        elif context.user_priority == "accuracy":
            accuracy_score *= 1.5
        elif context.user_priority == "cost":
            resource_efficiency *= 1.5
        
        # Historical success bonus
        if model_name in self.selection_performance:
            perf = self.selection_performance[model_name]
            success_bonus = perf.get('success_rate', 0.5) * 15
        else:
            success_bonus = 0
        
        # Historical weight for target object
        if context.target_object in self.user_preferences:
            preferred = self.user_preferences[context.target_object]
            if preferred == model_name:
                scenario_bonus *= 1.3
        
        total_score = (accuracy_score - latency_penalty + resource_efficiency + 
                      scenario_bonus + success_bonus)
        
        return max(0, total_score)
    
    def select_ensemble(self, context: SelectionContext, 
                       num_models: int = 2) -> List[Tuple[ModelName, float]]:
        """
        Select ensemble of complementary models
        
        Args:
            context: Selection context
            num_models: Number of models for ensemble
            
        Returns:
            List of (model_name, weight) tuples
        """
        
        candidates = self._get_candidate_models(context)
        
        if len(candidates) < num_models:
            # Return what we can
            return [(m, 1.0/len(candidates)) for m in candidates]
        
        # Score all candidates
        scores = {}
        for model_name in candidates:
            scores[model_name] = self._score_model(model_name, context)
        
        # Select top models that are diverse
        selected = []
        selected_accuracy = []
        selected_latency = []
        
        for _ in range(num_models):
            best = None
            best_score = -1
            best_diversity = 0
            
            for model_name, score in scores.items():
                if model_name in [m[0] for m in selected]:
                    continue
                
                # Calculate diversity from already selected models
                if selected:
                    profile = self.models[model_name]
                    acc_std = np.std([self.models[m].accuracy for m in [model_name]] + selected_accuracy)
                    lat_std = np.std([self.models[m].latency_ms for m in [model_name]] + selected_latency)
                    diversity = (acc_std + lat_std) * 0.1
                else:
                    diversity = 0
                
                total = score + diversity
                
                if total > best_score:
                    best_score = total
                    best = model_name
            
            if best:
                selected.append((best, 1.0))
                selected_accuracy.append(self.models[best].accuracy)
                selected_latency.append(self.models[best].latency_ms)
        
        # Normalize weights
        total_weight = sum(w for _, w in selected)
        return [(m, w/total_weight) for m, w in selected]
    
    def _record_selection(self, model_name: ModelName, context: SelectionContext, 
                         score: float) -> None:
        """Record a model selection"""
        
        self.selection_history.append({
            'timestamp': datetime.now().isoformat(),
            'model': model_name.value,
            'scenario': context.scenario.value,
            'score': score,
            'target_object': context.target_object
        })
        
        # Keep only last 1000 selections
        if len(self.selection_history) > 1000:
            self.selection_history = self.selection_history[-1000:]
    
    def record_performance(self, model_name: ModelName, success: bool, 
                          latency_actual: float, accuracy_achieved: float) -> None:
        """Record performance of a model selection"""
        
        if model_name not in self.selection_performance:
            self.selection_performance[model_name] = {
                'total_uses': 0,
                'success_count': 0,
                'success_rate': 0.5,
                'avg_latency': 0,
                'avg_accuracy': 0
            }
        
        perf = self.selection_performance[model_name]
        
        # Update stats with exponential moving average
        alpha = 0.1
        perf['total_uses'] += 1
        perf['success_count'] += int(success)
        perf['success_rate'] = perf['success_count'] / perf['total_uses']
        perf['avg_latency'] = alpha * latency_actual + (1 - alpha) * perf['avg_latency']
        perf['avg_accuracy'] = alpha * accuracy_achieved + (1 - alpha) * perf['avg_accuracy']
    
    def set_object_preference(self, object_name: str, preferred_model: ModelName) -> None:
        """Set user preference for specific object detection"""
        self.user_preferences[object_name] = preferred_model
    
    def get_selection_stats(self) -> Dict:
        """Get statistics on model selections"""
        
        stats = {}
        for model_name, perf in self.selection_performance.items():
            stats[model_name.value] = perf
        
        return {
            'model_performance': stats,
            'total_selections': len(self.selection_history),
            'most_selected': self._get_most_selected_model() if self.selection_history else None
        }
    
    def _get_most_selected_model(self) -> Optional[str]:
        """Get most frequently selected model"""
        if not self.selection_history:
            return None
        
        models = [s['model'] for s in self.selection_history]
        return max(set(models), key=models.count)
    
    def recommend_model_for_object(self, target_object: str) -> ModelName:
        """Recommend best model for specific object based on history"""
        
        # Look at history for this object
        relevant = [s for s in self.selection_history if s['target_object'] == target_object]
        
        if not relevant:
            return ModelName.YOLO_BALANCED
        
        # Find most successful model
        model_success = {}
        for selection in relevant:
            model = selection['model']
            if model not in model_success:
                model_success[model] = {'count': 0, 'score': 0}
            model_success[model]['count'] += 1
            model_success[model]['score'] += selection['score']
        
        # Calculate average score
        best_model = max(
            model_success.items(),
            key=lambda x: x[1]['score'] / x[1]['count']
        )[0]
        
        return ModelName[best_model.upper().replace("-", "_")]
