"""
Reward System: Feedback and Reinforcement Learning Module
Enables AI to learn from rewards and feedback to optimize decision-making
"""

from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
from collections import defaultdict


class FeedbackType(Enum):
    """Types of feedback"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    PARTIAL = "partial"


class RewardSource(Enum):
    """Sources of rewards"""
    AUTOMATIC = "automatic"  # Detection success metrics
    USER = "user"  # Direct user feedback
    SYSTEM = "system"  # System performance metrics
    INFERENCE = "inference"  # Inferred from outcomes


@dataclass
class RewardEvent:
    """Single reward event"""
    timestamp: str
    action: str
    reward_value: float  # -1.0 to +1.0
    source: RewardSource
    feedback_type: FeedbackType
    reason: str = ""
    related_object: str = ""
    action_taken_ms: float = 0.0
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class RewardPolicy:
    """Policy defining reward rules"""
    name: str
    rules: Dict  # Condition -> reward
    priority: int
    enabled: bool = True
    description: str = ""


class RewardSystem:
    """
    Multi-source reward system for reinforcement learning
    Tracks what actions lead to desired outcomes
    """
    
    def __init__(self):
        self.reward_history: List[RewardEvent] = []
        self.action_rewards: Dict[str, List[float]] = defaultdict(list)
        self.reward_policies: Dict[str, RewardPolicy] = {}
        self.user_satisfaction_scores: Dict[str, float] = defaultdict(float)
        self.action_value_estimates: Dict[str, float] = {}
        
        # Reward aggregation parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_bonus = 0.05
        
        # Initialize default policies
        self._setup_default_policies()
    
    def _setup_default_policies(self) -> None:
        """Setup default reward policies"""
        
        # Policy 1: Detection success rewards
        detection_policy = RewardPolicy(
            name="detection_success",
            rules={
                "high_confidence_success": 0.9,
                "medium_confidence_success": 0.6,
                "low_confidence_success": 0.3,
                "high_confidence_failure": -0.5,
                "timeout": -0.3
            },
            priority=1,
            description="Rewards successful detection with confidence scaling"
        )
        self.reward_policies["detection_success"] = detection_policy
        
        # Policy 2: Speed rewards
        speed_policy = RewardPolicy(
            name="action_speed",
            rules={
                "fast_action": 0.2,
                "normal_action": 0.1,
                "slow_action": -0.1
            },
            priority=2,
            description="Rewards faster actions (but not at cost of accuracy)"
        )
        self.reward_policies["action_speed"] = speed_policy
        
        # Policy 3: Consistency rewards
        consistency_policy = RewardPolicy(
            name="consistency",
            rules={
                "consistent_high_confidence": 0.3,
                "inconsistent_detections": -0.2,
                "improving_confidence": 0.25
            },
            priority=3,
            description="Rewards consistent and improving performance"
        )
        self.reward_policies["consistency"] = consistency_policy
    
    def record_reward(self, action: str, reward_value: float, 
                     source: RewardSource, feedback_type: FeedbackType,
                     reason: str = "", **metadata) -> RewardEvent:
        """
        Record a reward event
        
        Args:
            action: Action that was taken
            reward_value: Reward value (-1.0 to 1.0)
            source: Source of reward
            feedback_type: Type of feedback
            reason: Explanation for reward
            **metadata: Additional context
            
        Returns:
            RewardEvent that was recorded
        """
        
        # Clamp reward value
        reward_value = max(-1.0, min(1.0, reward_value))
        
        event = RewardEvent(
            timestamp=datetime.now().isoformat(),
            action=action,
            reward_value=reward_value,
            source=source,
            feedback_type=feedback_type,
            reason=reason,
            related_object=metadata.get('target_object', ''),
            action_taken_ms=metadata.get('latency_ms', 0),
            confidence_before=metadata.get('confidence_before', 0),
            confidence_after=metadata.get('confidence_after', 0),
            metadata=metadata
        )
        
        self.reward_history.append(event)
        
        # Update action reward estimates
        self.action_rewards[action].append(reward_value)
        self._update_action_value(action, reward_value)
        
        # Update satisfaction if applicable
        if source == RewardSource.USER:
            obj = metadata.get('target_object', 'general')
            self._update_satisfaction(obj, reward_value)
        
        return event
    
    def automatic_reward(self, action: str, 
                        success: bool, 
                        confidence: float,
                        latency_ms: float,
                        **context) -> RewardEvent:
        """
        Automatically assign reward based on system metrics
        
        Args:
            action: Action taken
            success: Whether action succeeded
            confidence: Confidence score of detection
            latency_ms: Time taken for action
            **context: Additional context
            
        Returns:
            RewardEvent recorded
        """
        
        # Calculate reward based on success and confidence
        if success:
            base_reward = confidence * 0.9
            if confidence > 0.8:
                base_reward += 0.1
        else:
            base_reward = -0.7 if confidence > 0.7 else -0.3
        
        # Adjust for speed
        if latency_ms < 50:
            base_reward += 0.1
        elif latency_ms > 200:
            base_reward -= 0.1
        
        return self.record_reward(
            action=action,
            reward_value=base_reward,
            source=RewardSource.AUTOMATIC,
            feedback_type=FeedbackType.POSITIVE if success else FeedbackType.NEGATIVE,
            reason=f"Success={success}, Confidence={confidence:.2f}",
            success=success,
            confidence=confidence,
            latency_ms=latency_ms,
            **context
        )
    
    def user_feedback(self, action: str, satisfaction: float,
                     comment: str = "", **context) -> RewardEvent:
        """
        Record user feedback
        
        Args:
            action: Action that user is evaluating
            satisfaction: User satisfaction score (-1.0 to 1.0)
            comment: User comment
            **context: Additional context
            
        Returns:
            RewardEvent recorded
        """
        
        satisfaction = max(-1.0, min(1.0, satisfaction))
        
        if satisfaction > 0.3:
            feedback_type = FeedbackType.POSITIVE
        elif satisfaction < -0.3:
            feedback_type = FeedbackType.NEGATIVE
        else:
            feedback_type = FeedbackType.NEUTRAL
        
        return self.record_reward(
            action=action,
            reward_value=satisfaction,
            source=RewardSource.USER,
            feedback_type=feedback_type,
            reason=f"User feedback: {comment}",
            satisfaction=satisfaction,
            user_comment=comment,
            **context
        )
    
    def _update_action_value(self, action: str, reward: float) -> None:
        """Update estimated value of action using Q-learning"""
        
        if action not in self.action_value_estimates:
            self.action_value_estimates[action] = 0.0
        
        # Q-learning update
        current_estimate = self.action_value_estimates[action]
        alpha = self.learning_rate
        
        self.action_value_estimates[action] = (
            current_estimate + alpha * (reward - current_estimate)
        )
    
    def _update_satisfaction(self, object_name: str, reward: float) -> None:
        """Update user satisfaction for object"""
        
        alpha = self.learning_rate
        self.user_satisfaction_scores[object_name] = (
            alpha * reward + (1 - alpha) * self.user_satisfaction_scores[object_name]
        )
    
    def get_best_actions(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k actions by estimated value"""
        
        if not self.action_value_estimates:
            return []
        
        sorted_actions = sorted(
            self.action_value_estimates.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_actions[:top_k]
    
    def get_worst_actions(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get worst-k actions by estimated value"""
        
        if not self.action_value_estimates:
            return []
        
        sorted_actions = sorted(
            self.action_value_estimates.items(),
            key=lambda x: x[1]
        )
        
        return sorted_actions[:top_k]
    
    def get_action_value(self, action: str, with_exploration: bool = False) -> float:
        """
        Get estimated value of action
        
        Args:
            action: Action to evaluate
            with_exploration: Include exploration bonus
            
        Returns:
            Estimated value
        """
        
        base_value = self.action_value_estimates.get(action, 0.0)
        
        if with_exploration:
            # Count how many times this action has been tried
            count = len(self.action_rewards.get(action, []))
            exploration_bonus = self.exploration_bonus / np.sqrt(max(count, 1))
            return base_value + exploration_bonus
        
        return base_value
    
    def get_satisfaction_metrics(self) -> Dict:
        """Get satisfaction metrics across objects"""
        
        if not self.user_satisfaction_scores:
            return {'average': 0.0, 'by_object': {}}
        
        scores = dict(self.user_satisfaction_scores)
        avg = np.mean(list(scores.values()))
        
        return {
            'average': avg,
            'by_object': scores,
            'satisfied_objects': sum(1 for v in scores.values() if v > 0.3),
            'dissatisfied_objects': sum(1 for v in scores.values() if v < -0.3)
        }
    
    def get_improvement_trends(self, window: int = 50) -> Dict:
        """Analyze improvement trends over time"""
        
        if len(self.reward_history) < window:
            recent_rewards = [e.reward_value for e in self.reward_history]
        else:
            recent_rewards = [e.reward_value for e in self.reward_history[-window:]]
        
        if len(recent_rewards) < 2:
            return {'trend': 'insufficient_data', 'slope': 0, 'recent_avg': 0}
        
        # Linear regression to find trend
        x = np.arange(len(recent_rewards))
        z = np.polyfit(x, recent_rewards, 1)
        slope = z[0]
        
        # Calculate averages
        first_half_avg = np.mean(recent_rewards[:len(recent_rewards)//2])
        second_half_avg = np.mean(recent_rewards[len(recent_rewards)//2:])
        
        if slope > 0.01:
            trend = 'improving'
        elif slope < -0.01:
            trend = 'degrading'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': float(slope),
            'first_half_avg': float(first_half_avg),
            'second_half_avg': float(second_half_avg),
            'recent_avg': float(np.mean(recent_rewards[-10:]))
        }
    
    def generate_improvement_suggestions(self) -> List[str]:
        """Generate suggestions for improvement"""
        
        suggestions = []
        
        # Analyze worst actions
        worst = self.get_worst_actions(3)
        if worst and worst[0][1] < -0.3:
            suggestions.append(f"Reduce use of '{worst[0][0]}' - it has low value")
        
        # Check dissatisfaction
        sat_metrics = self.get_satisfaction_metrics()
        if sat_metrics['dissatisfied_objects']:
            objects = [k for k, v in sat_metrics['by_object'].items() if v < -0.3]
            suggestions.append(f"Improve handling of: {', '.join(objects)}")
        
        # Check trends
        trends = self.get_improvement_trends()
        if trends['trend'] == 'degrading':
            suggestions.append("Performance is degrading - review recent model changes")
        
        # Exploration bonus
        if len(self.action_rewards) < 3:
            suggestions.append("Explore more action types to find optimal strategy")
        
        return suggestions
    
    def export_reward_report(self) -> str:
        """Generate reward system report"""
        
        report = f"""
=== Reward System Report ===
Generated: {datetime.now().isoformat()}

Reward Statistics:
- Total Rewards Recorded: {len(self.reward_history)}
- Actions Tracked: {len(self.action_rewards)}
- Average Reward: {np.mean([e.reward_value for e in self.reward_history]):.3f}

Action Values (Top 5):
"""
        for action, value in self.get_best_actions(5):
            report += f"  {action}: {value:.3f}\n"
        
        report += "\nUser Satisfaction:\n"
        sat = self.get_satisfaction_metrics()
        for obj, score in list(sat['by_object'].items())[:5]:
            report += f"  {obj}: {score:.3f}\n"
        
        report += "\nPerformance Trend:\n"
        trends = self.get_improvement_trends()
        report += f"  Trend: {trends['trend']}\n"
        report += f"  Recent Average: {trends['recent_avg']:.3f}\n"
        
        report += "\nImprovement Suggestions:\n"
        for suggestion in self.generate_improvement_suggestions():
            report += f"  - {suggestion}\n"
        
        return report
