"""
VisionRAG AI Training System Integration Demo
Demonstrates how adaptive reasoning, learning, model selection, and rewards work together
"""

import sys
import time
import numpy as np
from typing import Dict, List
from datetime import datetime

# Import AI training modules
from adaptive_reasoning import (
    AdaptiveReasoningEngine, 
    ReasoningContext, 
    ReasoningLevel
)
from learning_system import (
    AILearningSystem,
    DetectionRecord,
    ActionType,
    OutcomeType
)
from model_selector import (
    ModelSelector,
    SelectionContext,
    ScenarioType,
    ModelName
)
from reward_system import (
    RewardSystem,
    RewardSource,
    FeedbackType
)


class AITrainingSystemDemo:
    """Comprehensive demo of VisionRAG AI training system"""
    
    def __init__(self):
        self.reasoning_engine = AdaptiveReasoningEngine()
        self.learning_system = AILearningSystem(persistence_path="learning_data")
        self.model_selector = ModelSelector()
        self.reward_system = RewardSystem()
        
        self.demo_results = {
            'reasoning_demos': [],
            'learning_demos': [],
            'selection_demos': [],
            'reward_demos': []
        }
    
    def print_header(self, title: str):
        """Print formatted section header"""
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")
    
    def demo_adaptive_reasoning(self):
        """Demo 1: Adaptive Reasoning with ReAct"""
        self.print_header("DEMO 1: Adaptive Reasoning (ReAct Chains)")
        
        # Scenario 1: High confidence - Fast reasoning
        print("Scenario 1: High Confidence Detection (FAST REASONING)")
        print("-" * 60)
        
        context1 = ReasoningContext(
            hand_detected=True,
            target_detected=True,
            confidence_score=0.92,
            depth_info={'is_same_depth': True, 'depth_diff': 5},
            user_feedback="grab the cell phone"
        )
        
        decision1 = self.reasoning_engine.reason_about_action(context1)
        print(f"Action: {decision1.action}")
        print(f"Confidence: {decision1.confidence:.2f}")
        print(f"Success Rate: {decision1.estimated_success_rate:.1%}")
        print(f"Reasoning Level: {decision1.reasoning_level_used.value}")
        print(f"Steps: {len(decision1.reasoning_steps)}")
        
        # Scenario 2: Low confidence - Deep reasoning
        print("\n\nScenario 2: Low Confidence Detection (DEEP REASONING)")
        print("-" * 60)
        
        context2 = ReasoningContext(
            hand_detected=False,
            target_detected=True,
            confidence_score=0.35,
            depth_info={'is_same_depth': False, 'depth_diff': 80},
            user_feedback=None
        )
        
        decision2 = self.reasoning_engine.reason_about_action(context2)
        print(f"Action: {decision2.action}")
        print(f"Confidence: {decision2.confidence:.2f}")
        print(f"Success Rate: {decision2.estimated_success_rate:.1%}")
        print(f"Reasoning Level: {decision2.reasoning_level_used.value}")
        print(f"Steps: {len(decision2.reasoning_steps)}")
        for step in decision2.reasoning_steps[:3]:
            print(f"  - {step}")
        
        print(f"\nReasoning Statistics: {self.reasoning_engine.get_reasoning_stats()}")
        
        self.demo_results['reasoning_demos'].append({
            'fast_decision': decision1.action,
            'deep_decision': decision2.action
        })
    
    def demo_learning_system(self):
        """Demo 2: Learning System"""
        self.print_header("DEMO 2: Learning System (AI Training)")
        
        print("Simulating 20 interactions with learning...")
        print("-" * 60)
        
        objects = ['cell phone', 'cup', 'keyboard']
        actions = ['confirm', 'refine', 'acquire']
        
        for i in range(20):
            obj = objects[i % 3]
            action = actions[i % 3]
            
            # Simulate interaction
            success = np.random.random() > 0.3  # 70% success rate
            confidence = np.random.uniform(0.5, 0.95)
            
            record = DetectionRecord(
                timestamp=datetime.now().isoformat(),
                action_taken=action,
                target_object=obj,
                hand_detected=np.random.random() > 0.2,
                target_detected=True,
                confidence_score=confidence,
                depth_aligned=np.random.random() > 0.3,
                outcome=OutcomeType.SUCCESS.value if success else OutcomeType.FAILURE.value,
                user_satisfaction=0.7 if success else 0.2,
                time_to_completion=np.random.uniform(10, 100)
            )
            
            self.learning_system.record_interaction(record)
            
            if (i + 1) % 5 == 0:
                print(f"  Recorded {i+1} interactions...")
        
        # Get learning progress
        progress = self.learning_system.get_learning_progress()
        print(f"\nLearning Progress After 20 Interactions:")
        print(f"  Success Rate: {progress['success_rate']:.1%}")
        print(f"  Objects Encountered: {progress['objects_mastered']['total']}")
        print(f"  Patterns Discovered: {progress['patterns_learned']['total']}")
        print(f"  Learning Efficiency: {progress['learning_efficiency']:.6f}")
        
        # Get recommendation
        test_context = {
            'target_object': 'cell phone',
            'confidence_score': 0.85,
            'hand_detected': True,
            'target_detected': True
        }
        action, confidence = self.learning_system.get_recommendation(test_context)
        print(f"\nRecommendation for 'cell phone': {action} (confidence: {confidence:.2f})")
        
        self.demo_results['learning_demos'].append(progress)
    
    def demo_model_selection(self):
        """Demo 3: Model Selection"""
        self.print_header("DEMO 3: Intelligent Model Selection")
        
        print("Scenario 1: Real-time detection (fast response required)")
        print("-" * 60)
        
        context1 = SelectionContext(
            scenario=ScenarioType.FAST_RESPONSE,
            target_object='cell phone',
            frame_resolution=(1920, 1080),
            available_resources={
                'cpu_percent': 80,
                'memory_mb': 2000,
                'gpu_available': True
            },
            latency_budget_ms=50,
            accuracy_requirement=0.7,
            user_priority='speed'
        )
        
        model1, score1 = self.model_selector.select_model(context1)
        print(f"Selected Model: {model1.value}")
        print(f"Selection Score: {score1:.2f}")
        print(f"Expected Latency: 15ms")
        print(f"Expected Accuracy: 0.75")
        
        print("\n\nScenario 2: High accuracy required (low resource)")
        print("-" * 60)
        
        context2 = SelectionContext(
            scenario=ScenarioType.HIGH_ACCURACY,
            target_object='keyboard',
            frame_resolution=(640, 480),
            available_resources={
                'cpu_percent': 40,
                'memory_mb': 800,
                'gpu_available': False
            },
            latency_budget_ms=200,
            accuracy_requirement=0.85,
            user_priority='accuracy'
        )
        
        model2, score2 = self.model_selector.select_model(context2)
        print(f"Selected Model: {model2.value}")
        print(f"Selection Score: {score2:.2f}")
        print(f"Expected Latency: 30ms")
        print(f"Expected Accuracy: 0.82")
        
        # Ensemble selection
        print("\n\nBonus: Ensemble Selection")
        print("-" * 60)
        
        ensemble = self.model_selector.select_ensemble(context1, num_models=3)
        print("Recommended Ensemble:")
        for model, weight in ensemble:
            print(f"  {model.value}: {weight:.1%}")
        
        self.demo_results['selection_demos'].append({
            'fast_model': model1.value,
            'accurate_model': model2.value
        })
    
    def demo_reward_system(self):
        """Demo 4: Reward System and Reinforcement Learning"""
        self.print_header("DEMO 4: Reward System (Reinforcement Learning)")
        
        print("Simulating action rewards over 15 trials...")
        print("-" * 60)
        
        actions = ['confirm', 'refine', 'acquire', 'fallback']
        
        for trial in range(15):
            action = actions[trial % 4]
            
            # Simulate outcome
            success = np.random.random() > 0.3
            confidence = np.random.uniform(0.4, 0.95)
            latency = np.random.uniform(10, 150)
            
            # Record automatic reward
            self.reward_system.automatic_reward(
                action=action,
                success=success,
                confidence=confidence,
                latency_ms=latency,
                target_object='object',
                trial_number=trial
            )
            
            # Occasionally add user feedback
            if trial % 3 == 0:
                user_rating = 0.7 if success else -0.5
                self.reward_system.user_feedback(
                    action=action,
                    satisfaction=user_rating,
                    comment=f"User trial {trial+1}",
                    target_object='object'
                )
        
        # Analyze results
        print(f"Total Rewards Recorded: {len(self.reward_system.reward_history)}")
        
        print("\nTop Actions by Value:")
        for action, value in self.reward_system.get_best_actions(3):
            print(f"  {action}: {value:.3f}")
        
        print("\nWorst Actions by Value:")
        for action, value in self.reward_system.get_worst_actions(3):
            print(f"  {action}: {value:.3f}")
        
        print("\nUser Satisfaction Metrics:")
        sat = self.reward_system.get_satisfaction_metrics()
        print(f"  Average Satisfaction: {sat['average']:.3f}")
        print(f"  Satisfied Objects: {sat['satisfied_objects']}")
        print(f"  Dissatisfied Objects: {sat['dissatisfied_objects']}")
        
        print("\nPerformance Trend:")
        trends = self.reward_system.get_improvement_trends()
        print(f"  Trend: {trends['trend']}")
        print(f"  Slope: {trends['slope']:.4f}")
        print(f"  Recent Avg: {trends['recent_avg']:.3f}")
        
        print("\nImprovement Suggestions:")
        for suggestion in self.reward_system.generate_improvement_suggestions():
            print(f"  • {suggestion}")
        
        self.demo_results['reward_demos'].append({
            'best_action': self.reward_system.get_best_actions(1)[0] if self.reward_system.action_value_estimates else None,
            'satisfaction': sat['average']
        })
    
    def demo_full_integration(self):
        """Demo 5: Full System Integration"""
        self.print_header("DEMO 5: Full Integration (Complete AI Pipeline)")
        
        print("Simulating complete AI pipeline with all components...")
        print("-" * 60)
        
        # Step 1: Model selection
        context = SelectionContext(
            scenario=ScenarioType.BALANCED,
            target_object='cell phone',
            frame_resolution=(1920, 1080),
            available_resources={
                'cpu_percent': 60,
                'memory_mb': 2000,
                'gpu_available': True
            },
            latency_budget_ms=100,
            accuracy_requirement=0.75
        )
        
        selected_model, _ = self.model_selector.select_model(context)
        print(f"1. Selected Model: {selected_model.value}")
        
        # Step 2: Reasoning
        reasoning_context = ReasoningContext(
            hand_detected=True,
            target_detected=True,
            confidence_score=0.88,
            depth_info={'is_same_depth': True}
        )
        
        decision = self.reasoning_engine.reason_about_action(reasoning_context)
        print(f"2. Reasoning Decision: {decision.action}")
        print(f"   Reasoning Level: {decision.reasoning_level_used.value}")
        
        # Step 3: Learning recommendation
        action, learned_confidence = self.learning_system.get_recommendation({
            'target_object': 'cell phone',
            'confidence_score': 0.88,
            'hand_detected': True,
            'target_detected': True
        })
        print(f"3. Learning Recommendation: {action} (confidence: {learned_confidence:.2f})")
        
        # Step 4: Execute and reward
        success = True
        latency_ms = 45
        
        # Record for learning
        record = DetectionRecord(
            timestamp=datetime.now().isoformat(),
            action_taken=decision.action,
            target_object='cell phone',
            hand_detected=True,
            target_detected=True,
            confidence_score=0.88,
            depth_aligned=True,
            outcome=OutcomeType.SUCCESS.value if success else OutcomeType.FAILURE.value,
            user_satisfaction=0.9 if success else 0.2,
            time_to_completion=latency_ms
        )
        self.learning_system.record_interaction(record)
        
        # Record reward
        self.reward_system.automatic_reward(
            action=decision.action,
            success=success,
            confidence=0.88,
            latency_ms=latency_ms,
            target_object='cell phone'
        )
        
        print(f"4. Execution Result: {'SUCCESS' if success else 'FAILURE'}")
        print(f"   Latency: {latency_ms}ms")
        
        print("\n✓ Full pipeline completed successfully!")
    
    def run_all_demos(self):
        """Run all demonstrations"""
        print("\n")
        print("*" * 70)
        print("*  VisionRAG AI Training System - Complete Demo Suite  *")
        print("*" * 70)
        
        self.demo_adaptive_reasoning()
        self.demo_model_selection()
        self.demo_learning_system()
        self.demo_reward_system()
        self.demo_full_integration()
        
        self.print_header("DEMO SUMMARY")
        print("✓ Adaptive Reasoning: ReAct chains with 3 complexity levels")
        print("✓ Model Selection: Intelligent selection based on context")
        print("✓ Learning System: Extracted patterns and recommendations")
        print("✓ Reward System: Reinforcement learning with multi-source feedback")
        print("✓ Full Integration: End-to-end AI training pipeline")
        print("\nAll demonstrations completed successfully! 🎉")


if __name__ == "__main__":
    demo = AITrainingSystemDemo()
    demo.run_all_demos()
