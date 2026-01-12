"""
AI Agent Evaluation Framework

Based on Anthropic's approach to evaluating AI agents as described in
"Demystifying evals for AI agents".

This framework provides interfaces and implementations for:
- Task definitions with inputs and success criteria
- Trial execution and transcript recording
- Multiple grader types (code-based, model-based, human)
- Evaluation harness for running tasks and aggregating results
- Flexible scoring strategies (weighted, binary, hybrid)
"""

from .models import Task, Trial, Transcript, Outcome, ToolCall, Step
from .graders import Grader, CodeGrader, ModelGrader, HumanGrader, GraderResult
from .scoring import ScoringStrategy, WeightedScoring, BinaryScoring, HybridScoring
from .harness import EvaluationHarness, EvaluationResult

__version__ = "0.1.0"

__all__ = [
    # Models
    "Task",
    "Trial",
    "Transcript",
    "Outcome",
    "ToolCall",
    "Step",
    # Graders
    "Grader",
    "CodeGrader",
    "ModelGrader",
    "HumanGrader",
    "GraderResult",
    # Scoring
    "ScoringStrategy",
    "WeightedScoring",
    "BinaryScoring",
    "HybridScoring",
    # Harness
    "EvaluationHarness",
    "EvaluationResult",
]
