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

from .models import (
    Task,
    Trial,
    Transcript,
    Outcome,
    ToolCall,
    Step,
    TaskDifficulty,
    StepType,
)
from .graders import (
    Grader,
    CodeGrader,
    ModelGrader,
    HumanGrader,
    GraderResult,
    GraderType,
    outcome_success_grader,
    tool_call_count_grader,
    final_state_grader,
)
from .scoring import (
    ScoringStrategy,
    WeightedScoring,
    BinaryScoring,
    HybridScoring,
    ScoringMode,
    TaskScore,
)
from .harness import (
    EvaluationHarness,
    EvaluationResult,
    TaskResult,
    TrialResult,
)

__version__ = "0.1.0"

__all__ = [
    # Models
    "Task",
    "Trial",
    "Transcript",
    "Outcome",
    "ToolCall",
    "Step",
    "TaskDifficulty",
    "StepType",
    # Graders
    "Grader",
    "CodeGrader",
    "ModelGrader",
    "HumanGrader",
    "GraderResult",
    "GraderType",
    "outcome_success_grader",
    "tool_call_count_grader",
    "final_state_grader",
    # Scoring
    "ScoringStrategy",
    "WeightedScoring",
    "BinaryScoring",
    "HybridScoring",
    "ScoringMode",
    "TaskScore",
    # Harness
    "EvaluationHarness",
    "EvaluationResult",
    "TaskResult",
    "TrialResult",
]
