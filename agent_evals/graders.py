"""
Grader implementations for evaluating agent performance.

Graders score different aspects of an agent's performance by examining
either the transcript (what the agent did) or the outcome (what actually
happened in the environment).

Three types of graders:
- CodeGrader: Deterministic, rule-based evaluation
- ModelGrader: Uses an LLM to evaluate agent behavior
- HumanGrader: Requires human judgment
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List
from enum import Enum

from .models import Transcript, Outcome


class GraderType(Enum):
    """Type of grader."""
    CODE = "code"
    MODEL = "model"
    HUMAN = "human"


@dataclass
class GraderResult:
    """
    Result from a grader evaluation.

    Attributes:
        score: Numeric score (typically 0.0 to 1.0)
        passed: Whether the evaluation passed
        explanation: Explanation of the score
        metadata: Additional metadata about the grading
        confidence: Confidence level in the score (for model-based graders)
    """
    score: float
    passed: bool
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "score": self.score,
            "passed": self.passed,
            "explanation": self.explanation,
            "metadata": self.metadata,
            "confidence": self.confidence,
        }


class Grader(ABC):
    """
    Abstract base class for all graders.

    Graders evaluate agent performance by examining transcripts and outcomes.
    They can look at what the agent did (transcript) or what actually happened
    (outcome) in the environment.
    """

    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        required: bool = False,
        grader_type: GraderType = GraderType.CODE,
    ):
        """
        Initialize a grader.

        Args:
            name: Name of this grader
            weight: Weight for scoring (used in weighted scoring)
            required: Whether this grader must pass for task to pass
            grader_type: Type of grader
        """
        self.name = name
        self.weight = weight
        self.required = required
        self.grader_type = grader_type

    @abstractmethod
    def grade(
        self,
        transcript: Optional[Transcript] = None,
        outcome: Optional[Outcome] = None,
    ) -> GraderResult:
        """
        Grade the agent's performance.

        Args:
            transcript: The transcript to grade (optional)
            outcome: The outcome to grade (optional)

        Returns:
            GraderResult with score and explanation
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', weight={self.weight})"


class CodeGrader(Grader):
    """
    Deterministic, rule-based grader.

    Code graders use deterministic logic to evaluate performance.
    They're fast, reliable, and easy to debug but may not capture
    nuanced behavior.

    Example use cases:
    - Check if a file was created
    - Verify database state
    - Count specific tool calls
    - Validate output format
    """

    def __init__(
        self,
        name: str,
        grading_function: Callable[[Optional[Transcript], Optional[Outcome]], GraderResult],
        weight: float = 1.0,
        required: bool = False,
    ):
        """
        Initialize a code-based grader.

        Args:
            name: Name of this grader
            grading_function: Function that performs the grading
            weight: Weight for scoring
            required: Whether this grader must pass
        """
        super().__init__(name, weight, required, GraderType.CODE)
        self.grading_function = grading_function

    def grade(
        self,
        transcript: Optional[Transcript] = None,
        outcome: Optional[Outcome] = None,
    ) -> GraderResult:
        """Grade using the provided grading function."""
        return self.grading_function(transcript, outcome)


class ModelGrader(Grader):
    """
    LLM-based grader.

    Model graders use a language model to evaluate agent behavior.
    They can capture nuanced aspects of performance but are slower
    and less deterministic than code graders.

    Example use cases:
    - Evaluate output quality
    - Check if reasoning was sound
    - Assess helpfulness or tone
    - Verify complex criteria that are hard to code
    """

    def __init__(
        self,
        name: str,
        prompt_template: str,
        model_client: Optional[Any] = None,
        weight: float = 1.0,
        required: bool = False,
        pass_threshold: float = 0.7,
    ):
        """
        Initialize a model-based grader.

        Args:
            name: Name of this grader
            prompt_template: Template for the grading prompt
            model_client: Client for calling the LLM (e.g., Anthropic client)
            weight: Weight for scoring
            required: Whether this grader must pass
            pass_threshold: Threshold score for passing
        """
        super().__init__(name, weight, required, GraderType.MODEL)
        self.prompt_template = prompt_template
        self.model_client = model_client
        self.pass_threshold = pass_threshold

    def grade(
        self,
        transcript: Optional[Transcript] = None,
        outcome: Optional[Outcome] = None,
    ) -> GraderResult:
        """
        Grade using an LLM.

        This is a placeholder implementation. In practice, you would:
        1. Format the prompt with transcript/outcome data
        2. Call the LLM with the prompt
        3. Parse the LLM's response to extract score and explanation
        """
        if self.model_client is None:
            return GraderResult(
                score=0.0,
                passed=False,
                explanation="No model client provided",
            )

        # Format the prompt
        context = self._format_context(transcript, outcome)
        prompt = self.prompt_template.format(context=context)

        # In a real implementation, you would call the LLM here
        # For now, return a placeholder result
        return GraderResult(
            score=0.0,
            passed=False,
            explanation="Model grading not yet implemented - requires LLM client",
            metadata={"prompt": prompt},
        )

    def _format_context(
        self,
        transcript: Optional[Transcript],
        outcome: Optional[Outcome],
    ) -> str:
        """Format transcript and outcome data for the prompt."""
        parts = []

        if transcript:
            parts.append("=== TRANSCRIPT ===")
            for i, step in enumerate(transcript.steps):
                parts.append(f"Step {i+1} ({step.step_type.value}): {step.content}")
            if transcript.outcome:
                parts.append(f"\nOutcome: Success={transcript.outcome.success}")

        if outcome:
            parts.append("\n=== OUTCOME ===")
            parts.append(f"Success: {outcome.success}")
            parts.append(f"Final State: {outcome.final_state}")

        return "\n".join(parts)


class HumanGrader(Grader):
    """
    Grader that requires human judgment.

    Human graders present the transcript and outcome to a human reviewer
    and collect their assessment. They're useful for subjective criteria
    or when automated grading is insufficient.

    Example use cases:
    - Evaluate user experience
    - Assess creativity or style
    - Judge complex multi-step reasoning
    - Validate edge cases
    """

    def __init__(
        self,
        name: str,
        instructions: str,
        weight: float = 1.0,
        required: bool = False,
        review_callback: Optional[Callable[[Transcript, Outcome], GraderResult]] = None,
    ):
        """
        Initialize a human grader.

        Args:
            name: Name of this grader
            instructions: Instructions for the human reviewer
            weight: Weight for scoring
            required: Whether this grader must pass
            review_callback: Optional callback to collect human review
        """
        super().__init__(name, weight, required, GraderType.HUMAN)
        self.instructions = instructions
        self.review_callback = review_callback

    def grade(
        self,
        transcript: Optional[Transcript] = None,
        outcome: Optional[Outcome] = None,
    ) -> GraderResult:
        """
        Collect human judgment.

        In a real implementation, this would present the transcript and outcome
        to a human reviewer (via UI, CLI, or API) and collect their score.
        """
        if self.review_callback and transcript and outcome:
            return self.review_callback(transcript, outcome)

        # Placeholder for human review
        return GraderResult(
            score=0.0,
            passed=False,
            explanation="Human review required - no callback provided",
            metadata={"instructions": self.instructions},
        )


# Convenience functions for creating common graders

def outcome_success_grader(name: str = "Outcome Success", weight: float = 1.0) -> CodeGrader:
    """Create a grader that checks if the outcome was successful."""
    def check_success(transcript: Optional[Transcript], outcome: Optional[Outcome]) -> GraderResult:
        if outcome is None:
            return GraderResult(score=0.0, passed=False, explanation="No outcome available")
        return GraderResult(
            score=1.0 if outcome.success else 0.0,
            passed=outcome.success,
            explanation="Task completed successfully" if outcome.success else "Task failed",
        )
    return CodeGrader(name, check_success, weight)


def tool_call_count_grader(
    tool_name: str,
    expected_count: int,
    name: Optional[str] = None,
    weight: float = 1.0,
) -> CodeGrader:
    """Create a grader that checks if a specific tool was called the expected number of times."""
    grader_name = name or f"{tool_name} Call Count"

    def check_tool_calls(transcript: Optional[Transcript], outcome: Optional[Outcome]) -> GraderResult:
        if transcript is None:
            return GraderResult(score=0.0, passed=False, explanation="No transcript available")

        count = sum(
            1 for step in transcript.steps
            if step.tool_call and step.tool_call.tool_name == tool_name
        )

        passed = count == expected_count
        score = 1.0 if passed else max(0.0, 1.0 - abs(count - expected_count) / expected_count)

        return GraderResult(
            score=score,
            passed=passed,
            explanation=f"{tool_name} called {count} times (expected {expected_count})",
            metadata={"actual_count": count, "expected_count": expected_count},
        )

    return CodeGrader(grader_name, check_tool_calls, weight)


def final_state_grader(
    key: str,
    expected_value: Any,
    name: Optional[str] = None,
    weight: float = 1.0,
) -> CodeGrader:
    """Create a grader that checks if a specific key in the final state has the expected value."""
    grader_name = name or f"Final State: {key}"

    def check_final_state(transcript: Optional[Transcript], outcome: Optional[Outcome]) -> GraderResult:
        if outcome is None:
            return GraderResult(score=0.0, passed=False, explanation="No outcome available")

        actual_value = outcome.final_state.get(key)
        passed = actual_value == expected_value

        return GraderResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            explanation=f"{key} = {actual_value} (expected {expected_value})",
            metadata={"actual_value": actual_value, "expected_value": expected_value},
        )

    return CodeGrader(grader_name, check_final_state, weight)
