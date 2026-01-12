"""
Scoring strategies for combining multiple grader results.

Different strategies for combining grader scores into a final task score:
- WeightedScoring: Combines weighted grader scores, requires threshold
- BinaryScoring: All graders must pass
- HybridScoring: Combines required graders (must pass) with weighted optional graders
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum

from .graders import Grader, GraderResult


class ScoringMode(Enum):
    """Scoring strategy mode."""
    WEIGHTED = "weighted"
    BINARY = "binary"
    HYBRID = "hybrid"


@dataclass
class TaskScore:
    """
    Final score for a task combining all grader results.

    Attributes:
        overall_score: Overall score (0.0 to 1.0)
        passed: Whether the task passed
        grader_results: Individual grader results
        explanation: Explanation of the overall score
        metadata: Additional metadata
    """
    overall_score: float
    passed: bool
    grader_results: Dict[str, GraderResult] = field(default_factory=dict)
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "overall_score": self.overall_score,
            "passed": self.passed,
            "grader_results": {
                name: result.to_dict() for name, result in self.grader_results.items()
            },
            "explanation": self.explanation,
            "metadata": self.metadata,
        }


class ScoringStrategy(ABC):
    """Abstract base class for scoring strategies."""

    def __init__(self, mode: ScoringMode):
        """
        Initialize a scoring strategy.

        Args:
            mode: The scoring mode
        """
        self.mode = mode

    @abstractmethod
    def compute_score(
        self,
        graders: List[Grader],
        grader_results: Dict[str, GraderResult],
    ) -> TaskScore:
        """
        Compute the final task score from grader results.

        Args:
            graders: List of graders used
            grader_results: Results from each grader (keyed by grader name)

        Returns:
            TaskScore with overall score and pass/fail status
        """
        pass


class WeightedScoring(ScoringStrategy):
    """
    Weighted scoring strategy.

    Combines grader scores using their weights. The task passes if the
    weighted average score meets or exceeds the threshold.

    Formula:
        score = sum(grader.weight * result.score) / sum(grader.weight)

    Example:
        - Grader A: weight=2, score=1.0 -> contributes 2.0
        - Grader B: weight=1, score=0.5 -> contributes 0.5
        - Total: (2.0 + 0.5) / (2 + 1) = 0.833
    """

    def __init__(self, pass_threshold: float = 0.8):
        """
        Initialize weighted scoring.

        Args:
            pass_threshold: Minimum score required to pass (0.0 to 1.0)
        """
        super().__init__(ScoringMode.WEIGHTED)
        self.pass_threshold = pass_threshold

    def compute_score(
        self,
        graders: List[Grader],
        grader_results: Dict[str, GraderResult],
    ) -> TaskScore:
        """Compute weighted average score."""
        if not graders:
            return TaskScore(
                overall_score=0.0,
                passed=False,
                explanation="No graders provided",
            )

        total_weight = 0.0
        weighted_sum = 0.0
        explanations = []

        for grader in graders:
            result = grader_results.get(grader.name)
            if result is None:
                explanations.append(f"{grader.name}: No result")
                continue

            total_weight += grader.weight
            weighted_sum += grader.weight * result.score
            explanations.append(
                f"{grader.name} (weight={grader.weight}): "
                f"score={result.score:.2f} - {result.explanation}"
            )

        if total_weight == 0:
            overall_score = 0.0
        else:
            overall_score = weighted_sum / total_weight

        passed = overall_score >= self.pass_threshold

        return TaskScore(
            overall_score=overall_score,
            passed=passed,
            grader_results=grader_results,
            explanation=f"Weighted score: {overall_score:.2f} (threshold: {self.pass_threshold:.2f})\n"
            + "\n".join(explanations),
            metadata={
                "pass_threshold": self.pass_threshold,
                "total_weight": total_weight,
                "weighted_sum": weighted_sum,
            },
        )


class BinaryScoring(ScoringStrategy):
    """
    Binary scoring strategy.

    All graders must pass for the task to pass. The overall score is 1.0
    if all graders pass, 0.0 otherwise.

    This is useful for tasks with strict requirements where partial credit
    doesn't make sense.

    Example:
        - Grader A: passed=True
        - Grader B: passed=True
        - Grader C: passed=False
        - Result: Task fails (score=0.0)
    """

    def __init__(self):
        """Initialize binary scoring."""
        super().__init__(ScoringMode.BINARY)

    def compute_score(
        self,
        graders: List[Grader],
        grader_results: Dict[str, GraderResult],
    ) -> TaskScore:
        """Check if all graders passed."""
        if not graders:
            return TaskScore(
                overall_score=0.0,
                passed=False,
                explanation="No graders provided",
            )

        all_passed = True
        failed_graders = []
        explanations = []

        for grader in graders:
            result = grader_results.get(grader.name)
            if result is None:
                all_passed = False
                failed_graders.append(grader.name)
                explanations.append(f"{grader.name}: No result")
                continue

            if not result.passed:
                all_passed = False
                failed_graders.append(grader.name)

            status = "✓" if result.passed else "✗"
            explanations.append(
                f"{status} {grader.name}: score={result.score:.2f} - {result.explanation}"
            )

        overall_score = 1.0 if all_passed else 0.0

        explanation_text = "All graders passed" if all_passed else f"Failed graders: {', '.join(failed_graders)}"
        explanation_text += "\n" + "\n".join(explanations)

        return TaskScore(
            overall_score=overall_score,
            passed=all_passed,
            grader_results=grader_results,
            explanation=explanation_text,
            metadata={"failed_graders": failed_graders},
        )


class HybridScoring(ScoringStrategy):
    """
    Hybrid scoring strategy.

    Combines required graders (must all pass) with weighted optional graders.
    The task passes only if:
    1. All required graders pass, AND
    2. The weighted score of optional graders meets the threshold

    This is useful for tasks with both strict requirements and flexible
    quality criteria.

    Example:
        Required graders (must all pass):
        - File created: passed=True
        - Valid format: passed=True

        Optional graders (weighted, threshold=0.7):
        - Code quality (weight=2): score=0.8
        - Performance (weight=1): score=0.6
        - Weighted average: (2*0.8 + 1*0.6) / 3 = 0.733

        Result: Task passes (all required pass, weighted score >= 0.7)
    """

    def __init__(self, pass_threshold: float = 0.8):
        """
        Initialize hybrid scoring.

        Args:
            pass_threshold: Minimum weighted score for optional graders (0.0 to 1.0)
        """
        super().__init__(ScoringMode.HYBRID)
        self.pass_threshold = pass_threshold

    def compute_score(
        self,
        graders: List[Grader],
        grader_results: Dict[str, GraderResult],
    ) -> TaskScore:
        """Compute hybrid score with required and weighted optional graders."""
        if not graders:
            return TaskScore(
                overall_score=0.0,
                passed=False,
                explanation="No graders provided",
            )

        required_graders = [g for g in graders if g.required]
        optional_graders = [g for g in graders if not g.required]

        # Check required graders
        all_required_passed = True
        failed_required = []
        required_explanations = []

        for grader in required_graders:
            result = grader_results.get(grader.name)
            if result is None or not result.passed:
                all_required_passed = False
                failed_required.append(grader.name)

            if result:
                status = "✓" if result.passed else "✗"
                required_explanations.append(
                    f"{status} {grader.name} (REQUIRED): {result.explanation}"
                )

        # Compute weighted score for optional graders
        optional_score = 0.0
        if optional_graders:
            total_weight = sum(g.weight for g in optional_graders)
            weighted_sum = 0.0
            optional_explanations = []

            for grader in optional_graders:
                result = grader_results.get(grader.name)
                if result:
                    weighted_sum += grader.weight * result.score
                    optional_explanations.append(
                        f"{grader.name} (weight={grader.weight}): "
                        f"score={result.score:.2f} - {result.explanation}"
                    )

            optional_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            optional_explanations = ["No optional graders"]
            optional_score = 1.0  # If no optional graders, consider this part passed

        # Task passes if all required pass AND optional score meets threshold
        optional_meets_threshold = optional_score >= self.pass_threshold
        passed = all_required_passed and optional_meets_threshold

        # Combine scores: if all required pass, use optional score; otherwise 0
        overall_score = optional_score if all_required_passed else 0.0

        explanation_parts = ["=== REQUIRED GRADERS ==="]
        if required_graders:
            explanation_parts.extend(required_explanations)
            if not all_required_passed:
                explanation_parts.append(f"Failed required: {', '.join(failed_required)}")
        else:
            explanation_parts.append("None")

        explanation_parts.append("\n=== OPTIONAL GRADERS ===")
        explanation_parts.extend(optional_explanations)
        explanation_parts.append(
            f"\nOptional score: {optional_score:.2f} (threshold: {self.pass_threshold:.2f})"
        )

        explanation_parts.append(
            f"\n=== RESULT ===\n"
            f"Overall score: {overall_score:.2f}\n"
            f"Passed: {passed}"
        )

        return TaskScore(
            overall_score=overall_score,
            passed=passed,
            grader_results=grader_results,
            explanation="\n".join(explanation_parts),
            metadata={
                "all_required_passed": all_required_passed,
                "failed_required": failed_required,
                "optional_score": optional_score,
                "optional_meets_threshold": optional_meets_threshold,
                "pass_threshold": self.pass_threshold,
            },
        )
