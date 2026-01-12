"""
Unit tests for agent_evals.scoring module.
"""

import pytest

from agent_evals import (
    ScoringMode,
    TaskScore,
    ScoringStrategy,
    WeightedScoring,
    BinaryScoring,
    HybridScoring,
    CodeGrader,
    GraderResult,
)


class TestTaskScore:
    """Tests for TaskScore class."""

    def test_task_score_creation(self):
        """Test creating a task score."""
        score = TaskScore(
            overall_score=0.85,
            passed=True,
            explanation="Good performance",
        )

        assert score.overall_score == 0.85
        assert score.passed is True
        assert score.explanation == "Good performance"
        assert isinstance(score.grader_results, dict)

    def test_task_score_with_grader_results(self):
        """Test task score with grader results."""
        grader_results = {
            "grader1": GraderResult(1.0, True, "Pass"),
            "grader2": GraderResult(0.8, True, "Good"),
        }

        score = TaskScore(
            overall_score=0.9,
            passed=True,
            grader_results=grader_results,
        )

        assert len(score.grader_results) == 2
        assert "grader1" in score.grader_results

    def test_task_score_to_dict(self):
        """Test converting task score to dictionary."""
        grader_results = {
            "test": GraderResult(1.0, True, "Pass"),
        }

        score = TaskScore(
            overall_score=1.0,
            passed=True,
            grader_results=grader_results,
        )

        data = score.to_dict()
        assert data["overall_score"] == 1.0
        assert data["passed"] is True
        assert "grader_results" in data
        assert "test" in data["grader_results"]


class TestWeightedScoring:
    """Tests for WeightedScoring strategy."""

    def test_weighted_scoring_creation(self):
        """Test creating weighted scoring strategy."""
        scoring = WeightedScoring(pass_threshold=0.8)

        assert scoring.mode == ScoringMode.WEIGHTED
        assert scoring.pass_threshold == 0.8

    def test_weighted_scoring_all_pass(self, simple_grader):
        """Test weighted scoring with all graders passing."""
        grader1 = simple_grader
        grader2 = CodeGrader(
            "Grader 2",
            lambda t, o: GraderResult(1.0, True, "Pass"),
            weight=2.0,
        )

        graders = [grader1, grader2]
        grader_results = {
            grader1.name: grader1.grade(None, None),
            grader2.name: grader2.grade(None, None),
        }

        scoring = WeightedScoring(pass_threshold=0.8)
        task_score = scoring.compute_score(graders, grader_results)

        assert task_score.overall_score == 1.0
        assert task_score.passed is True

    def test_weighted_scoring_partial_credit(self):
        """Test weighted scoring with partial credit."""
        grader1 = CodeGrader(
            "Grader 1",
            lambda t, o: GraderResult(1.0, True, "Pass"),
            weight=1.0,
        )
        grader2 = CodeGrader(
            "Grader 2",
            lambda t, o: GraderResult(0.5, False, "Partial"),
            weight=1.0,
        )

        graders = [grader1, grader2]
        grader_results = {
            grader1.name: grader1.grade(None, None),
            grader2.name: grader2.grade(None, None),
        }

        scoring = WeightedScoring(pass_threshold=0.7)
        task_score = scoring.compute_score(graders, grader_results)

        assert task_score.overall_score == 0.75  # (1.0 + 0.5) / 2
        assert task_score.passed is True  # 0.75 >= 0.7

    def test_weighted_scoring_below_threshold(self):
        """Test weighted scoring below threshold."""
        grader1 = CodeGrader(
            "Grader 1",
            lambda t, o: GraderResult(0.5, False, "Partial"),
            weight=1.0,
        )
        grader2 = CodeGrader(
            "Grader 2",
            lambda t, o: GraderResult(0.6, False, "Partial"),
            weight=1.0,
        )

        graders = [grader1, grader2]
        grader_results = {
            grader1.name: grader1.grade(None, None),
            grader2.name: grader2.grade(None, None),
        }

        scoring = WeightedScoring(pass_threshold=0.8)
        task_score = scoring.compute_score(graders, grader_results)

        assert task_score.overall_score == 0.55  # (0.5 + 0.6) / 2
        assert task_score.passed is False  # 0.55 < 0.8

    def test_weighted_scoring_different_weights(self):
        """Test weighted scoring with different weights."""
        grader1 = CodeGrader(
            "Important",
            lambda t, o: GraderResult(1.0, True, "Pass"),
            weight=3.0,
        )
        grader2 = CodeGrader(
            "Less Important",
            lambda t, o: GraderResult(0.0, False, "Fail"),
            weight=1.0,
        )

        graders = [grader1, grader2]
        grader_results = {
            grader1.name: grader1.grade(None, None),
            grader2.name: grader2.grade(None, None),
        }

        scoring = WeightedScoring(pass_threshold=0.7)
        task_score = scoring.compute_score(graders, grader_results)

        assert task_score.overall_score == 0.75  # (3.0 + 0.0) / 4.0
        assert task_score.passed is True

    def test_weighted_scoring_no_graders(self):
        """Test weighted scoring with no graders."""
        scoring = WeightedScoring()
        task_score = scoring.compute_score([], {})

        assert task_score.overall_score == 0.0
        assert task_score.passed is False
        assert "No graders" in task_score.explanation


class TestBinaryScoring:
    """Tests for BinaryScoring strategy."""

    def test_binary_scoring_creation(self):
        """Test creating binary scoring strategy."""
        scoring = BinaryScoring()

        assert scoring.mode == ScoringMode.BINARY

    def test_binary_scoring_all_pass(self, simple_grader):
        """Test binary scoring with all graders passing."""
        grader1 = simple_grader
        grader2 = CodeGrader(
            "Grader 2",
            lambda t, o: GraderResult(1.0, True, "Pass"),
        )

        graders = [grader1, grader2]
        grader_results = {
            grader1.name: grader1.grade(None, None),
            grader2.name: grader2.grade(None, None),
        }

        scoring = BinaryScoring()
        task_score = scoring.compute_score(graders, grader_results)

        assert task_score.overall_score == 1.0
        assert task_score.passed is True

    def test_binary_scoring_one_fail(self, simple_grader, failing_grader):
        """Test binary scoring with one grader failing."""
        graders = [simple_grader, failing_grader]
        grader_results = {
            simple_grader.name: simple_grader.grade(None, None),
            failing_grader.name: failing_grader.grade(None, None),
        }

        scoring = BinaryScoring()
        task_score = scoring.compute_score(graders, grader_results)

        assert task_score.overall_score == 0.0
        assert task_score.passed is False
        assert "Failing Grader" in task_score.metadata["failed_graders"]

    def test_binary_scoring_all_fail(self):
        """Test binary scoring with all graders failing."""
        grader1 = CodeGrader(
            "Fail 1",
            lambda t, o: GraderResult(0.0, False, "Fail"),
        )
        grader2 = CodeGrader(
            "Fail 2",
            lambda t, o: GraderResult(0.0, False, "Fail"),
        )

        graders = [grader1, grader2]
        grader_results = {
            grader1.name: grader1.grade(None, None),
            grader2.name: grader2.grade(None, None),
        }

        scoring = BinaryScoring()
        task_score = scoring.compute_score(graders, grader_results)

        assert task_score.overall_score == 0.0
        assert task_score.passed is False
        assert len(task_score.metadata["failed_graders"]) == 2

    def test_binary_scoring_partial_credit_fails(self):
        """Test that partial credit still fails in binary scoring."""
        grader = CodeGrader(
            "Partial",
            lambda t, o: GraderResult(0.8, False, "Partial"),  # High score but failed
        )

        graders = [grader]
        grader_results = {grader.name: grader.grade(None, None)}

        scoring = BinaryScoring()
        task_score = scoring.compute_score(graders, grader_results)

        assert task_score.overall_score == 0.0
        assert task_score.passed is False

    def test_binary_scoring_no_graders(self):
        """Test binary scoring with no graders."""
        scoring = BinaryScoring()
        task_score = scoring.compute_score([], {})

        assert task_score.overall_score == 0.0
        assert task_score.passed is False


class TestHybridScoring:
    """Tests for HybridScoring strategy."""

    def test_hybrid_scoring_creation(self):
        """Test creating hybrid scoring strategy."""
        scoring = HybridScoring(pass_threshold=0.7)

        assert scoring.mode == ScoringMode.HYBRID
        assert scoring.pass_threshold == 0.7

    def test_hybrid_scoring_all_pass(self):
        """Test hybrid scoring with all graders passing."""
        required = CodeGrader(
            "Required",
            lambda t, o: GraderResult(1.0, True, "Pass"),
            required=True,
        )
        optional = CodeGrader(
            "Optional",
            lambda t, o: GraderResult(0.9, True, "Good"),
            weight=1.0,
        )

        graders = [required, optional]
        grader_results = {
            required.name: required.grade(None, None),
            optional.name: optional.grade(None, None),
        }

        scoring = HybridScoring(pass_threshold=0.8)
        task_score = scoring.compute_score(graders, grader_results)

        assert task_score.overall_score == 0.9
        assert task_score.passed is True
        assert task_score.metadata["all_required_passed"] is True

    def test_hybrid_scoring_required_fail(self):
        """Test hybrid scoring with required grader failing."""
        required = CodeGrader(
            "Required",
            lambda t, o: GraderResult(0.0, False, "Fail"),
            required=True,
        )
        optional = CodeGrader(
            "Optional",
            lambda t, o: GraderResult(1.0, True, "Pass"),
            weight=1.0,
        )

        graders = [required, optional]
        grader_results = {
            required.name: required.grade(None, None),
            optional.name: optional.grade(None, None),
        }

        scoring = HybridScoring(pass_threshold=0.8)
        task_score = scoring.compute_score(graders, grader_results)

        assert task_score.overall_score == 0.0  # Required fail = 0 overall
        assert task_score.passed is False
        assert "Required" in task_score.metadata["failed_required"]

    def test_hybrid_scoring_optional_below_threshold(self):
        """Test hybrid scoring with optional below threshold."""
        required = CodeGrader(
            "Required",
            lambda t, o: GraderResult(1.0, True, "Pass"),
            required=True,
        )
        optional = CodeGrader(
            "Optional",
            lambda t, o: GraderResult(0.5, False, "Low"),
            weight=1.0,
        )

        graders = [required, optional]
        grader_results = {
            required.name: required.grade(None, None),
            optional.name: optional.grade(None, None),
        }

        scoring = HybridScoring(pass_threshold=0.8)
        task_score = scoring.compute_score(graders, grader_results)

        assert task_score.overall_score == 0.5
        assert task_score.passed is False  # Optional score < threshold
        assert task_score.metadata["all_required_passed"] is True
        assert task_score.metadata["optional_meets_threshold"] is False

    def test_hybrid_scoring_multiple_required(self):
        """Test hybrid scoring with multiple required graders."""
        req1 = CodeGrader(
            "Required 1",
            lambda t, o: GraderResult(1.0, True, "Pass"),
            required=True,
        )
        req2 = CodeGrader(
            "Required 2",
            lambda t, o: GraderResult(1.0, True, "Pass"),
            required=True,
        )
        opt = CodeGrader(
            "Optional",
            lambda t, o: GraderResult(0.8, True, "Good"),
            weight=1.0,
        )

        graders = [req1, req2, opt]
        grader_results = {
            req1.name: req1.grade(None, None),
            req2.name: req2.grade(None, None),
            opt.name: opt.grade(None, None),
        }

        scoring = HybridScoring(pass_threshold=0.7)
        task_score = scoring.compute_score(graders, grader_results)

        assert task_score.passed is True

    def test_hybrid_scoring_no_optional_graders(self):
        """Test hybrid scoring with only required graders."""
        required = CodeGrader(
            "Required",
            lambda t, o: GraderResult(1.0, True, "Pass"),
            required=True,
        )

        graders = [required]
        grader_results = {required.name: required.grade(None, None)}

        scoring = HybridScoring(pass_threshold=0.8)
        task_score = scoring.compute_score(graders, grader_results)

        assert task_score.overall_score == 1.0  # No optional = 1.0
        assert task_score.passed is True

    def test_hybrid_scoring_weighted_optional(self):
        """Test hybrid scoring with weighted optional graders."""
        required = CodeGrader(
            "Required",
            lambda t, o: GraderResult(1.0, True, "Pass"),
            required=True,
        )
        opt1 = CodeGrader(
            "High Weight",
            lambda t, o: GraderResult(0.8, True, "Good"),
            weight=3.0,
        )
        opt2 = CodeGrader(
            "Low Weight",
            lambda t, o: GraderResult(0.4, False, "Poor"),
            weight=1.0,
        )

        graders = [required, opt1, opt2]
        grader_results = {
            required.name: required.grade(None, None),
            opt1.name: opt1.grade(None, None),
            opt2.name: opt2.grade(None, None),
        }

        scoring = HybridScoring(pass_threshold=0.7)
        task_score = scoring.compute_score(graders, grader_results)

        # Weighted score: (3.0 * 0.8 + 1.0 * 0.4) / 4.0 = 0.7
        assert task_score.overall_score == pytest.approx(0.7)
        assert task_score.passed is True

    def test_hybrid_scoring_no_graders(self):
        """Test hybrid scoring with no graders."""
        scoring = HybridScoring()
        task_score = scoring.compute_score([], {})

        assert task_score.overall_score == 0.0
        assert task_score.passed is False
