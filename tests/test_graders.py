"""
Unit tests for agent_evals.graders module.
"""

import pytest

from agent_evals import (
    GraderType,
    GraderResult,
    Grader,
    CodeGrader,
    ModelGrader,
    HumanGrader,
    outcome_success_grader,
    tool_call_count_grader,
    final_state_grader,
    Transcript,
    Outcome,
    Step,
    StepType,
    ToolCall,
)


class TestGraderResult:
    """Tests for GraderResult class."""

    def test_grader_result_creation(self):
        """Test creating a grader result."""
        result = GraderResult(
            score=0.85,
            passed=True,
            explanation="Good performance",
        )

        assert result.score == 0.85
        assert result.passed is True
        assert result.explanation == "Good performance"
        assert result.confidence is None

    def test_grader_result_with_confidence(self):
        """Test creating a grader result with confidence."""
        result = GraderResult(
            score=0.7,
            passed=True,
            explanation="Likely correct",
            confidence=0.9,
        )

        assert result.confidence == 0.9

    def test_grader_result_to_dict(self):
        """Test converting grader result to dictionary."""
        result = GraderResult(
            score=0.5,
            passed=False,
            explanation="Failed",
            metadata={"detail": "error"},
        )

        data = result.to_dict()
        assert data["score"] == 0.5
        assert data["passed"] is False
        assert data["explanation"] == "Failed"
        assert data["metadata"]["detail"] == "error"


class TestCodeGrader:
    """Tests for CodeGrader class."""

    def test_code_grader_creation(self):
        """Test creating a code grader."""
        def grade_fn(transcript, outcome):
            return GraderResult(score=1.0, passed=True, explanation="Pass")

        grader = CodeGrader("Test Grader", grade_fn)

        assert grader.name == "Test Grader"
        assert grader.weight == 1.0
        assert grader.required is False
        assert grader.grader_type == GraderType.CODE

    def test_code_grader_with_weight(self):
        """Test creating a code grader with custom weight."""
        def grade_fn(transcript, outcome):
            return GraderResult(score=1.0, passed=True, explanation="Pass")

        grader = CodeGrader("Weighted Grader", grade_fn, weight=2.5)

        assert grader.weight == 2.5

    def test_code_grader_required(self):
        """Test creating a required code grader."""
        def grade_fn(transcript, outcome):
            return GraderResult(score=1.0, passed=True, explanation="Pass")

        grader = CodeGrader("Required Grader", grade_fn, required=True)

        assert grader.required is True

    def test_code_grader_execution(self, sample_transcript, sample_outcome):
        """Test executing a code grader."""
        def grade_fn(transcript, outcome):
            if outcome and outcome.success:
                return GraderResult(score=1.0, passed=True, explanation="Success")
            return GraderResult(score=0.0, passed=False, explanation="Failed")

        grader = CodeGrader("Success Checker", grade_fn)
        result = grader.grade(transcript=sample_transcript, outcome=sample_outcome)

        assert result.score == 1.0
        assert result.passed is True

    def test_code_grader_repr(self):
        """Test code grader string representation."""
        def grade_fn(transcript, outcome):
            return GraderResult(score=1.0, passed=True, explanation="Pass")

        grader = CodeGrader("Test Grader", grade_fn, weight=2.0)
        repr_str = repr(grader)

        assert "CodeGrader" in repr_str
        assert "Test Grader" in repr_str
        assert "2.0" in repr_str


class TestModelGrader:
    """Tests for ModelGrader class."""

    def test_model_grader_creation(self):
        """Test creating a model grader."""
        grader = ModelGrader(
            name="LLM Grader",
            prompt_template="Evaluate: {context}",
        )

        assert grader.name == "LLM Grader"
        assert grader.grader_type == GraderType.MODEL
        assert grader.prompt_template == "Evaluate: {context}"
        assert grader.pass_threshold == 0.7

    def test_model_grader_without_client(self, sample_transcript, sample_outcome):
        """Test model grader without client returns error."""
        grader = ModelGrader(
            name="LLM Grader",
            prompt_template="Evaluate: {context}",
        )

        result = grader.grade(transcript=sample_transcript, outcome=sample_outcome)

        assert result.score == 0.0
        assert result.passed is False
        assert "No model client" in result.explanation

    def test_model_grader_format_context(self, sample_transcript, sample_outcome):
        """Test model grader context formatting."""
        grader = ModelGrader(
            name="LLM Grader",
            prompt_template="Evaluate: {context}",
        )

        context = grader._format_context(sample_transcript, sample_outcome)

        assert "TRANSCRIPT" in context
        assert "OUTCOME" in context
        assert "trial_123" in context or "Success" in context


class TestHumanGrader:
    """Tests for HumanGrader class."""

    def test_human_grader_creation(self):
        """Test creating a human grader."""
        grader = HumanGrader(
            name="Human Review",
            instructions="Please review the output",
        )

        assert grader.name == "Human Review"
        assert grader.grader_type == GraderType.HUMAN
        assert grader.instructions == "Please review the output"

    def test_human_grader_without_callback(self, sample_transcript, sample_outcome):
        """Test human grader without callback."""
        grader = HumanGrader(
            name="Human Review",
            instructions="Review this",
        )

        result = grader.grade(transcript=sample_transcript, outcome=sample_outcome)

        assert result.score == 0.0
        assert result.passed is False
        assert "Human review required" in result.explanation

    def test_human_grader_with_callback(self, sample_transcript, sample_outcome):
        """Test human grader with callback."""
        def review_callback(transcript, outcome):
            return GraderResult(
                score=0.9,
                passed=True,
                explanation="Human approved",
            )

        grader = HumanGrader(
            name="Human Review",
            instructions="Review this",
            review_callback=review_callback,
        )

        result = grader.grade(transcript=sample_transcript, outcome=sample_outcome)

        assert result.score == 0.9
        assert result.passed is True
        assert result.explanation == "Human approved"


class TestConvenienceGraders:
    """Tests for convenience grader functions."""

    def test_outcome_success_grader_pass(self):
        """Test outcome success grader with successful outcome."""
        grader = outcome_success_grader()
        outcome = Outcome(success=True, final_state={})

        result = grader.grade(transcript=None, outcome=outcome)

        assert result.score == 1.0
        assert result.passed is True
        assert "successfully" in result.explanation.lower()

    def test_outcome_success_grader_fail(self):
        """Test outcome success grader with failed outcome."""
        grader = outcome_success_grader()
        outcome = Outcome(success=False, final_state={})

        result = grader.grade(transcript=None, outcome=outcome)

        assert result.score == 0.0
        assert result.passed is False

    def test_outcome_success_grader_no_outcome(self):
        """Test outcome success grader with no outcome."""
        grader = outcome_success_grader()

        result = grader.grade(transcript=None, outcome=None)

        assert result.score == 0.0
        assert result.passed is False

    def test_tool_call_count_grader_exact(self):
        """Test tool call count grader with exact match."""
        grader = tool_call_count_grader("calculator", 2)

        transcript = Transcript(trial_id="test")
        transcript.add_step(Step(
            step_type=StepType.TOOL_CALL,
            content="Call 1",
            tool_call=ToolCall("calculator", {}),
        ))
        transcript.add_step(Step(
            step_type=StepType.TOOL_CALL,
            content="Call 2",
            tool_call=ToolCall("calculator", {}),
        ))

        result = grader.grade(transcript=transcript, outcome=None)

        assert result.score == 1.0
        assert result.passed is True
        assert "2 times" in result.explanation

    def test_tool_call_count_grader_mismatch(self):
        """Test tool call count grader with mismatch."""
        grader = tool_call_count_grader("calculator", 2)

        transcript = Transcript(trial_id="test")
        transcript.add_step(Step(
            step_type=StepType.TOOL_CALL,
            content="Call 1",
            tool_call=ToolCall("calculator", {}),
        ))

        result = grader.grade(transcript=transcript, outcome=None)

        assert result.score < 1.0
        assert result.passed is False

    def test_tool_call_count_grader_different_tool(self):
        """Test tool call count grader ignores different tools."""
        grader = tool_call_count_grader("calculator", 1)

        transcript = Transcript(trial_id="test")
        transcript.add_step(Step(
            step_type=StepType.TOOL_CALL,
            content="Call",
            tool_call=ToolCall("other_tool", {}),
        ))

        result = grader.grade(transcript=transcript, outcome=None)

        assert result.passed is False
        assert result.metadata["actual_count"] == 0

    def test_final_state_grader_match(self):
        """Test final state grader with matching value."""
        grader = final_state_grader("result", 42)

        outcome = Outcome(success=True, final_state={"result": 42})
        result = grader.grade(transcript=None, outcome=outcome)

        assert result.score == 1.0
        assert result.passed is True

    def test_final_state_grader_mismatch(self):
        """Test final state grader with mismatched value."""
        grader = final_state_grader("result", 42)

        outcome = Outcome(success=True, final_state={"result": 100})
        result = grader.grade(transcript=None, outcome=outcome)

        assert result.score == 0.0
        assert result.passed is False

    def test_final_state_grader_missing_key(self):
        """Test final state grader with missing key."""
        grader = final_state_grader("result", 42)

        outcome = Outcome(success=True, final_state={})
        result = grader.grade(transcript=None, outcome=outcome)

        assert result.score == 0.0
        assert result.passed is False
        assert result.metadata["actual_value"] is None

    def test_final_state_grader_custom_name(self):
        """Test final state grader with custom name."""
        grader = final_state_grader("status", "complete", name="Status Check")

        assert grader.name == "Status Check"
