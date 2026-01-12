"""
Pytest configuration and shared fixtures for agent evaluation tests.
"""

import pytest
import uuid
from datetime import datetime

from agent_evals import (
    Task,
    TaskDifficulty,
    Trial,
    Transcript,
    Outcome,
    Step,
    StepType,
    ToolCall,
    CodeGrader,
    GraderResult,
)


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    return Task(
        task_id="test_task_1",
        name="Test Task",
        description="A test task for unit tests",
        inputs={"x": 10, "y": 20},
        success_criteria="x + y should equal 30",
        difficulty=TaskDifficulty.EASY,
        expected_outcome={"result": 30},
        tags=["test", "math"],
    )


@pytest.fixture
def sample_tool_call():
    """Create a sample tool call."""
    return ToolCall(
        tool_name="calculator",
        arguments={"operation": "add", "a": 10, "b": 20},
        result=30,
        success=True,
    )


@pytest.fixture
def sample_step(sample_tool_call):
    """Create a sample step."""
    return Step(
        step_type=StepType.TOOL_CALL,
        content="Called calculator",
        tool_call=sample_tool_call,
    )


@pytest.fixture
def sample_outcome():
    """Create a sample outcome."""
    return Outcome(
        success=True,
        final_state={"result": 30},
        artifacts={"log": "execution.log"},
    )


@pytest.fixture
def sample_transcript(sample_step, sample_outcome):
    """Create a sample transcript."""
    transcript = Transcript(trial_id="trial_123")
    transcript.add_step(sample_step)
    transcript.add_step(Step(
        step_type=StepType.REASONING,
        content="Computing the sum",
    ))
    transcript.complete(sample_outcome)
    return transcript


@pytest.fixture
def sample_trial(sample_transcript):
    """Create a sample trial."""
    return Trial(
        trial_id="trial_123",
        task_id="test_task_1",
        transcript=sample_transcript,
        completed=True,
    )


@pytest.fixture
def simple_grader():
    """Create a simple passing grader."""
    def grade_fn(transcript, outcome):
        return GraderResult(
            score=1.0,
            passed=True,
            explanation="Always passes",
        )
    return CodeGrader("Simple Grader", grade_fn)


@pytest.fixture
def failing_grader():
    """Create a simple failing grader."""
    def grade_fn(transcript, outcome):
        return GraderResult(
            score=0.0,
            passed=False,
            explanation="Always fails",
        )
    return CodeGrader("Failing Grader", grade_fn)


@pytest.fixture
def partial_credit_grader():
    """Create a grader that gives partial credit."""
    def grade_fn(transcript, outcome):
        return GraderResult(
            score=0.5,
            passed=False,
            explanation="Partial credit",
        )
    return CodeGrader("Partial Grader", grade_fn, weight=2.0)
