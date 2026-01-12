"""
Unit tests for agent_evals.models module.
"""

import pytest
from datetime import datetime
import time

from agent_evals import (
    Task,
    TaskDifficulty,
    Trial,
    Transcript,
    Outcome,
    Step,
    StepType,
    ToolCall,
)


class TestToolCall:
    """Tests for ToolCall class."""

    def test_tool_call_creation(self):
        """Test creating a tool call."""
        tool_call = ToolCall(
            tool_name="test_tool",
            arguments={"param": "value"},
            result="success",
            success=True,
        )

        assert tool_call.tool_name == "test_tool"
        assert tool_call.arguments == {"param": "value"}
        assert tool_call.result == "success"
        assert tool_call.success is True
        assert tool_call.error is None
        assert isinstance(tool_call.timestamp, datetime)

    def test_tool_call_with_error(self):
        """Test creating a tool call with error."""
        tool_call = ToolCall(
            tool_name="failing_tool",
            arguments={},
            success=False,
            error="Tool execution failed",
        )

        assert tool_call.success is False
        assert tool_call.error == "Tool execution failed"

    def test_tool_call_to_dict(self, sample_tool_call):
        """Test converting tool call to dictionary."""
        data = sample_tool_call.to_dict()

        assert data["tool_name"] == "calculator"
        assert data["arguments"]["operation"] == "add"
        assert data["result"] == 30
        assert data["success"] is True
        assert "timestamp" in data


class TestStep:
    """Tests for Step class."""

    def test_step_creation(self):
        """Test creating a step."""
        step = Step(
            step_type=StepType.REASONING,
            content="Analyzing the problem",
        )

        assert step.step_type == StepType.REASONING
        assert step.content == "Analyzing the problem"
        assert step.tool_call is None
        assert isinstance(step.timestamp, datetime)
        assert isinstance(step.metadata, dict)

    def test_step_with_tool_call(self, sample_tool_call):
        """Test creating a step with a tool call."""
        step = Step(
            step_type=StepType.TOOL_CALL,
            content="Calling calculator",
            tool_call=sample_tool_call,
        )

        assert step.step_type == StepType.TOOL_CALL
        assert step.tool_call == sample_tool_call

    def test_step_to_dict(self, sample_step):
        """Test converting step to dictionary."""
        data = sample_step.to_dict()

        assert data["step_type"] == StepType.TOOL_CALL.value
        assert data["content"] == "Called calculator"
        assert data["tool_call"] is not None
        assert "timestamp" in data


class TestOutcome:
    """Tests for Outcome class."""

    def test_outcome_creation(self):
        """Test creating an outcome."""
        outcome = Outcome(
            success=True,
            final_state={"status": "complete"},
        )

        assert outcome.success is True
        assert outcome.final_state == {"status": "complete"}
        assert isinstance(outcome.artifacts, dict)
        assert isinstance(outcome.metadata, dict)

    def test_outcome_with_artifacts(self):
        """Test creating an outcome with artifacts."""
        outcome = Outcome(
            success=True,
            final_state={},
            artifacts={"file": "output.txt"},
        )

        assert outcome.artifacts["file"] == "output.txt"

    def test_outcome_to_dict(self, sample_outcome):
        """Test converting outcome to dictionary."""
        data = sample_outcome.to_dict()

        assert data["success"] is True
        assert data["final_state"]["result"] == 30
        assert "artifacts" in data


class TestTranscript:
    """Tests for Transcript class."""

    def test_transcript_creation(self):
        """Test creating a transcript."""
        transcript = Transcript(trial_id="test_123")

        assert transcript.trial_id == "test_123"
        assert len(transcript.steps) == 0
        assert transcript.outcome is None
        assert transcript.end_time is None
        assert isinstance(transcript.start_time, datetime)

    def test_add_step(self):
        """Test adding steps to transcript."""
        transcript = Transcript(trial_id="test_123")
        step = Step(step_type=StepType.REASONING, content="Thinking")

        transcript.add_step(step)

        assert len(transcript.steps) == 1
        assert transcript.steps[0] == step

    def test_complete_transcript(self):
        """Test completing a transcript."""
        transcript = Transcript(trial_id="test_123")
        outcome = Outcome(success=True, final_state={})

        transcript.complete(outcome)

        assert transcript.outcome == outcome
        assert transcript.end_time is not None

    def test_duration_calculation(self):
        """Test duration calculation."""
        transcript = Transcript(trial_id="test_123")
        time.sleep(0.1)  # Small delay
        outcome = Outcome(success=True, final_state={})
        transcript.complete(outcome)

        duration = transcript.duration
        assert duration is not None
        assert duration >= 0.1

    def test_num_steps(self, sample_transcript):
        """Test counting steps."""
        assert sample_transcript.num_steps == 2

    def test_num_tool_calls(self, sample_transcript):
        """Test counting tool calls."""
        assert sample_transcript.num_tool_calls == 1

    def test_transcript_to_dict(self, sample_transcript):
        """Test converting transcript to dictionary."""
        data = sample_transcript.to_dict()

        assert data["trial_id"] == "trial_123"
        assert len(data["steps"]) == 2
        assert data["outcome"] is not None
        assert data["num_steps"] == 2
        assert data["num_tool_calls"] == 1


class TestTrial:
    """Tests for Trial class."""

    def test_trial_creation(self):
        """Test creating a trial."""
        trial = Trial(trial_id="trial_1", task_id="task_1")

        assert trial.trial_id == "trial_1"
        assert trial.task_id == "task_1"
        assert trial.transcript is None
        assert trial.completed is False
        assert trial.error is None

    def test_trial_with_transcript(self, sample_transcript):
        """Test creating a trial with transcript."""
        trial = Trial(
            trial_id="trial_1",
            task_id="task_1",
            transcript=sample_transcript,
            completed=True,
        )

        assert trial.transcript == sample_transcript
        assert trial.completed is True

    def test_trial_with_error(self):
        """Test creating a trial with error."""
        trial = Trial(
            trial_id="trial_1",
            task_id="task_1",
            error="Execution failed",
        )

        assert trial.error == "Execution failed"

    def test_trial_to_dict(self, sample_trial):
        """Test converting trial to dictionary."""
        data = sample_trial.to_dict()

        assert data["trial_id"] == "trial_123"
        assert data["task_id"] == "test_task_1"
        assert data["completed"] is True
        assert data["transcript"] is not None


class TestTask:
    """Tests for Task class."""

    def test_task_creation(self):
        """Test creating a task."""
        task = Task(
            task_id="task_1",
            name="Test Task",
            description="A test task",
            inputs={"x": 1},
            success_criteria="x should be 1",
        )

        assert task.task_id == "task_1"
        assert task.name == "Test Task"
        assert task.description == "A test task"
        assert task.inputs == {"x": 1}
        assert task.success_criteria == "x should be 1"
        assert task.difficulty == TaskDifficulty.MEDIUM

    def test_task_with_difficulty(self):
        """Test creating a task with specific difficulty."""
        task = Task(
            task_id="task_1",
            name="Hard Task",
            description="A hard task",
            inputs={},
            success_criteria="Complete successfully",
            difficulty=TaskDifficulty.HARD,
        )

        assert task.difficulty == TaskDifficulty.HARD

    def test_task_with_tags(self):
        """Test creating a task with tags."""
        task = Task(
            task_id="task_1",
            name="Tagged Task",
            description="Task with tags",
            inputs={},
            success_criteria="Success",
            tags=["math", "calculation"],
        )

        assert "math" in task.tags
        assert "calculation" in task.tags

    def test_task_setup_teardown(self):
        """Test task setup and teardown functions."""
        setup_called = []
        teardown_called = []

        def setup():
            setup_called.append(True)

        def teardown():
            teardown_called.append(True)

        task = Task(
            task_id="task_1",
            name="Test Task",
            description="Task with setup/teardown",
            inputs={},
            success_criteria="Success",
            setup=setup,
            teardown=teardown,
        )

        task.run_setup()
        assert len(setup_called) == 1

        task.run_teardown()
        assert len(teardown_called) == 1

    def test_task_to_dict(self, sample_task):
        """Test converting task to dictionary."""
        data = sample_task.to_dict()

        assert data["task_id"] == "test_task_1"
        assert data["name"] == "Test Task"
        assert data["description"] == "A test task for unit tests"
        assert data["inputs"]["x"] == 10
        assert data["difficulty"] == TaskDifficulty.EASY.value
        assert "test" in data["tags"]
