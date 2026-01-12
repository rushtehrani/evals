"""
Unit tests for agent_evals.harness module.
"""

import pytest
import time
from unittest.mock import Mock, patch

from agent_evals import (
    Task,
    TaskDifficulty,
    Transcript,
    Outcome,
    Step,
    StepType,
    CodeGrader,
    GraderResult,
    WeightedScoring,
    BinaryScoring,
    EvaluationHarness,
    TrialResult,
    TaskResult,
    EvaluationResult,
)


# Helper agent function for testing
def simple_agent(task: Task) -> Transcript:
    """Simple test agent that always succeeds."""
    transcript = Transcript(trial_id=f"trial_{task.task_id}")
    transcript.add_step(Step(
        step_type=StepType.REASONING,
        content=f"Processing task {task.name}",
    ))
    outcome = Outcome(
        success=True,
        final_state={"task_completed": True},
    )
    transcript.complete(outcome)
    return transcript


def failing_agent(task: Task) -> Transcript:
    """Test agent that always fails."""
    transcript = Transcript(trial_id=f"trial_{task.task_id}")
    transcript.add_step(Step(
        step_type=StepType.REASONING,
        content="Attempting task",
    ))
    outcome = Outcome(
        success=False,
        final_state={"task_completed": False},
    )
    transcript.complete(outcome)
    return transcript


def error_agent(task: Task) -> Transcript:
    """Test agent that raises an error."""
    raise RuntimeError("Agent execution failed")


class TestTrialResult:
    """Tests for TrialResult class."""

    def test_trial_result_creation(self, sample_trial):
        """Test creating a trial result."""
        result = TrialResult(trial=sample_trial)

        assert result.trial == sample_trial
        assert result.task_score is None
        assert result.error is None

    def test_trial_result_with_score(self, sample_trial):
        """Test trial result with score."""
        from agent_evals.scoring import TaskScore

        task_score = TaskScore(overall_score=0.9, passed=True)
        result = TrialResult(trial=sample_trial, task_score=task_score)

        assert result.task_score.overall_score == 0.9
        assert result.task_score.passed is True

    def test_trial_result_with_error(self, sample_trial):
        """Test trial result with error."""
        result = TrialResult(trial=sample_trial, error="Execution failed")

        assert result.error == "Execution failed"

    def test_trial_result_to_dict(self, sample_trial):
        """Test converting trial result to dictionary."""
        from agent_evals.scoring import TaskScore

        task_score = TaskScore(overall_score=1.0, passed=True)
        result = TrialResult(trial=sample_trial, task_score=task_score)

        data = result.to_dict()
        assert "trial" in data
        assert "task_score" in data
        assert data["task_score"]["overall_score"] == 1.0


class TestTaskResult:
    """Tests for TaskResult class."""

    def test_task_result_creation(self, sample_task):
        """Test creating a task result."""
        result = TaskResult(task=sample_task)

        assert result.task == sample_task
        assert len(result.trial_results) == 0
        assert result.aggregate_score is None
        assert result.pass_rate is None

    def test_num_trials(self, sample_task, sample_trial):
        """Test counting trials."""
        result = TaskResult(task=sample_task)
        result.trial_results = [
            TrialResult(trial=sample_trial),
            TrialResult(trial=sample_trial),
        ]

        assert result.num_trials == 2

    def test_num_passed(self, sample_task, sample_trial):
        """Test counting passed trials."""
        from agent_evals.scoring import TaskScore

        result = TaskResult(task=sample_task)
        result.trial_results = [
            TrialResult(trial=sample_trial, task_score=TaskScore(1.0, True)),
            TrialResult(trial=sample_trial, task_score=TaskScore(0.5, False)),
            TrialResult(trial=sample_trial, task_score=TaskScore(1.0, True)),
        ]

        assert result.num_passed == 2

    def test_num_failed(self, sample_task, sample_trial):
        """Test counting failed trials."""
        from agent_evals.scoring import TaskScore

        result = TaskResult(task=sample_task)
        result.trial_results = [
            TrialResult(trial=sample_trial, task_score=TaskScore(1.0, True)),
            TrialResult(trial=sample_trial, task_score=TaskScore(0.5, False)),
        ]

        assert result.num_failed == 1

    def test_num_errors(self, sample_task, sample_trial):
        """Test counting error trials."""
        result = TaskResult(task=sample_task)
        result.trial_results = [
            TrialResult(trial=sample_trial, error="Error 1"),
            TrialResult(trial=sample_trial, error="Error 2"),
        ]

        assert result.num_errors == 2

    def test_compute_aggregate_metrics(self, sample_task, sample_trial):
        """Test computing aggregate metrics."""
        from agent_evals.scoring import TaskScore

        result = TaskResult(task=sample_task)
        result.trial_results = [
            TrialResult(trial=sample_trial, task_score=TaskScore(1.0, True)),
            TrialResult(trial=sample_trial, task_score=TaskScore(0.8, True)),
            TrialResult(trial=sample_trial, task_score=TaskScore(0.6, False)),
        ]

        result.compute_aggregate_metrics()

        assert result.aggregate_score == pytest.approx(0.8)  # (1.0 + 0.8 + 0.6) / 3
        assert result.pass_rate == pytest.approx(2/3)
        assert "score_min" in result.metadata
        assert "score_max" in result.metadata

    def test_task_result_to_dict(self, sample_task):
        """Test converting task result to dictionary."""
        result = TaskResult(task=sample_task)
        data = result.to_dict()

        assert "task" in data
        assert "num_trials" in data
        assert "pass_rate" in data


class TestEvaluationResult:
    """Tests for EvaluationResult class."""

    def test_evaluation_result_creation(self):
        """Test creating an evaluation result."""
        result = EvaluationResult()

        assert len(result.task_results) == 0
        assert result.end_time is None
        assert isinstance(result.start_time.year, int)

    def test_duration(self):
        """Test duration calculation."""
        result = EvaluationResult()
        time.sleep(0.1)
        result.end_time = result.start_time.__class__.now()

        assert result.duration is not None
        assert result.duration >= 0.1

    def test_num_tasks(self, sample_task):
        """Test counting tasks."""
        result = EvaluationResult()
        result.task_results = [
            TaskResult(task=sample_task),
            TaskResult(task=sample_task),
        ]

        assert result.num_tasks == 2

    def test_num_trials(self, sample_task, sample_trial):
        """Test counting total trials."""
        task_result1 = TaskResult(task=sample_task)
        task_result1.trial_results = [TrialResult(trial=sample_trial)] * 3

        task_result2 = TaskResult(task=sample_task)
        task_result2.trial_results = [TrialResult(trial=sample_trial)] * 2

        result = EvaluationResult()
        result.task_results = [task_result1, task_result2]

        assert result.num_trials == 5

    def test_overall_pass_rate(self, sample_task, sample_trial):
        """Test overall pass rate calculation."""
        from agent_evals.scoring import TaskScore

        task_result = TaskResult(task=sample_task)
        task_result.trial_results = [
            TrialResult(trial=sample_trial, task_score=TaskScore(1.0, True)),
            TrialResult(trial=sample_trial, task_score=TaskScore(1.0, True)),
            TrialResult(trial=sample_trial, task_score=TaskScore(0.5, False)),
        ]

        result = EvaluationResult()
        result.task_results = [task_result]

        assert result.overall_pass_rate == pytest.approx(2/3)

    def test_average_score(self, sample_task):
        """Test average score calculation."""
        task_result1 = TaskResult(task=sample_task)
        task_result1.aggregate_score = 0.8

        task_result2 = TaskResult(task=sample_task)
        task_result2.aggregate_score = 0.6

        result = EvaluationResult()
        result.task_results = [task_result1, task_result2]

        assert result.average_score == pytest.approx(0.7)

    def test_evaluation_result_to_dict(self):
        """Test converting evaluation result to dictionary."""
        result = EvaluationResult()
        data = result.to_dict()

        assert "num_tasks" in data
        assert "num_trials" in data
        assert "overall_pass_rate" in data

    def test_summary(self, sample_task):
        """Test generating summary string."""
        result = EvaluationResult()
        task_result = TaskResult(task=sample_task)
        task_result.aggregate_score = 0.9
        task_result.pass_rate = 1.0
        result.task_results = [task_result]

        summary = result.summary()

        assert "EVALUATION SUMMARY" in summary
        assert "Test Task" in summary


class TestEvaluationHarness:
    """Tests for EvaluationHarness class."""

    def test_harness_creation(self, simple_grader):
        """Test creating evaluation harness."""
        harness = EvaluationHarness(
            agent_function=simple_agent,
            graders=[simple_grader],
            scoring_strategy=WeightedScoring(),
            num_trials=3,
        )

        assert harness.agent_function == simple_agent
        assert len(harness.graders) == 1
        assert harness.num_trials == 3

    def test_run_trial_success(self, sample_task, simple_grader):
        """Test running a successful trial."""
        harness = EvaluationHarness(
            agent_function=simple_agent,
            graders=[simple_grader],
            num_trials=1,
        )

        trial_result = harness.run_trial(sample_task, 0)

        assert trial_result.trial.completed is True
        assert trial_result.trial.transcript is not None
        assert trial_result.task_score is not None
        assert trial_result.error is None

    def test_run_trial_with_failing_agent(self, sample_task):
        """Test running trial with failing agent."""
        fail_grader = CodeGrader(
            "Success Check",
            lambda t, o: GraderResult(
                1.0 if (o and o.success) else 0.0,
                o.success if o else False,
                "Checked success"
            )
        )

        harness = EvaluationHarness(
            agent_function=failing_agent,
            graders=[fail_grader],
            num_trials=1,
        )

        trial_result = harness.run_trial(sample_task, 0)

        assert trial_result.trial.completed is True
        assert trial_result.task_score.passed is False

    def test_run_trial_with_error(self, sample_task, simple_grader):
        """Test running trial that raises error."""
        harness = EvaluationHarness(
            agent_function=error_agent,
            graders=[simple_grader],
            num_trials=1,
        )

        trial_result = harness.run_trial(sample_task, 0)

        assert trial_result.error is not None
        assert "failed" in trial_result.error.lower()

    def test_run_trial_with_setup_teardown(self, simple_grader):
        """Test running trial with setup and teardown."""
        setup_called = []
        teardown_called = []

        task = Task(
            task_id="test",
            name="Test",
            description="Test task",
            inputs={},
            success_criteria="Success",
            setup=lambda: setup_called.append(True),
            teardown=lambda: teardown_called.append(True),
        )

        harness = EvaluationHarness(
            agent_function=simple_agent,
            graders=[simple_grader],
            num_trials=1,
        )

        harness.run_trial(task, 0)

        assert len(setup_called) == 1
        assert len(teardown_called) == 1

    def test_run_task_multiple_trials(self, sample_task, simple_grader):
        """Test running task with multiple trials."""
        harness = EvaluationHarness(
            agent_function=simple_agent,
            graders=[simple_grader],
            num_trials=3,
        )

        task_result = harness.run_task(sample_task)

        assert task_result.num_trials == 3
        assert task_result.aggregate_score is not None
        assert task_result.pass_rate is not None

    def test_run_single_task(self, sample_task, simple_grader):
        """Test running evaluation with single task."""
        harness = EvaluationHarness(
            agent_function=simple_agent,
            graders=[simple_grader],
            num_trials=2,
        )

        result = harness.run([sample_task])

        assert result.num_tasks == 1
        assert result.num_trials == 2
        assert result.end_time is not None

    def test_run_multiple_tasks(self, simple_grader):
        """Test running evaluation with multiple tasks."""
        tasks = [
            Task(
                task_id=f"task_{i}",
                name=f"Task {i}",
                description="Test task",
                inputs={},
                success_criteria="Success",
            )
            for i in range(3)
        ]

        harness = EvaluationHarness(
            agent_function=simple_agent,
            graders=[simple_grader],
            num_trials=1,
        )

        result = harness.run(tasks)

        assert result.num_tasks == 3
        assert result.num_trials == 3

    def test_run_parallel(self, simple_grader):
        """Test running evaluation in parallel."""
        tasks = [
            Task(
                task_id=f"task_{i}",
                name=f"Task {i}",
                description="Test task",
                inputs={},
                success_criteria="Success",
            )
            for i in range(3)
        ]

        harness = EvaluationHarness(
            agent_function=simple_agent,
            graders=[simple_grader],
            num_trials=2,
        )

        result = harness.run_parallel(tasks)

        assert result.num_tasks == 3
        assert result.num_trials == 6

    def test_multiple_graders(self, sample_task):
        """Test evaluation with multiple graders."""
        grader1 = CodeGrader(
            "Grader 1",
            lambda t, o: GraderResult(1.0, True, "Pass"),
            weight=1.0,
        )
        grader2 = CodeGrader(
            "Grader 2",
            lambda t, o: GraderResult(0.8, True, "Good"),
            weight=2.0,
        )

        harness = EvaluationHarness(
            agent_function=simple_agent,
            graders=[grader1, grader2],
            scoring_strategy=WeightedScoring(pass_threshold=0.7),
            num_trials=1,
        )

        result = harness.run([sample_task])
        task_result = result.task_results[0]
        trial_result = task_result.trial_results[0]

        assert len(trial_result.task_score.grader_results) == 2
        assert "Grader 1" in trial_result.task_score.grader_results
        assert "Grader 2" in trial_result.task_score.grader_results

    def test_binary_scoring_strategy(self, sample_task):
        """Test evaluation with binary scoring."""
        pass_grader = CodeGrader(
            "Pass",
            lambda t, o: GraderResult(1.0, True, "Pass"),
        )
        fail_grader = CodeGrader(
            "Fail",
            lambda t, o: GraderResult(0.0, False, "Fail"),
        )

        harness = EvaluationHarness(
            agent_function=simple_agent,
            graders=[pass_grader, fail_grader],
            scoring_strategy=BinaryScoring(),
            num_trials=1,
        )

        result = harness.run([sample_task])
        task_result = result.task_results[0]

        # Should fail because one grader failed (binary scoring)
        assert task_result.trial_results[0].task_score.passed is False

    def test_grader_exception_handling(self, sample_task):
        """Test that grader exceptions are caught and handled."""
        def error_grade_fn(transcript, outcome):
            raise ValueError("Grader error")

        error_grader = CodeGrader("Error Grader", error_grade_fn)

        harness = EvaluationHarness(
            agent_function=simple_agent,
            graders=[error_grader],
            num_trials=1,
        )

        result = harness.run([sample_task])
        task_result = result.task_results[0]
        trial_result = task_result.trial_results[0]

        # Should have grader result with error
        assert "Error Grader" in trial_result.task_score.grader_results
        grader_result = trial_result.task_score.grader_results["Error Grader"]
        assert grader_result.score == 0.0
        assert "error" in grader_result.explanation.lower()

    def test_max_workers_parameter(self, simple_grader):
        """Test setting max_workers parameter."""
        harness = EvaluationHarness(
            agent_function=simple_agent,
            graders=[simple_grader],
            num_trials=1,
            max_workers=2,
        )

        assert harness.max_workers == 2

    def test_evaluation_with_realistic_scenario(self):
        """Test a realistic evaluation scenario."""
        # Create tasks of varying difficulty
        tasks = [
            Task(
                task_id="easy",
                name="Easy Task",
                description="Simple task",
                inputs={"value": 10},
                success_criteria="Process value",
                difficulty=TaskDifficulty.EASY,
            ),
            Task(
                task_id="medium",
                name="Medium Task",
                description="Medium task",
                inputs={"value": 20},
                success_criteria="Process value",
                difficulty=TaskDifficulty.MEDIUM,
            ),
        ]

        # Create graders
        success_grader = CodeGrader(
            "Success",
            lambda t, o: GraderResult(
                1.0 if (o and o.success) else 0.0,
                o.success if o else False,
                "Outcome check"
            ),
            weight=2.0,
        )

        efficiency_grader = CodeGrader(
            "Efficiency",
            lambda t, o: GraderResult(
                1.0 if (t and t.num_steps <= 3) else 0.5,
                True,
                f"Steps: {t.num_steps if t else 'N/A'}"
            ),
            weight=1.0,
        )

        # Run evaluation
        harness = EvaluationHarness(
            agent_function=simple_agent,
            graders=[success_grader, efficiency_grader],
            scoring_strategy=WeightedScoring(pass_threshold=0.8),
            num_trials=2,
        )

        result = harness.run(tasks)

        # Verify results
        assert result.num_tasks == 2
        assert result.num_trials == 4
        assert result.overall_pass_rate is not None
        assert result.average_score is not None
        assert result.duration is not None

        # Check summary generation
        summary = result.summary()
        assert "Easy Task" in summary
        assert "Medium Task" in summary
