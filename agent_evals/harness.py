"""
Evaluation harness for running agent evaluations end-to-end.

The harness provides infrastructure that:
- Provides instructions and tools to agents
- Runs tasks concurrently
- Records all steps in transcripts
- Grades outputs using configured graders
- Aggregates results across multiple trials
"""

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import statistics

from .models import Task, Trial, Transcript, Outcome, Step, StepType
from .graders import Grader, GraderResult
from .scoring import ScoringStrategy, TaskScore, WeightedScoring


@dataclass
class TrialResult:
    """
    Result from a single trial.

    Attributes:
        trial: The trial that was run
        task_score: Score for this trial
        error: Error message if trial failed
    """
    trial: Trial
    task_score: Optional[TaskScore] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "trial": self.trial.to_dict(),
            "task_score": self.task_score.to_dict() if self.task_score else None,
            "error": self.error,
        }


@dataclass
class TaskResult:
    """
    Aggregated results from multiple trials of a task.

    Attributes:
        task: The task that was evaluated
        trial_results: Results from individual trials
        aggregate_score: Aggregated score across trials
        pass_rate: Percentage of trials that passed
        metadata: Additional metadata
    """
    task: Task
    trial_results: List[TrialResult] = field(default_factory=list)
    aggregate_score: Optional[float] = None
    pass_rate: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_trials(self) -> int:
        """Get the number of trials run."""
        return len(self.trial_results)

    @property
    def num_passed(self) -> int:
        """Get the number of trials that passed."""
        return sum(
            1 for result in self.trial_results
            if result.task_score and result.task_score.passed
        )

    @property
    def num_failed(self) -> int:
        """Get the number of trials that failed."""
        return sum(
            1 for result in self.trial_results
            if result.task_score and not result.task_score.passed
        )

    @property
    def num_errors(self) -> int:
        """Get the number of trials that errored."""
        return sum(1 for result in self.trial_results if result.error)

    def compute_aggregate_metrics(self) -> None:
        """Compute aggregate metrics across all trials."""
        valid_scores = [
            result.task_score.overall_score
            for result in self.trial_results
            if result.task_score is not None
        ]

        if valid_scores:
            self.aggregate_score = statistics.mean(valid_scores)
            self.metadata["score_std"] = statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0
            self.metadata["score_min"] = min(valid_scores)
            self.metadata["score_max"] = max(valid_scores)

        if self.num_trials > 0:
            self.pass_rate = self.num_passed / self.num_trials

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task": self.task.to_dict(),
            "num_trials": self.num_trials,
            "num_passed": self.num_passed,
            "num_failed": self.num_failed,
            "num_errors": self.num_errors,
            "aggregate_score": self.aggregate_score,
            "pass_rate": self.pass_rate,
            "trial_results": [result.to_dict() for result in self.trial_results],
            "metadata": self.metadata,
        }


@dataclass
class EvaluationResult:
    """
    Complete evaluation results across all tasks.

    Attributes:
        task_results: Results for each task
        start_time: When evaluation started
        end_time: When evaluation ended
        metadata: Additional metadata
    """
    task_results: List[TaskResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Get evaluation duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def num_tasks(self) -> int:
        """Get number of tasks evaluated."""
        return len(self.task_results)

    @property
    def num_trials(self) -> int:
        """Get total number of trials across all tasks."""
        return sum(result.num_trials for result in self.task_results)

    @property
    def overall_pass_rate(self) -> Optional[float]:
        """Get overall pass rate across all tasks."""
        total_trials = sum(result.num_trials for result in self.task_results)
        if total_trials == 0:
            return None
        total_passed = sum(result.num_passed for result in self.task_results)
        return total_passed / total_trials

    @property
    def average_score(self) -> Optional[float]:
        """Get average score across all tasks."""
        valid_scores = [
            result.aggregate_score
            for result in self.task_results
            if result.aggregate_score is not None
        ]
        if not valid_scores:
            return None
        return statistics.mean(valid_scores)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "num_tasks": self.num_tasks,
            "num_trials": self.num_trials,
            "overall_pass_rate": self.overall_pass_rate,
            "average_score": self.average_score,
            "duration": self.duration,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "task_results": [result.to_dict() for result in self.task_results],
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Get a human-readable summary of results."""
        lines = [
            "=" * 60,
            "EVALUATION SUMMARY",
            "=" * 60,
            f"Tasks: {self.num_tasks}",
            f"Total Trials: {self.num_trials}",
            f"Overall Pass Rate: {self.overall_pass_rate:.1%}" if self.overall_pass_rate else "Overall Pass Rate: N/A",
            f"Average Score: {self.average_score:.3f}" if self.average_score else "Average Score: N/A",
            f"Duration: {self.duration:.2f}s" if self.duration else "Duration: N/A",
            "",
            "TASK RESULTS:",
            "-" * 60,
        ]

        for task_result in self.task_results:
            lines.append(f"\n{task_result.task.name} ({task_result.task.task_id})")
            lines.append(f"  Trials: {task_result.num_trials}")
            lines.append(f"  Passed: {task_result.num_passed}/{task_result.num_trials} ({task_result.pass_rate:.1%})" if task_result.pass_rate else "  Pass Rate: N/A")
            lines.append(f"  Score: {task_result.aggregate_score:.3f}" if task_result.aggregate_score else "  Score: N/A")
            if task_result.num_errors > 0:
                lines.append(f"  Errors: {task_result.num_errors}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


class EvaluationHarness:
    """
    Main evaluation harness for running agent evaluations.

    The harness orchestrates the evaluation process:
    1. Sets up the environment for each task
    2. Runs the agent with the task instructions
    3. Records all steps in a transcript
    4. Evaluates the outcome using graders
    5. Aggregates results across multiple trials
    """

    def __init__(
        self,
        agent_function: Callable[[Task], Transcript],
        graders: List[Grader],
        scoring_strategy: Optional[ScoringStrategy] = None,
        num_trials: int = 3,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize the evaluation harness.

        Args:
            agent_function: Function that runs the agent and returns a transcript
            graders: List of graders to evaluate performance
            scoring_strategy: Strategy for combining grader scores
            num_trials: Number of trials to run per task
            max_workers: Maximum number of parallel workers (None = num CPUs)
        """
        self.agent_function = agent_function
        self.graders = graders
        self.scoring_strategy = scoring_strategy or WeightedScoring()
        self.num_trials = num_trials
        self.max_workers = max_workers

    def run_trial(self, task: Task, trial_num: int) -> TrialResult:
        """
        Run a single trial of a task.

        Args:
            task: Task to run
            trial_num: Trial number (for ID generation)

        Returns:
            TrialResult with transcript and scores
        """
        trial_id = f"{task.task_id}_trial_{trial_num}_{uuid.uuid4().hex[:8]}"
        trial = Trial(trial_id=trial_id, task_id=task.task_id)

        try:
            # Setup environment
            task.run_setup()

            # Run agent and get transcript
            transcript = self.agent_function(task)
            trial.transcript = transcript
            trial.completed = True

            # Grade the transcript and outcome
            grader_results = {}
            for grader in self.graders:
                try:
                    result = grader.grade(
                        transcript=transcript,
                        outcome=transcript.outcome if transcript else None,
                    )
                    grader_results[grader.name] = result
                except Exception as e:
                    grader_results[grader.name] = GraderResult(
                        score=0.0,
                        passed=False,
                        explanation=f"Grader error: {str(e)}",
                    )

            # Compute overall score
            task_score = self.scoring_strategy.compute_score(self.graders, grader_results)

            # Teardown environment
            task.run_teardown()

            return TrialResult(trial=trial, task_score=task_score)

        except Exception as e:
            trial.error = str(e)
            return TrialResult(trial=trial, error=str(e))

    def run_task(self, task: Task) -> TaskResult:
        """
        Run multiple trials of a task and aggregate results.

        Args:
            task: Task to evaluate

        Returns:
            TaskResult with aggregated metrics
        """
        task_result = TaskResult(task=task)

        # Run trials (potentially in parallel)
        trial_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.run_trial, task, i)
                for i in range(self.num_trials)
            ]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    trial_results.append(result)
                except Exception as e:
                    # Create error trial result
                    trial_id = f"{task.task_id}_trial_error_{uuid.uuid4().hex[:8]}"
                    trial = Trial(trial_id=trial_id, task_id=task.task_id)
                    trial_results.append(TrialResult(trial=trial, error=str(e)))

        task_result.trial_results = trial_results
        task_result.compute_aggregate_metrics()

        return task_result

    def run(self, tasks: List[Task]) -> EvaluationResult:
        """
        Run evaluation on multiple tasks.

        Args:
            tasks: List of tasks to evaluate

        Returns:
            EvaluationResult with all task results
        """
        result = EvaluationResult()
        result.start_time = datetime.now()

        for task in tasks:
            task_result = self.run_task(task)
            result.task_results.append(task_result)

        result.end_time = datetime.now()
        return result

    def run_parallel(self, tasks: List[Task]) -> EvaluationResult:
        """
        Run evaluation on multiple tasks in parallel.

        Args:
            tasks: List of tasks to evaluate

        Returns:
            EvaluationResult with all task results
        """
        result = EvaluationResult()
        result.start_time = datetime.now()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.run_task, task) for task in tasks]

            for future in as_completed(futures):
                try:
                    task_result = future.result()
                    result.task_results.append(task_result)
                except Exception as e:
                    # Log error but continue with other tasks
                    result.metadata.setdefault("errors", []).append(str(e))

        result.end_time = datetime.now()
        return result
