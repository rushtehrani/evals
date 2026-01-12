"""
Core data models for the evaluation framework.

These models represent the fundamental concepts in agent evaluation:
- Task: A single test with defined inputs and success criteria
- Trial: Each attempt at running a task
- Transcript: Complete record of a trial (trace/trajectory)
- Outcome: Final state in the environment at the end of trial
- Step: Individual action taken during a trial
- ToolCall: Specific tool invocation within a step
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum


class TaskDifficulty(Enum):
    """Difficulty level of a task."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class StepType(Enum):
    """Type of step in a transcript."""
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    OUTPUT = "output"


@dataclass
class ToolCall:
    """
    Represents a tool invocation during agent execution.

    Attributes:
        tool_name: Name of the tool being called
        arguments: Arguments passed to the tool
        result: Result returned by the tool
        timestamp: When the tool was called
        success: Whether the tool call succeeded
        error: Error message if the tool call failed
    """
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "error": self.error,
        }


@dataclass
class Step:
    """
    Represents a single step in the agent's execution.

    A step can be reasoning, a tool call, an observation, or an output.

    Attributes:
        step_type: Type of step
        content: Content of the step (reasoning text, output, etc.)
        tool_call: Tool call if this is a tool call step
        timestamp: When the step occurred
        metadata: Additional metadata about the step
    """
    step_type: StepType
    content: str
    tool_call: Optional[ToolCall] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "step_type": self.step_type.value,
            "content": self.content,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class Outcome:
    """
    Represents the final state in the environment at the end of a trial.

    The outcome is distinct from what the agent says it did. For example,
    an agent might say "Your flight has been booked" (in the transcript),
    but the outcome is whether a reservation actually exists in the database.

    Attributes:
        success: Whether the task was completed successfully
        final_state: The final state of the environment
        artifacts: Any artifacts produced (files, database records, etc.)
        metadata: Additional outcome metadata
    """
    success: bool
    final_state: Dict[str, Any]
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "final_state": self.final_state,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }


@dataclass
class Transcript:
    """
    Complete record of a trial execution.

    Also known as a trace or trajectory. Contains all outputs, tool calls,
    reasoning, intermediate results, and interactions during the trial.

    Attributes:
        trial_id: Unique identifier for this trial
        steps: List of steps taken during execution
        start_time: When the trial started
        end_time: When the trial ended
        outcome: The final outcome of the trial
        metadata: Additional transcript metadata
    """
    trial_id: str
    steps: List[Step] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    outcome: Optional[Outcome] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: Step) -> None:
        """Add a step to the transcript."""
        self.steps.append(step)

    def complete(self, outcome: Outcome) -> None:
        """Mark the transcript as complete with an outcome."""
        self.end_time = datetime.now()
        self.outcome = outcome

    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the trial in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def num_steps(self) -> int:
        """Get the number of steps in the transcript."""
        return len(self.steps)

    @property
    def num_tool_calls(self) -> int:
        """Get the number of tool calls in the transcript."""
        return sum(1 for step in self.steps if step.step_type == StepType.TOOL_CALL)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "trial_id": self.trial_id,
            "steps": [step.to_dict() for step in self.steps],
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "duration": self.duration,
            "num_steps": self.num_steps,
            "num_tool_calls": self.num_tool_calls,
            "metadata": self.metadata,
        }


@dataclass
class Trial:
    """
    Represents a single attempt at executing a task.

    Multiple trials are typically run for each task to produce consistent
    results due to output variation in agent behavior.

    Attributes:
        trial_id: Unique identifier for this trial
        task_id: ID of the task being attempted
        transcript: The transcript of this trial's execution
        completed: Whether the trial has completed
        error: Error message if the trial failed
    """
    trial_id: str
    task_id: str
    transcript: Optional[Transcript] = None
    completed: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "trial_id": self.trial_id,
            "task_id": self.task_id,
            "transcript": self.transcript.to_dict() if self.transcript else None,
            "completed": self.completed,
            "error": self.error,
        }


@dataclass
class Task:
    """
    Represents a single test case with defined inputs and success criteria.

    A task is the fundamental unit of evaluation. It defines what the agent
    should do and what success looks like.

    Attributes:
        task_id: Unique identifier for this task
        name: Human-readable name
        description: Description of what the agent should do
        inputs: Input data for the task
        success_criteria: Criteria for determining success
        difficulty: Difficulty level of the task
        setup: Optional function to set up the environment
        teardown: Optional function to clean up after the task
        expected_outcome: Optional expected outcome for validation
        metadata: Additional task metadata
        tags: Tags for organizing and filtering tasks
    """
    task_id: str
    name: str
    description: str
    inputs: Dict[str, Any]
    success_criteria: str
    difficulty: TaskDifficulty = TaskDifficulty.MEDIUM
    setup: Optional[Callable[[], None]] = None
    teardown: Optional[Callable[[], None]] = None
    expected_outcome: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def run_setup(self) -> None:
        """Run the setup function if provided."""
        if self.setup:
            self.setup()

    def run_teardown(self) -> None:
        """Run the teardown function if provided."""
        if self.teardown:
            self.teardown()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "success_criteria": self.success_criteria,
            "difficulty": self.difficulty.value,
            "expected_outcome": self.expected_outcome,
            "metadata": self.metadata,
            "tags": self.tags,
        }
