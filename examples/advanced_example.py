"""
Advanced example demonstrating more sophisticated evaluation scenarios.

This example shows:
1. File management tasks
2. Custom graders
3. Hybrid scoring with required and optional graders
4. Task setup and teardown
5. Error handling
"""

import sys
import os
import tempfile
import shutil

# Add parent directory to path to import agent_evals
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent_evals import (
    Task,
    TaskDifficulty,
    Transcript,
    Outcome,
    Step,
    StepType,
    ToolCall,
    CodeGrader,
    GraderResult,
    HybridScoring,
    EvaluationHarness,
)
import uuid
from pathlib import Path


class FileManagerAgent:
    """Mock agent that performs file operations."""

    def __init__(self, workspace: Path):
        self.workspace = workspace

    def run(self, task: Task) -> Transcript:
        """Execute a file management task."""
        transcript = Transcript(trial_id=f"trial_{uuid.uuid4().hex[:8]}")

        action = task.inputs.get("action")
        filename = task.inputs.get("filename")
        content = task.inputs.get("content", "")

        # Reasoning step
        transcript.add_step(Step(
            step_type=StepType.REASONING,
            content=f"I need to {action} the file '{filename}'",
        ))

        success = True
        error = None
        file_path = self.workspace / filename

        try:
            if action == "create":
                # Create file
                tool_call = ToolCall(
                    tool_name="write_file",
                    arguments={"path": str(file_path), "content": content},
                )
                file_path.write_text(content)
                tool_call.result = f"Created file at {file_path}"

                transcript.add_step(Step(
                    step_type=StepType.TOOL_CALL,
                    content=f"Created file {filename}",
                    tool_call=tool_call,
                ))

            elif action == "read":
                # Read file
                tool_call = ToolCall(
                    tool_name="read_file",
                    arguments={"path": str(file_path)},
                )
                if file_path.exists():
                    file_content = file_path.read_text()
                    tool_call.result = file_content
                else:
                    success = False
                    error = "File not found"
                    tool_call.success = False
                    tool_call.error = error

                transcript.add_step(Step(
                    step_type=StepType.TOOL_CALL,
                    content=f"Read file {filename}",
                    tool_call=tool_call,
                ))

            elif action == "delete":
                # Delete file
                tool_call = ToolCall(
                    tool_name="delete_file",
                    arguments={"path": str(file_path)},
                )
                if file_path.exists():
                    file_path.unlink()
                    tool_call.result = "File deleted"
                else:
                    success = False
                    error = "File not found"
                    tool_call.success = False
                    tool_call.error = error

                transcript.add_step(Step(
                    step_type=StepType.TOOL_CALL,
                    content=f"Deleted file {filename}",
                    tool_call=tool_call,
                ))

        except Exception as e:
            success = False
            error = str(e)

        # Output step
        if success:
            transcript.add_step(Step(
                step_type=StepType.OUTPUT,
                content=f"Successfully completed {action} on {filename}",
            ))
        else:
            transcript.add_step(Step(
                step_type=StepType.OUTPUT,
                content=f"Failed to {action} {filename}: {error}",
            ))

        # Create outcome
        outcome = Outcome(
            success=success,
            final_state={
                "file_exists": file_path.exists(),
                "action": action,
                "filename": filename,
            },
            artifacts={
                "workspace": str(self.workspace),
            },
        )

        if error:
            outcome.metadata["error"] = error

        transcript.complete(outcome)
        return transcript


def create_file_task(action: str, filename: str, content: str = "", workspace: Path = None) -> Task:
    """Create a file management task."""

    def setup():
        """Setup workspace."""
        if workspace:
            workspace.mkdir(parents=True, exist_ok=True)

    def teardown():
        """Clean up workspace."""
        if workspace and workspace.exists():
            shutil.rmtree(workspace)

    return Task(
        task_id=f"file_{action}_{uuid.uuid4().hex[:8]}",
        name=f"File {action.capitalize()}",
        description=f"{action.capitalize()} file '{filename}'",
        inputs={"action": action, "filename": filename, "content": content},
        success_criteria=f"File '{filename}' should be {action}d successfully",
        difficulty=TaskDifficulty.EASY,
        setup=setup,
        teardown=teardown,
        metadata={"workspace": str(workspace)},
    )


# Custom graders for file operations

def file_exists_grader(expected: bool, name: str = "File Exists") -> CodeGrader:
    """Check if file exists after operation."""
    def check(transcript, outcome):
        if outcome is None:
            return GraderResult(
                score=0.0,
                passed=False,
                explanation="No outcome available",
            )

        file_exists = outcome.final_state.get("file_exists", False)
        passed = file_exists == expected

        return GraderResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            explanation=f"File exists: {file_exists} (expected: {expected})",
        )

    return CodeGrader(name, check, required=True)


def efficient_execution_grader(max_steps: int) -> CodeGrader:
    """Check if task was completed efficiently (within step limit)."""
    def check(transcript, outcome):
        if transcript is None:
            return GraderResult(
                score=0.0,
                passed=False,
                explanation="No transcript available",
            )

        num_steps = transcript.num_steps
        if num_steps <= max_steps:
            score = 1.0
            explanation = f"Efficient execution: {num_steps} steps (max: {max_steps})"
        else:
            # Partial credit based on how much over the limit
            score = max(0.0, 1.0 - (num_steps - max_steps) / max_steps)
            explanation = f"Inefficient: {num_steps} steps (max: {max_steps})"

        return GraderResult(
            score=score,
            passed=num_steps <= max_steps,
            explanation=explanation,
            metadata={"num_steps": num_steps, "max_steps": max_steps},
        )

    return CodeGrader("Efficient Execution", check, weight=2.0)


def error_handling_grader() -> CodeGrader:
    """Check if errors were handled gracefully."""
    def check(transcript, outcome):
        if transcript is None:
            return GraderResult(
                score=0.0,
                passed=False,
                explanation="No transcript available",
            )

        # Check if any tool calls failed
        failed_calls = [
            step for step in transcript.steps
            if step.tool_call and not step.tool_call.success
        ]

        if not failed_calls:
            return GraderResult(
                score=1.0,
                passed=True,
                explanation="No errors occurred",
            )

        # Check if errors have explanations
        all_have_errors = all(
            step.tool_call.error is not None
            for step in failed_calls
        )

        score = 0.5 if all_have_errors else 0.0

        return GraderResult(
            score=score,
            passed=all_have_errors,
            explanation=f"{len(failed_calls)} tool calls failed, "
                       f"{'all' if all_have_errors else 'not all'} have error messages",
            metadata={"num_failed_calls": len(failed_calls)},
        )

    return CodeGrader("Error Handling", check, weight=1.0)


def main():
    print("=" * 60)
    print("AI Agent Evaluation Framework - Advanced Example")
    print("=" * 60)
    print()

    # Create temporary workspace
    temp_dir = Path(tempfile.mkdtemp(prefix="agent_eval_"))
    print(f"Using workspace: {temp_dir}\n")

    # Create agent
    agent = FileManagerAgent(workspace=temp_dir)

    # Create tasks
    tasks = [
        create_file_task("create", "test.txt", "Hello, World!", workspace=temp_dir),
        create_file_task("read", "test.txt", workspace=temp_dir),
        create_file_task("delete", "test.txt", workspace=temp_dir),
    ]

    print(f"Created {len(tasks)} tasks:")
    for task in tasks:
        print(f"  - {task.name}: {task.description}")
    print()

    # Create graders with hybrid strategy
    # Required: File operation must succeed
    # Optional: Efficiency and error handling (with weights)
    graders = [
        # Required graders
        file_exists_grader(expected=True, name="File Created Successfully"),

        # Optional graders (weighted)
        efficient_execution_grader(max_steps=5),
        error_handling_grader(),
    ]

    print("Configured graders:")
    for grader in graders:
        req_status = "REQUIRED" if grader.required else f"weight={grader.weight}"
        print(f"  - {grader.name} ({req_status})")
    print()

    print("\n" + "=" * 60)
    print("Running Evaluation with Hybrid Scoring")
    print("(Required graders must pass + weighted optional graders)")
    print("=" * 60)

    # Create harness with hybrid scoring
    harness = EvaluationHarness(
        agent_function=agent.run,
        graders=graders,
        scoring_strategy=HybridScoring(pass_threshold=0.7),
        num_trials=2,
    )

    result = harness.run(tasks)
    print(result.summary())

    # Detailed results for first task
    if result.task_results:
        first_task_result = result.task_results[0]
        print("\n" + "=" * 60)
        print(f"Detailed Results for: {first_task_result.task.name}")
        print("=" * 60)

        for i, trial_result in enumerate(first_task_result.trial_results):
            print(f"\nTrial {i+1}:")
            if trial_result.task_score:
                print(f"  Score: {trial_result.task_score.overall_score:.3f}")
                print(f"  Passed: {trial_result.task_score.passed}")
                print(f"\n  Grader Details:")
                for grader_name, grader_result in trial_result.task_score.grader_results.items():
                    print(f"    {grader_name}: {grader_result.score:.2f} - {grader_result.explanation}")

    # Clean up
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"\n\nCleaned up workspace: {temp_dir}")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
