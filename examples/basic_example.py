"""
Basic example demonstrating the AI agent evaluation framework.

This example shows how to:
1. Define a simple task
2. Create an agent function
3. Set up graders
4. Run evaluations with the harness
"""

import sys
import os

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
    outcome_success_grader,
    final_state_grader,
    WeightedScoring,
    BinaryScoring,
    EvaluationHarness,
)
import uuid
from datetime import datetime


# Example 1: Simple calculator agent
def calculator_agent(task: Task) -> Transcript:
    """
    A simple agent that performs arithmetic operations.

    This is a mock agent for demonstration purposes.
    """
    transcript = Transcript(trial_id=f"trial_{uuid.uuid4().hex[:8]}")

    # Get the operation and operands from task inputs
    operation = task.inputs.get("operation")
    a = task.inputs.get("a")
    b = task.inputs.get("b")

    # Step 1: Reasoning
    transcript.add_step(Step(
        step_type=StepType.REASONING,
        content=f"I need to perform {operation} on {a} and {b}",
    ))

    # Step 2: Tool call to calculator
    result = None
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        result = a / b if b != 0 else None

    tool_call = ToolCall(
        tool_name="calculator",
        arguments={"operation": operation, "a": a, "b": b},
        result=result,
        success=result is not None,
    )

    transcript.add_step(Step(
        step_type=StepType.TOOL_CALL,
        content=f"Calculator({operation}, {a}, {b}) = {result}",
        tool_call=tool_call,
    ))

    # Step 3: Output
    transcript.add_step(Step(
        step_type=StepType.OUTPUT,
        content=f"The result is {result}",
    ))

    # Create outcome
    outcome = Outcome(
        success=result is not None,
        final_state={
            "result": result,
            "operation": operation,
        },
    )

    transcript.complete(outcome)
    return transcript


def create_calculator_task(operation: str, a: float, b: float, expected: float) -> Task:
    """Create a calculator task."""
    return Task(
        task_id=f"calc_{operation}_{uuid.uuid4().hex[:8]}",
        name=f"Calculate {operation}",
        description=f"Perform {operation} on {a} and {b}",
        inputs={"operation": operation, "a": a, "b": b},
        success_criteria=f"Result should be {expected}",
        expected_outcome={"result": expected},
        difficulty=TaskDifficulty.EASY,
    )


def main():
    print("=" * 60)
    print("AI Agent Evaluation Framework - Basic Example")
    print("=" * 60)
    print()

    # Create tasks
    tasks = [
        create_calculator_task("add", 5, 3, 8),
        create_calculator_task("multiply", 4, 7, 28),
        create_calculator_task("subtract", 10, 4, 6),
    ]

    print(f"Created {len(tasks)} tasks:")
    for task in tasks:
        print(f"  - {task.name}: {task.description}")
    print()

    # Create graders
    graders = [
        outcome_success_grader(weight=2.0),  # High weight for success
        final_state_grader("result", None, name="Result Present", weight=1.0),
    ]

    # We need to create custom graders for checking expected results
    def create_result_checker(expected_result):
        def check_result(transcript, outcome):
            if outcome is None:
                return type('obj', (object,), {
                    'score': 0.0,
                    'passed': False,
                    'explanation': "No outcome"
                })()

            actual = outcome.final_state.get("result")
            passed = actual == expected_result

            return type('obj', (object,), {
                'score': 1.0 if passed else 0.0,
                'passed': passed,
                'explanation': f"Result {actual} {'matches' if passed else 'does not match'} expected {expected_result}"
            })()
        return check_result

    print("Configured graders:")
    for grader in graders:
        print(f"  - {grader.name} (weight={grader.weight})")
    print()

    # Example 1: Weighted Scoring
    print("\n" + "=" * 60)
    print("Example 1: Weighted Scoring (threshold=0.8)")
    print("=" * 60)

    harness = EvaluationHarness(
        agent_function=calculator_agent,
        graders=graders,
        scoring_strategy=WeightedScoring(pass_threshold=0.8),
        num_trials=2,  # Run each task twice
    )

    result = harness.run(tasks)
    print(result.summary())

    # Example 2: Binary Scoring
    print("\n" + "=" * 60)
    print("Example 2: Binary Scoring (all graders must pass)")
    print("=" * 60)

    harness_binary = EvaluationHarness(
        agent_function=calculator_agent,
        graders=graders,
        scoring_strategy=BinaryScoring(),
        num_trials=2,
    )

    result_binary = harness_binary.run(tasks)
    print(result_binary.summary())

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
