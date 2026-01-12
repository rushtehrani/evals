# Examples

This directory contains example implementations demonstrating the AI Agent Evaluation Framework.

## Running the Examples

All examples are standalone Python scripts that can be run directly:

```bash
python examples/basic_example.py
python examples/advanced_example.py
```

## Example Overview

### 1. Basic Example (`basic_example.py`)

**What it demonstrates:**
- Simple agent implementation (calculator)
- Task creation with inputs and success criteria
- Basic graders (outcome success, final state checks)
- Weighted vs. Binary scoring strategies
- Running evaluations with multiple trials

**Use this example to:**
- Understand the core framework concepts
- Learn basic task and grader setup
- See how scoring strategies differ

**Agent:** Calculator agent that performs arithmetic operations

### 2. Advanced Example (`advanced_example.py`)

**What it demonstrates:**
- Complex agent with state (file manager)
- Task setup and teardown functions
- Custom graders with partial credit
- Hybrid scoring (required + optional graders)
- Error handling evaluation
- Working with real file system state

**Use this example to:**
- Learn advanced grader patterns
- Understand hybrid scoring
- See task lifecycle management
- Handle real environmental state

**Agent:** File manager agent that creates, reads, and deletes files

## Creating Your Own Examples

When creating evaluation examples for your own agents:

1. **Define Clear Tasks**: Each task should have specific inputs and success criteria
2. **Mock or Integrate**: Either mock your agent or integrate with a real implementation
3. **Choose Appropriate Graders**: Mix code, model, and human graders based on needs
4. **Select Scoring Strategy**: Match your strategy to your task requirements
5. **Run Multiple Trials**: Account for agent output variation

## Example Template

```python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent_evals import (
    Task, Transcript, Outcome, Step, StepType,
    CodeGrader, GraderResult,
    WeightedScoring, EvaluationHarness
)

# 1. Implement your agent
def my_agent(task: Task) -> Transcript:
    transcript = Transcript(trial_id="trial_1")
    # ... your agent logic ...
    outcome = Outcome(success=True, final_state={})
    transcript.complete(outcome)
    return transcript

# 2. Create tasks
tasks = [
    Task(
        task_id="task_1",
        name="My Task",
        description="Do something",
        inputs={"key": "value"},
        success_criteria="It should work"
    )
]

# 3. Create graders
def my_grader_fn(transcript, outcome):
    return GraderResult(score=1.0, passed=True, explanation="Good!")

graders = [CodeGrader("My Grader", my_grader_fn)]

# 4. Run evaluation
harness = EvaluationHarness(
    agent_function=my_agent,
    graders=graders,
    scoring_strategy=WeightedScoring(),
    num_trials=3
)

result = harness.run(tasks)
print(result.summary())
```

## Common Patterns

### Pattern 1: Checking Tool Usage

```python
def tool_usage_grader(required_tool: str):
    def check(transcript, outcome):
        if transcript is None:
            return GraderResult(0.0, False, "No transcript")

        used = any(
            step.tool_call and step.tool_call.tool_name == required_tool
            for step in transcript.steps
        )

        return GraderResult(
            score=1.0 if used else 0.0,
            passed=used,
            explanation=f"{'Used' if used else 'Did not use'} {required_tool}"
        )

    return CodeGrader(f"Uses {required_tool}", check)
```

### Pattern 2: Validating Output Format

```python
def output_format_grader(expected_format: str):
    def check(transcript, outcome):
        # Check if output matches expected format
        # (e.g., JSON, specific structure, etc.)
        return GraderResult(1.0, True, "Format valid")

    return CodeGrader("Output Format", check)
```

### Pattern 3: Performance Metrics

```python
def performance_grader(max_duration: float):
    def check(transcript, outcome):
        if transcript is None or transcript.duration is None:
            return GraderResult(0.0, False, "No timing data")

        duration = transcript.duration
        if duration <= max_duration:
            score = 1.0
        else:
            # Partial credit based on how much over
            score = max(0.0, 1.0 - (duration - max_duration) / max_duration)

        return GraderResult(
            score=score,
            passed=duration <= max_duration,
            explanation=f"Completed in {duration:.2f}s (max: {max_duration}s)"
        )

    return CodeGrader("Performance", check)
```

## Tips

1. **Start Simple**: Begin with basic evaluations before adding complexity
2. **Use Real Failures**: Create tasks based on actual agent failures you've observed
3. **Iterate Quickly**: Run small eval sets frequently during development
4. **Track Over Time**: Save evaluation results to track improvement
5. **Balance Coverage**: Mix easy, medium, and hard tasks

## Need Help?

Check the main README.md for:
- API documentation
- Best practices
- Architecture overview
- Contributing guidelines
