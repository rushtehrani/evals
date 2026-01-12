# AI Agent Evaluation Framework

A comprehensive Python framework for evaluating AI agents, based on Anthropic's approach described in ["Demystifying evals for AI agents"](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).

## Overview

This framework provides a structured way to evaluate AI agents through:

- **Task Definitions**: Define test cases with inputs and success criteria
- **Trial Execution**: Run agents multiple times to account for output variation
- **Transcript Recording**: Capture complete execution traces (tool calls, reasoning, outputs)
- **Outcome Validation**: Verify actual environmental changes, not just agent claims
- **Multi-type Grading**: Code-based, model-based, and human evaluation
- **Flexible Scoring**: Weighted, binary, or hybrid scoring strategies
- **Evaluation Harness**: Infrastructure to run evaluations concurrently and aggregate results

## Key Concepts

### Task
A single test with defined inputs and success criteria. Tasks define what the agent should do and what success looks like.

### Trial
Each attempt at executing a task. Multiple trials are run to produce consistent results due to output variation.

### Transcript (Trace/Trajectory)
The complete record of a trial including:
- Outputs
- Tool calls
- Reasoning steps
- Intermediate results
- All interactions

### Outcome
The final state in the environment at the end of a trial. This is distinct from what the agent says it did. For example, an agent might say "Your flight has been booked" (transcript), but the outcome is whether a reservation actually exists in the database.

### Graders
Logic that scores aspects of agent performance:
- **CodeGrader**: Deterministic, rule-based evaluation (fast, reliable)
- **ModelGrader**: LLM-based evaluation (captures nuance, slower)
- **HumanGrader**: Requires human judgment (for subjective criteria)

### Scoring Strategies

#### Weighted Scoring
Combines grader scores using weights. Task passes if weighted average meets threshold.

```
score = sum(grader.weight × result.score) / sum(grader.weight)
```

#### Binary Scoring
All graders must pass for the task to pass. No partial credit.

#### Hybrid Scoring
Combines required graders (must all pass) with weighted optional graders. Useful for strict requirements with flexible quality criteria.

### Evaluation Harness
Infrastructure that runs evaluations end-to-end:
- Provides instructions and tools
- Runs tasks concurrently
- Records all steps
- Grades outputs
- Aggregates results

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd evals

# No external dependencies required for basic usage
# For model-based graders, install:
pip install anthropic  # or your preferred LLM client
```

## Quick Start

```python
from agent_evals import (
    Task, TaskDifficulty, Transcript, Outcome,
    CodeGrader, outcome_success_grader,
    WeightedScoring, EvaluationHarness
)

# 1. Define your agent function
def my_agent(task: Task) -> Transcript:
    transcript = Transcript(trial_id="trial_1")
    # ... execute agent logic ...
    outcome = Outcome(success=True, final_state={"result": 42})
    transcript.complete(outcome)
    return transcript

# 2. Create tasks
task = Task(
    task_id="task_1",
    name="Example Task",
    description="Solve a problem",
    inputs={"problem": "2+2"},
    success_criteria="Result should be 4",
    difficulty=TaskDifficulty.EASY
)

# 3. Configure graders
graders = [
    outcome_success_grader(weight=2.0),
    # Add custom graders...
]

# 4. Run evaluation
harness = EvaluationHarness(
    agent_function=my_agent,
    graders=graders,
    scoring_strategy=WeightedScoring(pass_threshold=0.8),
    num_trials=3
)

result = harness.run([task])
print(result.summary())
```

## Examples

### Basic Example

See `examples/basic_example.py` for a simple calculator agent:

```bash
python examples/basic_example.py
```

This demonstrates:
- Simple task creation
- Basic graders
- Weighted and binary scoring

### Advanced Example

See `examples/advanced_example.py` for a file management agent:

```bash
python examples/advanced_example.py
```

This demonstrates:
- Task setup and teardown
- Custom graders
- Hybrid scoring
- Error handling evaluation

## Creating Custom Graders

### Code-based Grader

```python
from agent_evals import CodeGrader, GraderResult

def my_custom_grader():
    def grade_fn(transcript, outcome):
        # Your grading logic
        score = 1.0 if some_condition else 0.0
        return GraderResult(
            score=score,
            passed=score >= 0.5,
            explanation="Description of why this score was given"
        )

    return CodeGrader(
        name="My Custom Grader",
        grading_function=grade_fn,
        weight=1.0
    )
```

### Model-based Grader

```python
from agent_evals import ModelGrader

grader = ModelGrader(
    name="Quality Assessment",
    prompt_template="Evaluate the following agent behavior:\n{context}\n\nScore from 0-1:",
    model_client=your_llm_client,
    pass_threshold=0.7
)
```

### Human Grader

```python
from agent_evals import HumanGrader

def review_callback(transcript, outcome):
    # Present to human reviewer (UI, CLI, etc.)
    score = float(input("Enter score (0-1): "))
    return GraderResult(score=score, passed=score >= 0.7)

grader = HumanGrader(
    name="User Experience Review",
    instructions="Evaluate the user experience...",
    review_callback=review_callback
)
```

## Scoring Strategies

### Weighted Scoring

```python
from agent_evals import WeightedScoring

scoring = WeightedScoring(pass_threshold=0.8)
```

Good for: Tasks where some criteria are more important than others.

### Binary Scoring

```python
from agent_evals import BinaryScoring

scoring = BinaryScoring()
```

Good for: Tasks with strict requirements where partial credit doesn't make sense.

### Hybrid Scoring

```python
from agent_evals import HybridScoring

# Mark some graders as required
graders = [
    CodeGrader(..., required=True),   # Must pass
    CodeGrader(..., weight=2.0),       # Optional, weighted
    CodeGrader(..., weight=1.0),       # Optional, weighted
]

scoring = HybridScoring(pass_threshold=0.7)
```

Good for: Tasks with both strict requirements and flexible quality criteria.

## API Reference

### Core Models

- `Task`: Test case definition
- `Trial`: Single execution attempt
- `Transcript`: Complete execution record
- `Outcome`: Final environmental state
- `Step`: Individual action in transcript
- `ToolCall`: Tool invocation details

### Graders

- `Grader`: Abstract base class
- `CodeGrader`: Deterministic grading
- `ModelGrader`: LLM-based grading
- `HumanGrader`: Human judgment
- `GraderResult`: Grading result

### Scoring

- `ScoringStrategy`: Abstract base class
- `WeightedScoring`: Weighted average
- `BinaryScoring`: All must pass
- `HybridScoring`: Required + weighted
- `TaskScore`: Final task score

### Harness

- `EvaluationHarness`: Main orchestrator
- `TrialResult`: Single trial result
- `TaskResult`: Aggregated task results
- `EvaluationResult`: Complete evaluation results

## Best Practices

### Task Design

1. **Start Small**: Begin with 20-50 simple tasks from real failures
2. **Focus on Outcomes**: Grade what actually happened, not what the agent said
3. **Include Partial Credit**: For complex tasks, reward progress

### Grader Selection

1. **Prefer Code Graders**: Fast, reliable, easy to debug
2. **Use Model Graders for Nuance**: Quality, tone, complex reasoning
3. **Reserve Human Graders**: Subjective criteria, edge cases

### Running Evaluations

1. **Multiple Trials**: Run 3-5 trials per task to account for variation
2. **Parallel Execution**: Use `run_parallel()` for large task sets
3. **Monitor Metrics**: Track pass rates, score distributions, execution time

### Iteration

1. **Analyze Failures**: Review failed trials to understand agent weaknesses
2. **Refine Tasks**: Update tasks based on what you learn
3. **Adjust Graders**: Fine-tune weights and thresholds
4. **Track Progress**: Compare results across agent versions

## Architecture

```
agent_evals/
├── __init__.py          # Package exports
├── models.py            # Core data models
├── graders.py           # Grader implementations
├── scoring.py           # Scoring strategies
└── harness.py           # Evaluation harness

examples/
├── basic_example.py     # Simple calculator agent
└── advanced_example.py  # File management agent
```

## References

This framework is based on concepts from:

- [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) - Anthropic Engineering Blog
- [Building Effective AI Agents](https://www.anthropic.com/research/building-effective-agents) - Anthropic Research

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
