# SentiAgent

**An Agentic LLM Framework for Context-Aware and Explainable Sentiment Analysis**

This repository contains the reference implementation of SentiAgent, a
five-agent LLM pipeline that performs context-aware, knowledge-grounded,
and explainable sentiment classification on free-text reviews.

> This repository accompanies the manuscript *"SentiAgent: An Agentic LLM
> Framework for Context-Aware and Explainable Sentiment Analysis"* and is
> provided to satisfy code-availability requirements. Please cite the
> paper if you use this code (see [Citation](#citation)).

## Architecture

Each input review `X` is passed sequentially through five specialized
agents, each implemented as a constrained LLM call with a structured JSON
input/output contract:

| # | Agent | Role | Output |
|---|-------|------|--------|
| 1 | **Context Agent** (`C_A`) | Identify entities, aspects, negation, contrast, and evidence | `O_C` |
| 2 | **Knowledge Agent** (`K_A`) | Enrich aspects with external commonsense knowledge (ConceptNet / COMET) | `O_K` |
| 3 | **Sentiment Agent** (`S_A`) | Infer aspect-level and initial overall sentiment with confidence | `O_S` |
| 4 | **Reasoning Agent** (`R_A`) | Aggregate evidence, detect contradictions, finalize the decision | `O_R` |
| 5 | **Explainability Agent** (`E_A`) | Produce a faithful, human-readable explanation of the final decision | `O_E` |

```
Input Review (X) -> C_A -> K_A -> S_A -> R_A -> E_A -> Output
                                                        (final sentiment,
                                                         confidence,
                                                         explanation)
```

Allowed sentiment classes: `Positive`, `Negative`, `Neutral`, `Mixed`.

## Repository structure

```
sentiagent/
├── sentiagent/
│   ├── __init__.py          # Public API
│   ├── agents.py            # ContextAgent, KnowledgeAgent, SentimentAgent,
│   │                         # ReasoningAgent, ExplainabilityAgent
│   ├── llm_client.py        # Provider-agnostic LLM wrapper (Anthropic / OpenAI)
│   ├── knowledge_sources.py # ConceptNet retriever + COMET stub
│   ├── pipeline.py          # SentiAgentPipeline orchestrator
│   └── cli.py                # Command-line interface
├── examples/
│   └── run_example.py        # Reproduces the graphical-abstract worked example
├── tests/
│   └── test_pipeline.py      # Unit tests with a mocked LLM backend
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/<org>/sentiagent.git
cd sentiagent
pip install -r requirements.txt
pip install -e .
```

## Configuration

SentiAgent calls an LLM provider for each agent. Set the relevant API key
as an environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # default backend
# or
export OPENAI_API_KEY=sk-...          # if using backend="openai"
```

## Usage

### Python API

```python
from sentiagent import SentiAgentPipeline

pipeline = SentiAgentPipeline(model="claude-sonnet-4-6", verbose=True)
result = pipeline.run("The phone camera is excellent, but the battery drains quickly.")

print(result.final_label)      # "Mixed"
print(result.confidence)       # 0.96
print(result.explanation)      # human-readable rationale
print(result.to_dict())        # full trace: every agent's structured output
```

### Command line

```bash
python -m sentiagent.cli --text "The phone camera is excellent, but the battery drains quickly."

# Batch mode
python -m sentiagent.cli --file reviews.txt --output results.json
```

### Worked example

```bash
python examples/run_example.py
```

reproduces the pipeline trace shown in the paper's graphical abstract:

```
Context Agent   -> aspects: camera (positive), battery (negative)
Knowledge Agent -> "Poor battery life decreases user satisfaction."
Sentiment Agent -> Mixed Sentiment, confidence 0.93
Reasoning Agent -> Mixed Sentiment, confidence 0.96
Explainability  -> "The review appreciates camera quality but expresses
                     dissatisfaction with battery performance... Mixed."
```

## Extending

- **Swap the knowledge source:** implement a class with a
  `retrieve(aspects) -> Dict[str, List[Dict[str, str]]]` method (see
  `CometKnowledgeRetriever` stub in `knowledge_sources.py`) and pass it to
  `SentiAgentPipeline(knowledge_retriever=...)`.
- **Swap the LLM backend:** add a `_call_<backend>` method to `LLMClient`
  in `llm_client.py` following the existing Anthropic/OpenAI pattern.
- **Batch evaluation on a labeled dataset:** use
  `SentiAgentPipeline.run_batch(list_of_reviews)` and compare
  `result.final_label` against ground truth to reproduce accuracy metrics.

## Testing

```bash
pip install pytest
pytest tests/
```

Tests use a mocked LLM client so they run offline and do not require API
credentials.

## Citation

If you use this code, please cite:

```bibtex
@article{sentiagent2026,
  title   = {SentiAgent: An Agentic LLM Framework for Context-Aware and Explainable Sentiment Analysis},
  author  = {<Authors>},
  journal = {<Venue>},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
