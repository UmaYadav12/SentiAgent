"""
run_example.py
---------------
Reproduces the worked example from the paper's graphical abstract:

    Input:  "The phone camera is excellent, but the battery drains quickly."
    Expected trace:
        Context Agent   -> aspects: camera (positive), battery (negative)
        Knowledge Agent -> "Poor battery life decreases user satisfaction."
        Sentiment Agent -> Mixed Sentiment, confidence 0.93
        Reasoning Agent -> Mixed Sentiment, confidence 0.96
        Explainability  -> natural-language explanation of the Mixed verdict

Run with:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/run_example.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentiagent import SentiAgentPipeline  # noqa: E402


def main() -> None:
    pipeline = SentiAgentPipeline(model="claude-sonnet-4-6", verbose=True)

    review = "The phone camera is excellent, but the battery drains quickly."
    result = pipeline.run(review)

    print("\n=== Final Output ===")
    print(json.dumps(result.to_dict()["output"], indent=2))

    print("\n=== Full Agent Trace ===")
    print(json.dumps(result.to_dict()["trace"], indent=2))


if __name__ == "__main__":
    main()
