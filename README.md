# LLM-PSO

This repository provides the source code and supplementary materials for the LLM-guided particle swarm optimization method.

## Contents

- `LLM-PSO-2020.py`: LLM-PSO implementation for CEC2020 benchmark functions.
- `LLM-PSO-2022.py`: LLM-PSO implementation for CEC2022 benchmark functions.
- `LLM-PSO-Engineer.py`: LLM-PSO implementation for engineering optimization problems.
- `prompt_templates/llmpso_prompt_template.md`: representative prompt template used for LLM-based operator scheduling.

## Experimental Settings

For the CEC2020 experiments reported in the manuscript, the main settings are:

- Dimension: 50
- Population size: 100
- Maximum iterations per run: 500
- Maximum function evaluations: `1000 * dimension`
- Number of independent runs: 30
- LLM scheduling interval: `num = 5`
- Inference temperature: `0.1`

Here, "30 independent runs" denotes 30 repeated trials with different random seeds. It is different from the optimization iteration number, which is controlled by `MaxIter`.

## LLM Configuration

The LLM is used only for selecting a sequence of operator indices from a predefined operator pool. It is not allowed to generate new operators or modify the PSO update framework.

Before running the code, configure the API key locally. For example:

```python
client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://api.deepseek.com"
)
