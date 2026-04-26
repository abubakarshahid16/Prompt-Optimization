# Prompt Optimization

A comparative project on automated prompt engineering strategies for improving large language model performance on multi-step math reasoning tasks. The work evaluates several prompt-optimization algorithms under a shared token budget and compares their effectiveness on GSM8K-style problems.

## Overview

This repository explores prompt optimization as an applied research problem rather than relying on manual prompt tuning only. The central idea is to automatically refine prompts so that a language model performs better on math word problems that require multi-step reasoning and numerical accuracy.

The project compares three approaches:

- Naive APE-style optimization
- Evolutionary prompt optimization
- Thompson Sampling

All experiments were evaluated under a controlled token budget to keep the comparison fair.

## Research Goal

The objective is to improve prompt quality for solving GSM8K-style math tasks while tracking:

- answer correctness
- optimization efficiency
- token usage
- relative performance of different search strategies

## Visual Results

| Algorithm Comparison | Individual Algorithm Behavior |
| --- | --- |
| ![Comparison plot](comparison_plot.png) | ![Individual algorithms](individual_algorithms.png) |

## Repository Contents

- `Prompt_Optimization.ipynb`: main notebook with implementation and evaluation
- `comparison_plot.png`: comparative performance visualization
- `individual_algorithms.png`: algorithm-specific visualization
- `requirements.txt`: Python dependencies

## Method Summary

The notebook workflow includes:

1. defining a base prompt for math reasoning
2. evaluating prompts against a subset of GSM8K problems
3. tracking token consumption
4. optimizing prompts using three different strategies
5. comparing final performance and tradeoffs

## Algorithms Compared

### 1. Naive APE-Style Optimization

A simpler paraphrasing-based method that attempts to improve the base prompt through straightforward variation and testing.

### 2. Evolutionary Prompt Optimization

A tournament-style optimization process that treats prompts like evolving candidates and uses selection pressure to keep stronger variants.

### 3. Thompson Sampling

A bandit-inspired strategy that balances exploration and exploitation using Bayesian reasoning when choosing prompt candidates.

## Why This Project Matters

This repo is a strong portfolio project because it demonstrates:

- practical experimentation with LLM prompt engineering
- comparative algorithmic thinking rather than one-off prompting
- token-budget-aware evaluation
- reproducible notebook-based research workflow
- interest in optimizing model behavior systematically

## Running the Project

1. install dependencies
2. open the notebook
3. configure your API key securely
4. run the experiment cells in sequence

### Example setup

```bash
pip install -r requirements.txt
```

Then open:

```text
Prompt_Optimization.ipynb
```

## Important Note

The notebook text in the original project may reference a different author name in some cells or markdown from the original experiment context. This repository is maintained here as part of Abubakar Shahid's portfolio and comparative prompt-engineering work.

## Author

Abubakar Shahid  
GitHub: <https://github.com/abubakarshahid16>
