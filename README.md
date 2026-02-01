# Prompt Optimization: A Comparative Study of Three Algorithms

**By Bhuvanesh (DrixoT) Sunderraj**

## **Table of Contents**
1. [Introduction](#introduction)
2. [Problem Setup](#problem-setup)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Reflection and Insights](#reflection-and-insights)
6. [References](#references)

---

## Introduction

Prompt engineering is a critical skill in getting the best performance out of large language models (LLMs). In this project, I explored three different algorithms for automatically optimizing prompts to improve LLM performance on math word problems from the GSM8K dataset.

The goal was to enhance prompts using various optimization strategies to maximize the model's ability to generate correct responses for multi-step reasoning problems that require numerical precision. I implemented and compared three approaches:

1. **Naive APE (Automatic Prompt Engineer) Optimization** - A straightforward paraphrasing approach
2. **Evolutionary Prompt Optimization** - A tournament-based evolutionary algorithm
3. **Thompson Sampling** - A multi-armed bandit approach using Bayesian inference

This document describes my implementation, results, and insights from this exploration.

---

## Problem Setup

### Task Overview

The task is to optimize prompts that help an LLM model understand math word problems and provide correct answers. The evaluation is done on GSM8K math word problems, which require multi-step reasoning and numerical precision.

### Configuration

I used GPT-3.5-turbo as the model and set a token budget of 50,000 tokens for all experiments to ensure fair comparison. Here's the initial setup:

```python
import os
import json
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Callable
from dataclasses import dataclass
import tiktoken
from scipy import stats
from openai import OpenAI

random.seed(42)
np.random.seed(42)

MODEL_NAME = "gpt-3.5-turbo"
TOKEN_BUDGET = 50000 

client = OpenAI(api_key='YOUR_API_KEY_HERE')  # Replace with your API key

tokenizer = tiktoken.encoding_for_model(MODEL_NAME)
```

### Base Prompt

The starting point for all optimization algorithms was a simple base prompt:

```python
BASE_PROMPT = "Solve the following math problem. Show your work and provide the final numerical answer."
```

### Test Dataset

I used a subset of 10 GSM8K problems for evaluation:

```python
DATASET = [
    {
        "problem": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "answer": 18
    },
    {
        "problem": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "answer": 3
    },
    # ... (8 more problems)
]
```

### Token Tracking

To monitor token usage and stay within budget, I implemented a simple token tracker:

```python
class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
    
    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
    
    def get_total(self) -> int:
        return self.total_tokens
    
    def reset(self):
        self.total_tokens = 0

# Global token tracker
token_tracker = TokenTracker()
```

### LLM Interaction

I created a wrapper function to call the LLM and automatically track token usage:

```python
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def call_llm(prompt: str, system_message: str = None, temperature: float = 0.7) -> str:
    prompt_tokens = count_tokens(prompt)
    if system_message:
        prompt_tokens += count_tokens(system_message)
        
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    response_obj = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature
    )
    response = response_obj.choices[0].message.content
    response_tokens = count_tokens(response)
    
    token_tracker.add_tokens(prompt_tokens + response_tokens)
    
    return response
```

### Evaluation Function

The evaluation function extracts numerical answers from the LLM response and compares them to the expected answer. It uses multiple patterns to handle different response formats:

```python
def extract_answer(text: str) -> float:
    patterns = [
        r'answer is[:\s]+([\d,]+\.?\d*)',
        r'final answer[:\s]+([\d,]+\.?\d*)',
        r'=[\s]*([\d,]+\.?\d*)\s*$',
        r'\$?([\d,]+\.?\d*)\s*(?:dollars?)?\s*$',
    ] 
    text_lower = text.lower().strip()
    
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.MULTILINE)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except:
                pass
                
    numbers = re.findall(r'([\d,]+\.?\d*)', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except:
            pass
    
    return None

def evaluate_response(response: str, expected_answer: float) -> float:
    extracted = extract_answer(response)
    
    if extracted is None:
        return 0.0
    if abs(extracted - expected_answer) < 0.01:
        return 1.0
    if expected_answer != 0:
        relative_error = abs(extracted - expected_answer) / abs(expected_answer)
        if relative_error < 0.05:
            return 0.8
    if str(int(expected_answer)) in response:
        return 0.5
    
    return 0.0

def evaluate_prompt(prompt_instruction: str, test_examples: List[Dict]) -> float:
    scores = []
    for example in test_examples:
        full_prompt = f"{prompt_instruction}\n\nProblem: {example['problem']}"
        response = call_llm(full_prompt)
        score = evaluate_response(response, example['answer'])
        scores.append(score)
    
    return np.mean(scores)
```

The scoring system gives:
- 1.0 for exact matches (within 0.01)
- 0.8 for close matches (within 5% relative error)
- 0.5 if the correct integer appears anywhere in the response
- 0.0 otherwise

---

## Methodology

### Prompt Mutators

All three algorithms use a set of mutators to generate variations of prompts. These mutators use the LLM itself to rewrite prompts in different styles:

```python
MUTATORS = [
    {
        "name": "concise",
        "instruction": "Rewrite the following instruction to be more concise while keeping the same meaning. Return only the rewritten instruction, nothing else.\n\nInstruction: {prompt}"
    },
    {
        "name": "detailed",
        "instruction": "Rewrite the following instruction to be more detailed and explicit. Return only the rewritten instruction, nothing else.\n\nInstruction: {prompt}"
    },
    {
        "name": "step_by_step",
        "instruction": "Rewrite the following instruction to emphasize step-by-step thinking. Return only the rewritten instruction, nothing else.\n\nInstruction: {prompt}"
    },
    {
        "name": "professional",
        "instruction": "Rewrite the following instruction in a more professional and formal tone. Return only the rewritten instruction, nothing else.\n\nInstruction: {prompt}"
    },
    {
        "name": "paraphrase",
        "instruction": "Paraphrase the following instruction using different words but keeping the same meaning. Return only the rewritten instruction, nothing else.\n\nInstruction: {prompt}"
    }
]

def mutate_prompt(prompt: str, mutator: Dict = None) -> str:
    if mutator is None:
        mutator = random.choice(MUTATORS)
    
    mutation_instruction = mutator["instruction"].format(prompt=prompt)
    mutated = call_llm(mutation_instruction, temperature=0.8)
    
    # Clean up the response (remove quotes, extra whitespace)
    mutated = mutated.strip().strip('"').strip("'").strip()
    
    return mutated
```

### Algorithm 1: Naive APE Optimization

The Automatic Prompt Engineer (APE) approach is straightforward: generate multiple paraphrases of the base prompt, evaluate all of them, and select the best one.

**How it works:**
1. Generate N paraphrases of the base prompt (using the paraphrase mutator)
2. Evaluate each prompt on the full test set
3. Return the prompt with the highest score

**Implementation:**

```python
def ape_optimization(base_prompt: str, test_examples: List[Dict], n_paraphrases: int = 10, token_budget: int = 50000) -> Tuple[str, float, List]:
    print(f"\n=== APE Optimization: Generating {n_paraphrases} paraphrases ===")
    print(f"Token budget: {token_budget} tokens")
    
    token_tracker.reset()
    paraphrase_mutator = [m for m in MUTATORS if m["name"] == "paraphrase"][0]
    
    prompts = [base_prompt]  
    print("Generating paraphrases...")
    for i in range(n_paraphrases - 1): 
        if token_tracker.get_total() >= token_budget:
            print(f"Budget reached during paraphrase generation. Generated {len(prompts)} prompts.")
            break
        
        mutated = mutate_prompt(base_prompt, paraphrase_mutator)
        prompts.append(mutated)
        if (i + 1) % 3 == 0:
            print(f"  Generated {i + 1}/{n_paraphrases - 1} paraphrases... (tokens: {token_tracker.get_total()}/{token_budget})")
    
    print(f"\nEvaluating all {len(prompts)} prompts on all test examples...")
    
    results = []
    for i, prompt in enumerate(prompts):
        if token_tracker.get_total() >= token_budget:
            print(f"Budget reached. Evaluated {len(results)}/{len(prompts)} prompts.")
            break
        
        score = evaluate_prompt(prompt, test_examples)
        current_tokens = token_tracker.get_total()
        results.append((prompt, score, current_tokens))
        
        print(f"  Prompt {i+1}/{len(prompts)}: Score = {score:.3f} | Tokens: {current_tokens}/{token_budget}")
    
    if not results:
        raise ValueError("No prompts were evaluated. Check token budget and test examples.")
    
    best_prompt, best_score, final_tokens = max(results, key=lambda x: x[1])
    
    print(f"\n=== APE Complete ===")
  
    return best_prompt, best_score, results
```

**Advantages:**
- Simple and easy to understand
- Guarantees evaluation of all candidates
- No hyperparameter tuning needed

**Disadvantages:**
- No iterative improvement
- Limited exploration of the prompt space
- Can be wasteful with tokens if many paraphrases are generated

### Algorithm 2: Evolutionary Prompt Optimization

The evolutionary approach uses a tournament-based selection mechanism inspired by genetic algorithms. Prompts compete against each other, and winners are mutated to create new generations.

**How it works:**
1. Initialize a population of prompts (including the base prompt and mutations)
2. Repeatedly:
   - Select two random prompts from the population
   - Evaluate both on the test set
   - The winner replaces the loser
   - Mutate the winner and insert it in place of the loser
3. Track the best prompt seen so far
4. Continue until token budget is exhausted

**Implementation:**

```python
def evolutionary(base_prompt: str, test_examples: List[Dict], 
                 population_size: int = 6, token_budget: int = TOKEN_BUDGET) -> Tuple[str, float, List]:
    
    print(f"\n=== Evolutionary Binary Tournament Optimization ===")
    print(f"Population size: {population_size}, Budget: {token_budget} tokens")

    token_tracker.reset()
    
    print("Initializing population...")
    population = [base_prompt]
    
    while len(population) < population_size and token_tracker.get_total() < token_budget:
        mutated = mutate_prompt(base_prompt)
        population.append(mutated)
    
    print(f"Population initialized with {len(population)} prompts")
    
    print("Evaluating base prompt...")
    best_overall_score = evaluate_prompt(base_prompt, test_examples)
    best_overall_prompt = base_prompt
    
    history = [(best_overall_prompt, best_overall_score, token_tracker.get_total())]
    
    round_num = 0
    print("\nStarting tournament rounds...")
    while token_tracker.get_total() < token_budget:
        round_num += 1
        idx1, idx2 = random.sample(range(len(population)), 2)
        prompt1 = population[idx1]
        prompt2 = population[idx2]
        print(f"  Round {round_num}: Evaluating prompts {idx1+1} vs {idx2+1}...")
        score1 = evaluate_prompt(prompt1, test_examples)
        if token_tracker.get_total() >= token_budget:
            break
        
        score2 = evaluate_prompt(prompt2, test_examples)
        
        if score1 >= score2:
            winner_idx, loser_idx = idx1, idx2
            winner_prompt, winner_score = prompt1, score1
        else:
            winner_idx, loser_idx = idx2, idx1
            winner_prompt, winner_score = prompt2, score2
        
        if winner_score > best_overall_score:
            best_overall_score = winner_score
            best_overall_prompt = winner_prompt
       
        history.append((best_overall_prompt, best_overall_score, token_tracker.get_total()))
        
        if token_tracker.get_total() >= token_budget:
            break
        
        mutated_winner = mutate_prompt(winner_prompt)
        population[loser_idx] = mutated_winner
        
        if round_num % 5 == 0:
            print(f"  Round {round_num}: Best score = {best_overall_score:.3f}, Tokens = {token_tracker.get_total()}/{token_budget}")
        
        if token_tracker.get_total() >= token_budget:
            break
    
    print(f"\n=== Evolutionary Tournament Complete ===")
    print(f"Rounds completed: {round_num}")
    print(f"Best score: {best_overall_score:.3f}")
    print(f"Total tokens used: {token_tracker.get_total()}")
    print(f"Best prompt: {best_overall_prompt[:100]}...")
    
    return best_overall_prompt, best_overall_score, history
```

**Advantages:**
- Iterative improvement over time
- Balanced exploration and exploitation
- Good performance with moderate token budgets

**Disadvantages:**
- Requires tuning population size
- Can get stuck in local optima
- Each tournament round uses significant tokens

### Algorithm 3: Thompson Sampling

Thompson Sampling is a Bayesian multi-armed bandit algorithm that balances exploration and exploitation efficiently. Each prompt is treated as an "arm" of a bandit, and we maintain a Beta-Bernoulli distribution to model each prompt's success probability.

**How it works:**
1. Initialize multiple prompt "arms" (variations of the base prompt)
2. For each round:
   - Sample from each arm's posterior distribution
   - Select the arm with the highest sample
   - Evaluate on a single random test example (cheap evaluation)
   - Update the posterior based on success/failure
3. Periodically do full evaluations of promising arms
4. Return the best arm found

**Beta-Bernoulli Model:**

```python
class BetaBernoulli:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha      # Success count (plus prior)
        self.beta = beta        # Failure count (plus prior)
        self.n = 0              # Total observations
    
    def sample(self) -> float:
        # Sample from Beta distribution
        return np.random.beta(self.alpha, self.beta)
    
    def update(self, success: bool):
        self.n += 1
        if success:
            self.alpha += 1
        else:
            self.beta += 1
    
    def get_estimated_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)
    
    def get_std(self) -> float:
        total = self.alpha + self.beta
        if total > 0:
            variance = (self.alpha * self.beta) / (total * total * (total + 1))
            return np.sqrt(variance)
        else:
            return 0.0
```

**Implementation:**

```python
def thompson_sampling_optimization(base_prompt: str, test_examples: List[Dict],
                                  n_arms: int = 8, token_budget: int = TOKEN_BUDGET) -> Tuple[str, float, List]:
    print(f"\n=== Thompson Sampling Multi-Armed Bandit Optimization ===")
    print(f"Number of arms: {n_arms}, Budget: {token_budget} tokens")
    
    token_tracker.reset()
    print("Initializing arms...")
    arms = [base_prompt]
    
    while len(arms) < n_arms and token_tracker.get_total() < token_budget:
        mutated = mutate_prompt(base_prompt)
        arms.append(mutated)
    
    print(f"Initialized {len(arms)} arms (prompts)")
    
    posteriors = [BetaBernoulli() for _ in range(len(arms))]
    
    history = []
    round_num = 0
    
    best_actual_score = 0.0
    best_actual_prompt = base_prompt
    
    FULL_EVAL_INTERVAL = 50
    
    print("\nStarting Thompson sampling rounds...")
    while token_tracker.get_total() < token_budget:
        round_num += 1
        
        # Sample from each posterior
        samples = [posterior.sample() for posterior in posteriors]
        
        # Select arm with highest sample
        selected_arm_idx = np.argmax(samples)
        selected_prompt = arms[selected_arm_idx]
        
        # Evaluate on single random example (cheap)
        random_example = random.choice(test_examples)
        single_example = [random_example]
        score = evaluate_prompt(selected_prompt, single_example)

        # Update posterior
        success = score >= 0.8
        posteriors[selected_arm_idx].update(success)

        # Periodic full evaluation
        if round_num % FULL_EVAL_INTERVAL == 0:
            print(f"  Round {round_num}: Performing full evaluation of top arms...")
            
            estimated_means = [p.get_estimated_mean() for p in posteriors]
            top_indices = np.argsort(estimated_means)[-3:][::-1]
            
            for idx in top_indices:
                if token_tracker.get_total() >= token_budget:
                    break
                
                full_score = evaluate_prompt(arms[idx], test_examples)
        
                if full_score > best_actual_score:
                    best_actual_score = full_score
                    best_actual_prompt = arms[idx]
            
            best_prompt = best_actual_prompt
            best_score = best_actual_score
        else:
            estimated_means = [p.get_estimated_mean() for p in posteriors]
            best_arm_idx = np.argmax(estimated_means)
            best_prompt = arms[best_arm_idx]
            best_score = estimated_means[best_arm_idx]
    
        history.append((best_prompt, best_score, token_tracker.get_total()))
        
        if round_num % 20 == 0:
            eval_type = "actual" if round_num % FULL_EVAL_INTERVAL == 0 else "estimated"
            print(f"  Round {round_num}: Best {eval_type} score = {best_score:.3f}, Tokens = {token_tracker.get_total()}/{token_budget}")
        
        if token_tracker.get_total() >= token_budget:
            break
    
    # Final evaluation of best arm
    if token_tracker.get_total() < token_budget:
        estimated_means = [p.get_estimated_mean() for p in posteriors]
        best_arm_idx = np.argmax(estimated_means)
        final_score = evaluate_prompt(arms[best_arm_idx], test_examples)
        if final_score > best_actual_score:
            best_actual_score = final_score
            best_actual_prompt = arms[best_arm_idx]
    
    print(f"\n=== Thompson Sampling Complete ===")
    print(f"Rounds completed: {round_num}")
    print(f"Best actual score: {best_actual_score:.3f}")
    print(f"Total tokens used: {token_tracker.get_total()}")
    print(f"Best prompt: {best_actual_prompt[:100]}...")
    
    return best_actual_prompt, best_actual_score, history
```

**Advantages:**
- Very token-efficient (single example evaluations are cheap)
- Good exploration-exploitation balance
- Can handle many arms efficiently

**Disadvantages:**
- Requires tuning the full evaluation interval
- Single-example evaluations can be noisy
- Estimated scores may not reflect true performance

---

## Results

I ran all three algorithms with the same token budget of 50,000 tokens. Here are the results:

### Algorithm 1: Naive APE

**Results:**
- **Best Score:** 0.700
- **Total Tokens Used:** 26,506
- **Efficiency:** 0.0264 score per 1k tokens
- **Best Prompt:** "Solve the following math problem. Show your work and provide the final numerical answer." (the base prompt itself!)

**Analysis:**
Interestingly, the base prompt performed best among all paraphrases. This suggests that the paraphrases didn't improve upon the original, or that the evaluation might have some variance. The algorithm is straightforward but doesn't show improvement in this case.

### Algorithm 2: Evolutionary Prompt Optimization

**Results:**
- **Best Score:** 0.730
- **Total Tokens Used:** 50,566
- **Efficiency:** 0.0144 score per 1k tokens
- **Rounds Completed:** 11
- **Best Prompt:** "Think through the steps needed to solve the math problem. Show your work and provide the final numerical answer."

**Analysis:**
The evolutionary approach achieved the highest score (0.730) and showed iterative improvement over time. The tournament mechanism allowed promising prompts to propagate through the population, leading to the best overall performance.

### Algorithm 3: Thompson Sampling

**Results:**
- **Best Score:** 0.650
- **Total Tokens Used:** 50,435
- **Efficiency:** 0.0129 score per 1k tokens
- **Rounds Completed:** 150
- **Best Prompt:** "Follow the step-by-step process to solve the math problem. Display your work and give the final numerical answer."

**Analysis:**
Thompson Sampling completed many more rounds (150 vs 11 for evolutionary) due to its efficient single-example evaluations. However, it achieved a lower final score (0.650), suggesting that the single-example evaluations may have been noisy or that the algorithm converged to a suboptimal solution.

### Comparison Summary

| Algorithm | Best Score | Tokens Used | Efficiency | Rounds |
|-----------|------------|-------------|------------|--------|
| APE Paraphrasing | 0.700 | 26,506 | 0.0264 | 10 |
| Evolutionary Tournament | **0.730** | 50,566 | 0.0144 | 11 |
| Thompson Sampling | 0.650 | 50,435 | 0.0129 | 150 |

**Key Observations:**
1. **Evolutionary Tournament** achieved the highest score (0.730)
2. **APE** was most token-efficient but didn't improve over the base prompt
3. **Thompson Sampling** completed the most rounds but converged to a lower score
4. The best prompts from Evolutionary and Thompson Sampling both emphasized step-by-step thinking

### Visualizations

The results were visualized showing:
- Token usage over time for each algorithm
- Score progression for iterative algorithms
- Comparative performance across all three methods

#### Individual Algorithm Performance

![Individual Algorithms Comparison](individual_algorithms.png)

*This plot shows the performance progression of each algorithm individually. The left plot displays all APE evaluation points, the middle shows the evolutionary tournament's best score over time, and the right shows Thompson Sampling's estimated scores.*

#### Comprehensive Comparison

![Comprehensive Comparison](comparison_plot.png)

*This plot compares all three algorithms side-by-side, showing how their performance evolves with token usage. The Evolutionary Tournament achieves the highest final score, while Thompson Sampling explores more efficiently with many rounds.*

---

## Reflection and Insights

### Trade-offs Between Methods

Each algorithm has distinct characteristics that make it suitable for different scenarios:

#### Algorithm 1: Naive APE Optimization

**Strengths:**
- Simple and easy to understand
- Guarantees evaluation of all candidates
- Most token-efficient approach
- Good for establishing baselines

**Weaknesses:**
- No iterative improvement
- Limited exploration of prompt space
- Doesn't adapt based on results
- May not find optimal prompts if initial paraphrases are poor

**Best for:** Small-scale tests, baseline comparisons, or when you have a very limited token budget.

#### Algorithm 2: Evolutionary Prompt Optimization

**Strengths:**
- Achieved the best overall performance (0.730)
- Iterative improvement over time
- Balanced exploration and exploitation
- Tournament mechanism ensures diversity

**Weaknesses:**
- Requires moderate to high token budgets
- Each round is expensive (full evaluation)
- Population size needs tuning
- Can get stuck in local optima

**Best for:** When you have adequate computational resources and need reliable, high-quality results. This is the method I would recommend for GSM8K math problems.

#### Algorithm 3: Thompson Sampling

**Strengths:**
- Very token-efficient per round
- Can explore many candidates quickly
- Good theoretical guarantees for bandit problems
- Adaptive exploration-exploitation balance

**Weaknesses:**
- Single-example evaluations can be noisy
- Lower final performance in this experiment (0.650)
- Requires tuning of full evaluation interval
- Estimated scores may not reflect true performance

**Best for:** Tight token budgets, large action spaces, or when you need to quickly explore many options.

### Which Method to Use?

For **GSM8K math problems**, I recommend the **Evolutionary Prompt Optimization** algorithm because:

1. It achieved the highest score (0.730)
2. The tournament feature gives each prompt a fair chance to compete
3. It shows consistent improvement over time
4. The balance between exploration and exploitation works well for this problem type

However, the choice depends on your constraints:

- **Tight budget:** Thompson Sampling
- **Moderate budget, need reliability:** Evolutionary Tournament
- **Small test set, simple baseline:** APE

### Future Improvements

Based on this exploration, here are some ideas for future work:

#### Hybrid Approach

Rather than choosing a single method, a hybrid approach could combine the strengths of each:

1. **Phase 1 - Thompson Sampling:** Use Thompson sampling with a diverse initial set of prompts to quickly identify promising candidates while using minimal tokens per evaluation.

2. **Phase 2 - Evolutionary Prompt Optimization:** Take the top-performing prompts from Thompson sampling and use them to seed an evolutionary tournament with full evaluations to refine and improve them further.

3. **Phase 3 - Validation:** Evaluate the final best candidates on the complete test set to get accurate performance estimates.

This would leverage Thompson Sampling's efficiency for exploration and Evolutionary's reliability for refinement.

#### Other Potential Improvements

- **Better mutators:** Develop domain-specific mutators for math problems
- **Ensemble prompts:** Combine multiple good prompts
- **Context-aware mutations:** Use problem characteristics to guide mutations
- **Transfer learning:** Use insights from one problem type to optimize prompts for another
- **Multi-objective optimization:** Optimize for both accuracy and token efficiency

### Lessons Learned

1. **Simplicity isn't always better:** The base prompt outperformed its paraphrases in APE, suggesting that not all variations are improvements.

2. **Iteration matters:** The evolutionary approach showed that iterative improvement can lead to better results than one-shot generation.

3. **Evaluation strategy matters:** Thompson Sampling's noisy single-example evaluations may have led it astray. More robust evaluation strategies could improve its performance.

4. **Token efficiency vs. performance:** There's a trade-off between token efficiency and final performance. Choose based on your constraints.

5. **Problem-specific insights:** All three best prompts emphasized step-by-step thinking, suggesting this is important for math word problems.

---

## References

- **Naive APE:** [Automatic Prompt Engineer Repository](https://github.com/keirp/automatic_prompt_engineer)
- **Evolutionary Prompt Optimization:** [Evolutionary Prompt Optimization Paper](https://arxiv.org/abs/2503.23503)
- **Thompson Sampling:** [Thompson Sampling Repository](https://github.com/PatWalters/TS)

---

## Conclusion

This project explored three different approaches to prompt optimization for math word problems. While the Evolutionary Prompt Optimization algorithm achieved the best results in this experiment, each method has its place depending on your constraints and goals.

The key takeaway is that automatic prompt optimization is feasible and can improve LLM performance, but the choice of algorithm matters. For math problems requiring multi-step reasoning, an iterative, tournament-based approach seems most effective.

I hope this document helps others understand these techniques and apply them to their own prompt optimization challenges. Feel free to use the code and adapt it for your needs!

For the full implementation and results, check out the [Jupyter notebook](Mini%20Project%202%20-%20Prompt%20Optimization.ipynb) in this repository.

