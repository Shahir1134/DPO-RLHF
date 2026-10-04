# Direct Preference Optimization (DPO)

A hands-on implementation of **Direct Preference Optimization (DPO)** for aligning Large Language Models with preference data.

The goal was not just to implement DPO, but to understand the math behind the objective and connect it to the actual training process.

---

## What is DPO?

Given a prompt $x$, suppose we have two responses:

- $y_w$: the preferred (chosen) response
- $y_l$: the rejected response

DPO directly optimizes the language model to prefer $y_w$ over $y_l$, with no separate reward model and no RL loop.

**PPO-based RLHF**

```text
Preference Data → Reward Model → Reward → PPO → Updated LLM
```

**DPO**

```text
Preference Data → DPO Objective → Updated LLM
```

---

## DPO Objective

$$
\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right]\right)\right]
$$

| Term | Meaning |
|---|---|
| $x$ | Prompt |
| $y_w$ | Chosen (preferred) response |
| $y_l$ | Rejected response |
| $\pi_\theta$ | Trainable policy model |
| $\pi_{ref}$ | Frozen reference model (the "before training" baseline) |
| $\beta$ | Strength of the preference objective (how tightly the policy is tied to the reference) |
| $\sigma$ | Sigmoid function |

---

## Breaking Down the Formula

**Step 1: log-ratio for each response** (how much has the model moved?)

$$
\Delta_w = \log \frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)}
\qquad
\Delta_l = \log \frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}
$$

**Step 2: preference margin**

$$
z = \beta \, (\Delta_w - \Delta_l)
$$

**Step 3: squash into (0, 1) with the sigmoid**

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Step 4: loss**

$$
\mathcal{L}_{DPO} = -\log \sigma(z)
$$

Minimizing this loss pushes up the chosen response and pushes down the rejected one, relative to the reference model.

---

## Numerical Example

Assume:

| | Chosen $y_w$ | Rejected $y_l$ |
|---|---|---|
| $\pi_\theta$ | 0.6 | 0.2 |
| $\pi_{ref}$ | 0.4 | 0.3 |

and $\beta = 0.5$.

$$
\Delta_w = \ln\frac{0.6}{0.4} \approx +0.4055
\qquad
\Delta_l = \ln\frac{0.2}{0.3} \approx -0.4055
$$

$$
z = 0.5\,\bigl(0.4055 - (-0.4055)\bigr) \approx 0.4055
\qquad
\sigma(z) \approx 0.60
$$

$$
\mathcal{L}_{DPO} = -\log(0.6) \approx \mathbf{0.511}
$$

### Sanity check

If the model is identical to the reference ($\pi_\theta = \pi_{ref}$), both log-ratios are $\log 1 = 0$, so $z = 0$, $\sigma(0) = 0.5$, and the loss is $-\log(0.5) = 0.693$. This is the starting loss.

Our example gives **0.511 < 0.693**, so the model has moved toward the chosen response and away from the rejected one.

### Effect of $\beta$ (same margin of 0.811)

| $\beta$ | $\sigma(\beta \cdot 0.811)$ | Loss |
|---|---|---|
| 0.1 | 0.520 | 0.653 |
| 0.5 | 0.600 | 0.511 |
| 1.0 | 0.692 | 0.368 |

---

## How This Maps to an LLM

A response is a sequence of tokens:

$$
y = (y_1, y_2, \ldots, y_T)
$$

Its probability is the product of per-token probabilities:

$$
\pi_\theta(y \mid x) = \prod_{t=1}^{T} \pi_\theta(y_t \mid x, y_{\lt t})
$$

In practice we work with log-probabilities, which turn the product into a sum:

$$
\log \pi_\theta(y \mid x) = \sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{\lt t})
$$

These sequence-level log-probabilities feed into the preference margin and loss above.

---

## Training Flow

```text
Preference Dataset
        ↓
Prompt + Chosen + Rejected
        ↓
Tokenization
        ↓
Policy Model + Reference Model (frozen)
        ↓
Sequence Log-Probabilities
        ↓
Log-Ratios (policy vs reference)
        ↓
Preference Margin
        ↓
DPO Loss
        ↓
Backpropagation
        ↓
Updated Policy
```

---

## Implementation

This project covers:

- Preference-pair training
- Policy and reference models
- Token-level and sequence-level log-probabilities
- Policy/reference log-ratios
- DPO preference margin and loss
- Gradient-based optimization

**Tech stack:** Python, PyTorch, Hugging Face Transformers, TRL, Datasets, Accelerate

---

## Key Idea

> **Make the model prefer what humans preferred.**

More precisely, training pushes the policy to move toward the chosen response more than it moves toward the rejected one:

$$
\frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} \;>\; \frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}
$$

The interesting part of DPO is how this simple preference idea becomes a differentiable objective that directly updates an LLM.

---

## Reference

Rafailov et al., [*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*](https://arxiv.org/abs/2305.18290)
