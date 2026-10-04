# Direct Preference Optimization (DPO)

A hands-on implementation of **Direct Preference Optimization (DPO)** for aligning Large Language Models using preference data.

The goal of this project was not just to implement DPO, but to understand the mathematics behind the objective and connect it to the actual training process.

---

## What is DPO?

Given a prompt \(x\), suppose we have two responses:

- \(y_w\) → preferred/chosen response
- \(y_l\) → rejected response

DPO directly optimizes the language model to prefer \(y_w\) over \(y_l\).

Unlike the traditional PPO-based RLHF pipeline, DPO does not require an explicit reward-model + PPO optimization loop.

### PPO-based RLHF

```text
Preference Data → Reward Model → Reward → PPO → Updated LLM
```

### DPO

```text
Preference Data → DPO Objective → Updated LLM
```

---

## DPO Objective

The core DPO loss is:

\[
\mathcal{L}_{DPO}
=
-\mathbb{E}
\left[
\log\sigma
\left(
\beta
\left[
\log
\frac{\pi_\theta(y_w|x)}
{\pi_{ref}(y_w|x)}
-
\log
\frac{\pi_\theta(y_l|x)}
{\pi_{ref}(y_l|x)}
\right]
\right)
\right]
\]

### What does each term mean?

| Term | Meaning |
|---|---|
| \(x\) | Prompt |
| \(y_w\) | Chosen/preferred response |
| \(y_l\) | Rejected response |
| \(\pi_\theta\) | Trainable policy model |
| \(\pi_{ref}\) | Frozen reference model |
| \(\beta\) | Controls the strength of the preference objective |
| \(\sigma\) | Sigmoid function |
| \(\mathcal{L}_{DPO}\) | DPO loss |

The reference model acts as a baseline, while the policy model is updated.

---

## Breaking Down the Formula

For the chosen response:

\[
\Delta_w =
\log
\frac{\pi_\theta(y_w|x)}
{\pi_{ref}(y_w|x)}
\]

For the rejected response:

\[
\Delta_l =
\log
\frac{\pi_\theta(y_l|x)}
{\pi_{ref}(y_l|x)}
\]

DPO compares these two values:

\[
z = \beta(\Delta_w-\Delta_l)
\]

This creates the **preference margin**.

The sigmoid converts the margin into a value between 0 and 1:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

Finally:

\[
\boxed{
L_{DPO}=-\log\sigma(z)
}
\]

Minimizing this loss encourages the policy to favor the chosen response over the rejected response.

---

## Numerical Example

Assume:

\[
\pi_\theta(y_w|x)=0.6
\]

\[
\pi_{ref}(y_w|x)=0.4
\]

\[
\pi_\theta(y_l|x)=0.2
\]

\[
\pi_{ref}(y_l|x)=0.3
\]

and:

\[
\beta=0.5
\]

Chosen response:

\[
\ln\left(\frac{0.6}{0.4}\right)
\approx0.4055
\]

Rejected response:

\[
\ln\left(\frac{0.2}{0.3}\right)
\approx-0.4055
\]

Therefore:

\[
z=0.5(0.4055-(-0.4055))
\approx0.4055
\]

\[
\sigma(z)\approx0.60
\]

\[
\boxed{L_{DPO}\approx0.511}
\]

---

## How This Maps to an LLM

A response consists of multiple tokens:

\[
y=(y_1,y_2,\ldots,y_T)
\]

Its probability is:

\[
\pi_\theta(y|x)
=
\prod_{t=1}^{T}
\pi_\theta(y_t|x,y_{<t})
\]

In practice, we work with log probabilities:

\[
\log\pi_\theta(y|x)
=
\sum_{t=1}^{T}
\log\pi_\theta(y_t|x,y_{<t})
\]

These sequence-level log probabilities are then used to calculate the DPO preference margin and loss.

---

## Training Flow

```text
Preference Dataset
        ↓
Prompt + Chosen + Rejected
        ↓
Tokenization
        ↓
Policy Model + Reference Model
        ↓
Sequence Log Probabilities
        ↓
Relative Log Probabilities
        ↓
Preference Margin
        ↓
DPO Loss
        ↓
Backpropagation
        ↓
Updated Policy
```

The **reference model remains frozen** while the policy model is trained.

---

## Implementation

This project focuses on implementing and understanding:

- Preference-pair training
- Policy and reference models
- Token-level log probabilities
- Sequence-level log probabilities
- Policy/reference probability ratios
- DPO preference margin
- DPO loss
- Gradient-based optimization

### Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face TRL
- Hugging Face Datasets
- Accelerate

---

## Key Idea

DPO can be summarized as:

\[
\boxed{
\text{Make the model prefer what humans preferred}
}
\]

More precisely:

\[
\boxed{
\frac{\pi_\theta(y_w|x)}
{\pi_{ref}(y_w|x)}
>
\frac{\pi_\theta(y_l|x)}
{\pi_{ref}(y_l|x)}
}
\]

The interesting part of DPO is how this simple preference idea is turned into a differentiable objective that can directly update an LLM.

---

## Reference

Rafailov et al.,  
**Direct Preference Optimization: Your Language Model is Secretly a Reward Model**

[Paper](https://arxiv.org/abs/2305.18290)
