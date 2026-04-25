# Safe Delta: Consistently Preserving Safety when Fine-Tuning LLMs

## 1. Overview

Safe Delta is a **post-training defense method** designed to preserve the safety of Large Language Models (LLMs) during fine-tuning, while maintaining their task-specific utility.

### Motivation
- Fine-tuning improves task performance (utility)
- However, it often **degrades safety alignment**
- Existing methods struggle with:
  - Diverse datasets (size, type, quality)
  - Trade-off between safety and utility

### Key Idea
Instead of directly applying all fine-tuned weight updates, Safe Delta:
1. **Selects beneficial weight changes**
2. **Suppresses harmful ones**
3. **Compensates for safety loss**

---

## 2. Problem Formulation

Let:
- \( W_{orig} \): original aligned model
- \( W_{sft} \): fine-tuned model
- \( \Delta W = W_{sft} - W_{orig} \)

Goal:
\[
W_{sd} = W_{orig} + F(\Delta W)
\]

Where:
- \( W_{sd} \): final safe model
- \( F(\cdot) \): Safe Delta transformation

### Optimization Objective
- Maximize utility (task performance)
- Constrain safety degradation

\[
\min \; L_{task}(W_{sd}) - L_{task}(W_{sft})
\]
\[
\text{s.t. } L_{safe}(W_{sd}) - L_{safe}(W_{orig}) \leq \epsilon
\]

---

## 3. Method Overview

Safe Delta operates in two main steps:

### Step 1: Delta Parameter Selection
### Step 2: Safety Compensation

Final formulation:
\[
W_{sd} = W_{orig} + M \odot \Delta W + C
\]

- \( M \): binary mask (selected parameters)
- \( C \): safety compensation vector

---

## 4. Step 1: Delta Parameter Selection

### Key Insight
Each parameter update contributes differently to:
- Utility improvement
- Safety degradation

### Metrics

#### (1) Safety Loss
\[
L_{safe} = \| W_{sd} X - W_{orig} X \|^2
\]

#### (2) Utility Approximation
\[
L_{util} = \| W_{sd} - W_{orig} \|^2
\]

---

### Per-Parameter Analysis

For each parameter \( \delta w_m \):

- Utility gain:
\[
\delta L_{util}^m = -(\delta w_m)^2
\]

- Safety loss:
\[
\delta L_{safe}^m = \frac{(\delta w_m)^2}{2[H^{-1}]_{mm}}
\]

---

### Utility-Safety Ratio

\[
r_m = \frac{-\delta L_{util}^m}{\delta L_{safe}^m} = 2[H^{-1}]_{mm}
\]

### Selection Strategy
- Sort parameters by \( r_m \) (descending)
- Greedily select parameters
- Stop when total safety loss exceeds threshold \( \epsilon \)

---

## 5. Step 2: Safety Compensation

### Problem
Even selected parameters introduce some safety degradation

### Solution
Add a compensation vector \( C \)

---

### Compensation Vector

For each selected parameter:

\[
C_m = \frac{\delta w_m}{[H^{-1}]_{mm}} \cdot H^{-1}_{:,m}
\]

Final compensation:

\[
C = (I - M) \odot \sum_{m \in S} C_m
\]

- Only applied to **unselected weights**
- Preserves utility while correcting safety

---

## 6. Algorithm Pipeline

### Preprocessing (One-time)
1. Compute safety Hessian:
\[
H = \nabla^2 L_{safe}
\]
2. Compute and store \( H^{-1} \)

---

### Per Fine-Tuning Request

1. Perform standard fine-tuning → obtain \( \Delta W \)
2. Compute:
   - safety loss per parameter
   - utility-safety ratio
3. Select parameters (mask \( M \))
4. Compute compensation vector \( C \)
5. Construct final model:
\[
W_{sd} = W_{orig} + M \odot \Delta W + C
\]

---

## 7. Key Characteristics

### 1. Adaptive to Dataset
- Dynamically adjusts based on safety degradation
- Works across diverse datasets

### 2. Parameter-Level Control
- Fine-grained selection of weight updates
- More precise than global projection methods

### 3. Efficient
- Hessian inverse computed once
- Low overhead per fine-tuning

### 4. Balanced Trade-off
- Maintains utility from benign data
- Prevents safety degradation from harmful data

---

## 8. Comparison to Existing Methods

| Method        | Approach                  | Limitation                    |
|--------------|--------------------------|------------------------------|
| SafeInstr    | Data augmentation         | Fails on large datasets      |
| BEA          | Backdoor alignment        | Dataset-dependent            |
| Safe LoRA    | Subspace projection       | Utility degradation          |
| Resta        | Safety vector addition    | Trade-off instability        |
| **Safe Delta** | Delta selection + compensation | Balanced & robust            |

---

## 9. Advantages

- Consistent safety preservation
- High utility retention
- Scalable to real-world fine-tuning APIs
- Compatible with full FT and LoRA

---

## 10. Limitations

- Vulnerable to adversarially crafted datasets
- Greedy selection may be suboptimal
- Currently limited to text-based models

---

## 11. Summary

Safe Delta is a **weight-level safety control mechanism** that:

- Selects fine-tuning updates that improve utility
- Restricts updates that harm safety
- Compensates residual safety loss via Hessian-based correction

### Core Concept
> "Control what to learn, and correct what breaks safety."

---