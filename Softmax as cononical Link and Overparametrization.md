# Deriving Softmax as Canonical Link for Categorical Distribution

We derive the softmax function as the inverse of the canonical link for a multiclass classification problem, by expressing the categorical distribution as a member of the exponential family.

---

## Step 1: Categorical Distribution

Let $y \in \{1, ..., K\}$ be a class label.

We one-hot encode $y$ as a vector:

```math
y = (y_1, y_2, ..., y_K), \quad y_k \in \{0, 1\}, \quad \sum_{k=1}^K y_k = 1
```

Let the class probabilities be:

```math
p = (p_1, ..., p_K), \quad \sum_{k=1}^K p_k = 1
```

Then the PMF of the categorical distribution is:

```math
P(y) = \prod_{k=1}^K p_k^{y_k}
```

The log-likelihood is:

```math
\log P(y) = \sum_{k=1}^K y_k \log p_k
```

---

## Step 2: Exponential Family Form

The exponential family has the form:

```math
f(y) = h(y) \exp\left( \theta^T T(y) - A(\theta) \right)
```

We set:

- Natural parameters: $\theta_k = \log(p_k / p_K)$ for $k = 1, ..., K-1$
- Reference class: $p_K$ (the denominator in the softmax)

Then:

```math
p_k = \frac{\exp(\theta_k)}{1 + \sum_{j=1}^{K-1} \exp(\theta_j)}, \quad \text{for } k = 1, ..., K-1
```

```math
p_K = \frac{1}{1 + \sum_{j=1}^{K-1} \exp(\theta_j)}
```

Substituting into the log-likelihood:

```math
\log P(y) = \sum_{k=1}^{K-1} y_k \theta_k + y_K \log p_K
```

But since:

```math
y_K = 1 - \sum_{k=1}^{K-1} y_k
```

We have:

```math
\log P(y) = \sum_{k=1}^{K-1} y_k \theta_k + \left(1 - \sum_{k=1}^{K-1} y_k\right) \log p_K
```

Simplifying:

```math
\log P(y) = \sum_{k=1}^{K-1} y_k \theta_k - \log\left(1 + \sum_{j=1}^{K-1} \exp(\theta_j)\right)
```

So:

```math
P(y) = \exp\left( \sum_{k=1}^{K-1} y_k \theta_k - A(\theta) \right)
```

---

## Step 3: Match to Exponential Family

Comparing to:

```math
f(y) = h(y) \exp\left( \theta^T T(y) - A(\theta) \right)
```

We identify:

- $T(y) = (y_1, ..., y_{K-1})$
- $\theta = (\theta_1, ..., \theta_{K-1})$
- $A(\theta) = \log\left(1 + \sum_{j=1}^{K-1} \exp(\theta_j)\right)$
- $h(y) = 1$

---

## Step 4: Canonical Link is Softmax

In a GLM, the canonical link is defined as:

```math
\theta_k = \log\left(\frac{p_k}{p_K}\right), \quad \text{for } k = 1, ..., K-1
```

To solve for $p_k$, introduce $\theta_K = 0$ for identifiability and rewrite:

```math
p_k = \frac{\exp(\theta_k)}{\sum_{j=1}^K \exp(\theta_j)}, \quad \text{for } k = 1, ..., K
```

This is exactly the **softmax function**.

---

## ✅ Summary

- The softmax function arises naturally as the **inverse of the canonical link** for the categorical distribution.
- The natural (canonical) parameter is the log-odds $\theta_k = \log(p_k / p_K)$.
- The partition function $A(\theta)$ ensures normalization: all class probabilities sum to 1.

---
# Why Not Use Only N−1 Logits in Multiclass Classification?

In binary classification, we only need one logit because the probability of the second class is determined as `1 - p`. But in **multiclass classification**, the situation is a bit different.

---

## 1. Softmax Uses All $N$ Logits

In softmax regression (multinomial logistic regression), we assign a score (logit) to **each** class:

```math
z_k = \text{logit for class } k, \quad k = 1, ..., N
```

We convert logits to class probabilities using the softmax function:

```math
p_k = \frac{e^{z_k}}{\sum_{j=1}^{N} e^{z_j}}
```

This ensures:

- All probabilities $p_k$ are between 0 and 1
- $\sum_{k=1}^N p_k = 1$

---

## 2. Why Not Use Only $N-1$?

Technically, the softmax function has **redundancy** because if you shift all logits by a constant (e.g., subtract the same value from all $z_k$), the probabilities don’t change:

```math
\text{softmax}(z_1 + c, ..., z_N + c) = \text{softmax}(z_1, ..., z_N)
```

This means only **$N - 1$ logits are linearly independent**.

In fact, in **statistical modeling**, this is exactly what we do:

- We fix one class (say, class $N$) as the reference.
- We parameterize the log-odds of other classes relative to it:

```math
\log\left(\frac{p_k}{p_N}\right) = \theta_k, \quad k = 1, ..., N-1
```

So, yes — **you only need $N - 1$ parameters** to fully specify the model. That’s often how it’s done in GLMs (Generalized Linear Models).

---

## 3. Why Use $N$ in Practice?

Libraries like XGBoost, PyTorch, TensorFlow, etc., typically use **$N$ logits** for simplicity and symmetry:

- It avoids having to pick a “reference class”
- Code is simpler and works for any $N$
- The softmax automatically handles normalization

The redundancy is harmless: model weights are **not uniquely identifiable**, but predictions are.

---

## ✅ Summary

- Softmax maps $N$ logits to $N$ probabilities summing to 1.
- Only $N - 1$ logits are linearly independent due to translation invariance.
- Statistical models often use $N - 1$ logits (e.g., GLMs), but practical implementations use all $N$ for convenience.
