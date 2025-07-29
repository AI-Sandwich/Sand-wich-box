# Deriving Softmax as the Canonical Link for Multiclass (Categorical) Distribution

We derive the **softmax** function as the **inverse of the canonical link** for a multiclass classification problem by expressing the **categorical distribution** in **exponential family form**.

---

## 🎯 Goal

Express the **categorical distribution** (i.e., multinomial with one trial) as a member of the **exponential family** and identify the corresponding **canonical link**.

---

## Step 1: Categorical (Multinomial) Distribution

Let:

- \( y \in \{1, 2, \dots, K\} \) (categorical label)
- One-hot encoded label:

  \[
  \mathbf{y} = (y_1, y_2, \dots, y_K), \quad \text{where } y_k = \begin{cases}1 & \text{if } y = k \\ 0 & \text{otherwise}\end{cases}
  \]

- Probability vector: \( \mathbf{p} = (p_1, \dots, p_K) \), with \( \sum_{k=1}^K p_k = 1 \)

The likelihood becomes:

\[
P(y) = \prod_{k=1}^K p_k^{y_k}
\]

Log-likelihood:

\[
\log P(y) = \sum_{k=1}^K y_k \log p_k
\]

---

## Step 2: Exponential Family Form

General form of exponential family:

\[
f(y) = h(y) \cdot \exp\left( \langle \boldsymbol{\theta}, \boldsymbol{T}(y) \rangle - A(\boldsymbol{\theta}) \right)
\]

Choose class \( K \) as the reference class and define natural parameters:

\[
\theta_k = \log\left( \frac{p_k}{p_K} \right), \quad \text{for } k = 1, \dots, K-1
\]

Then:

\[
p_k = \frac{e^{\theta_k}}{1 + \sum_{j=1}^{K-1} e^{\theta_j}}, \quad p_K = \frac{1}{1 + \sum_{j=1}^{K-1} e^{\theta_j}}
\]

Substitute into the log-likelihood:

\[
\log P(\mathbf{y}) = \sum_{k=1}^{K-1} y_k \theta_k + \log p_K
\]

Using:

\[
\log p_K = - \log\left( 1 + \sum_{j=1}^{K-1} e^{\theta_j} \right)
\]

So:

\[
P(\mathbf{y}) = \exp\left( \sum_{k=1}^{K-1} y_k \theta_k - \log\left( 1 + \sum_{j=1}^{K-1} e^{\theta_j} \right) \right)
\]

---

## Identifying Exponential Family Components

| Component                  | Expression                                                                 |
|---------------------------|----------------------------------------------------------------------------|
| Sufficient statistics \( \boldsymbol{T}(y) \) | \( \left[ y_1, y_2, \dots, y_{K-1} \right] \)                         |
| Natural parameters \( \boldsymbol{\theta} \) | \( \left[ \theta_1, \theta_2, \dots, \theta_{K-1} \right] \)         |
| Log-partition \( A(\boldsymbol{\theta}) \)   | \( \log\left( 1 + \sum_{j=1}^{K-1} e^{\theta_j} \right) \)         |
| Base measure \( h(y) \)                      | \( 1 \)                                                               |

Final form:

\[
P(\mathbf{y}) = \underbrace{1}_{h(y)} \cdot \exp\left( \underbrace{\sum_{k=1}^{K-1} y_k \theta_k}_{\langle \theta, T(y) \rangle} - \underbrace{\log\left(1 + \sum_{j=1}^{K-1} e^{\theta_j} \right)}_{A(\theta)} \right)
\]

---

## Canonical Link Function

In GLMs, the **canonical link** is:

\[
\eta_k = \log\left( \frac{p_k}{p_K} \right), \quad \text{for } k = 1, \dots, K-1
\]

This gives the inverse link (i.e., **softmax**):

\[
p_k = \frac{e^{\eta_k}}{\sum_{j=1}^K e^{\eta_j}}, \quad \text{for all } k
\]

---

## Summary Table

| Quantity                | Expression                                                  |
|------------------------|-------------------------------------------------------------|
| Target space           | \( y \in \{1, 2, \dots, K\} \), one-hot encoded             |
| Canonical parameters   | \( \theta_k = \log\left( \frac{p_k}{p_K} \right) \)         |
| Sufficient statistics  | \( T(y) = (y_1, \dots, y_{K-1}) \)                           |
| Log-partition function | \( A(\theta) = \log\left(1 + \sum_{j=1}^{K-1} e^{\theta_j} \right) \) |
| Canonical link         | \( \eta_k = \log\left( \frac{p_k}{p_K} \right) \)           |
| Inverse link (softmax) | \( p_k = \frac{e^{\eta_k}}{\sum_{j=1}^K e^{\eta_j}} \)      |


---

### ❓ Why Not Use Only N−1 Logits?

For binary classification:
- We only need **1 output** (since \( P(y=0) = 1 - P(y=1) \)).

For multiclass classification:
- Softmax models \( K \) logits:
  \[
  p_k = \frac{e^{\eta_k}}{\sum_{j=1}^K e^{\eta_j}}
  \]
- However, the softmax is **invariant to constant shifts** in logits:
  \[
  \text{softmax}(\eta_1 + c, ..., \eta_K + c) = \text{softmax}(\eta_1, ..., \eta_K)
  \]

🔁 This means the model is **overparameterized** — only **K−1** directions matter. One logit is redundant.

---

### 🧠 Modeling Strategies

| Strategy                     | Description                                                  | Parameters Used         |
|------------------------------|--------------------------------------------------------------|--------------------------|
| **Constrained Softmax**      | Model all \( K \) logits, apply normalization constraint     | \( K \) logits, \( K - 1 \) effective |
| **Reference Class (baseline)** | Fix one logit (e.g., class K) to 0, model rest relative to it | \( K - 1 \) logits        |

---

### ✅ Summary Table

| Problem Type    | # of Classes \( K \) | # of Outputs Needed | Link Function     | Inverse Link       |
|-----------------|----------------------|----------------------|-------------------|---------------------|
| Binary          | 2                    | 1                    | Logit             | Sigmoid             |
| Multiclass      | \( K > 2 \)          | \( K \), but \( K-1 \) effective | Multinomial logit | Softmax             |

> ✅ In GLMs, softmax (like sigmoid) is **probabilistically optimal** under the correct distributional assumptions (categorical or Bernoulli).

---

### 🧪 Implementation Note

In practice (e.g., XGBoost, PyTorch, scikit-learn):
- Multiclass models often **output \( K \) logits**.
- The overparameterization is handled implicitly by the **softmax layer**.
- Optimization proceeds in the probability space (after softmax), so the redundancy is not a practical problem.

