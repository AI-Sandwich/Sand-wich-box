# Credit Risk Modeling: Concepts, Methods, and Modern Approaches

## 1. Core Credit Risk Parameters

Credit risk is typically quantified using three main components:

### 1.1 Probability of Default (PD)
- Measures the likelihood that a borrower defaults within a given time horizon.
- Usually modeled per account or obligor, **unweighted by exposure**, using logistic regression, survival analysis, or machine learning.
- PD can be **point-in-time (PIT)** or **through-the-cycle (TTC)** depending on whether macroeconomic conditions are incorporated.

### 1.2 Loss Given Default (LGD)
- Fraction of exposure lost if default occurs:
  ```math
  LGD_i = \frac{\text{Loss}_i}{\text{EAD}_i}.
  ```
- Portfolio-level LGD is often **EAD-weighted**:
  ```math
  \text{LGD}_{\text{portfolio}} = \frac{\sum_i LGD_i \cdot EAD_i}{\sum_i EAD_i}.
  ```
- Conditional on default, which leads to **selection bias** if unobserved factors affect both PD and LGD.

### 1.3 Exposure at Default (EAD) / Credit Conversion Factor (CCF)
- EAD = expected exposure at the time of default. For undrawn commitments, modeled using **CCF**:
  ```math
  \text{CCF} = \frac{\text{EAD at default – current balance}}{\text{Undrawn commitment}}.
  ```
- Often weighted by facility size when estimating LGD or portfolio losses.
- Can be modeled using regression, Tobit, or machine learning.

---

## 2. Selection Bias and the Heckman Correction

### 2.1 Problem Setup
- LGD (or drawdown amount) is only observed **conditional on default/drawdown**.
- OLS regression on this truncated sample produces **biased coefficient estimates** if unobserved factors affect both selection and outcome.

### 2.2 Inverse Mills Ratio (IMR)
- Derived from a probit model for the selection equation:
  ```math
  D_i = 1 \iff D_i^* = X_i \beta + u_i > 0
  ```
  where \(u_i \sim N(0,1)\).
- Conditional expectation of the residual given selection:
  ```math
  E[u_i \mid D_i=1, X_i] = \frac{\phi(X_i\beta)}{\Phi(X_i\beta)} = \lambda(X_i\beta)
  ```
- Including \(\lambda(X_i\beta)\) as a regressor in LGD/drawdown amount regression corrects the bias.

### 2.3 Practical Implications
- At the **individual level**, coefficients without IMR are biased.
- At the **forecast level for observed defaults**, predicted LGD may be asymptotically unbiased if you are only predicting conditional on selection.
- Threshold in the latent variable (e.g., \(0\) or \(c\)) does not change the IMR formula; it is absorbed into the intercept.

---

## 3. Copula and Correlation Modeling

- **PD, LGD, and EAD are often modeled separately**, but their joint correlation matters for portfolio risk.
- **Empirical correlation can be enforced** post hoc using copulas, but care is needed due to **partial observability**:
  - LGD is only observed when default occurs.
  - Likelihood can be written in **full-information form** using marginal PDFs and copula density.
- Copula methods allow **latent factor correlation** without requiring joint parametric regression for each risk parameter.

---

## 4. Drawdown Modeling and EAD

### 4.1 Drawdown Probability vs Amount
- **Probability of drawdown (binary)**: whether a borrower draws unused commitments.
- **Drawdown amount / CCF (continuous)**: expected percentage or absolute amount of drawdown.
- **Two-stage or hurdle models**:
  1. Stage 1: drawdown probability.
  2. Stage 2: drawdown amount conditional on drawdown.
- Common drivers:
  - Financial distress indicators
  - Macroeconomic conditions
  - Facility features and utilization history
  - Customer relationship factors

### 4.2 Selection Bias in Drawdowns
- Conditional modeling of drawdown amount suffers from the same **selection bias** as LGD.
- Methods to correct:
  - Heckman-style IMR
  - Joint copula-based modeling
  - Inverse probability weighting (IPW)
  - Multi-task or multi-output ML (partial mitigation if latent factors captured by features)

---

## 5. Machine Learning in Credit Risk

### 5.1 Usage
- **PD modeling**: widely used in consumer credit and fintech; tree ensembles, gradient boosting, neural networks.
- **LGD/EAD**: growing adoption, often as benchmark or challenger models.
- **Behavioral / drawdown models**: ML captures nonlinearities and interactions.

### 5.2 Selection Bias Considerations
- ML does **not inherently remove selection bias** if latent factors exist outside observed features.
- Multi-output ML can learn correlations **conditional on features**, but cannot see unobserved outcomes for non-defaulted accounts.
- Prediction within the observed subset (e.g., defaults only) is consistent; portfolio-level aggregation may still be biased.

---

## 6. Other Credit Risk Models Beyond PD/LGD/EAD

- **Portfolio / Concentration Models**: CreditMetrics, CreditRisk+, factor models for correlations.
- **Pricing & Valuation Models**: credit spreads, CVA, loan pricing, hazard rate models.
- **Behavioral / Account Management Models**: scorecards, cure/prepayment models, churn models.
- **Specialized Models**: recovery timing, collateral valuation, early warning models, macro-driven stress models.
- **Rating Transition Models**: Markov chain or structural models for corporate bonds and credit ratings.

---

## 7. Key Takeaways

1. PD, LGD, and EAD are the core building blocks of credit risk and portfolio expected loss:
   ```math
   \text{Expected Loss} = \sum_i PD_i \cdot LGD_i \cdot EAD_i
   ```
2. **Selection bias** arises when LGD or drawdown amount is only observed conditional on default/drawdown. IMR or joint modeling corrects this at the coefficient level.
3. **EAD weighting** ensures LGD estimates are economically consistent at the portfolio level.
4. **Multi-output ML** can capture correlations **conditionally**, but cannot eliminate bias from unobserved latent factors.
5. Drawdown modeling (CCF) is part of **credit risk**, not just behavioral analytics.
6. Other models (portfolio, pricing, stress testing, behavioral, recovery, rating transitions) complement PD/LGD/EAD to fully manage credit risk.

---

