# Positive-Definite Hessian Approximations in Newton-Like Methods

## 1. Theoretical Foundation

### Role of the Hessian in Newton's Method
For a loss function $\mathcal{L}(z)$, the Newton update is:
```math
z_{k+1} = z_k - \eta [\nabla^2 \mathcal{L}(z_k)]^{-1} \nabla \mathcal{L}(z_k)
```

The Hessian $\nabla^2 \mathcal{L}$ encodes curvature information:
- **Positive eigenvalues**: Convex (bowl-shaped) curvature → descent direction
- **Negative eigenvalues**: Concave curvature → ascent direction (problematic)

### Problem with Non-Positive-Definite Hessians
- If $\nabla^2 \mathcal{L}$ has negative eigenvalues:
  - Update may move toward a saddle point or local maximum
  - Matrix inversion becomes unstable
- In non-convex problems (like neural nets or custom XGBoost losses), indefinite Hessians are common

## 2. Why Approximate with PD Part?

### A. Guarantee Descent Directions
By forcing the Hessian to be PD:
```math
\Delta z = - [\text{PD}(\nabla^2 \mathcal{L})]^{-1} \nabla \mathcal{L}
```
we ensure $\nabla \mathcal{L}^T \Delta z < 0$ (strict descent).

### B. Connection to Fisher Information
For probabilistic models (e.g., logistic regression):
- The **expected Hessian** (Fisher information matrix) is always PSD
- Using the observed Hessian can lead to indefiniteness due to finite samples

### C. Trust-Region Theory
Approximating with a PD matrix is equivalent to restricting steps to a local convex region where the quadratic model is valid.

## 3. Common PD Approximation Techniques
For a Hessian $H$:

| Method               | Formula                          | Application Context          |
|----------------------|----------------------------------|------------------------------|
| **Abs Eigenvalues**  | $H \approx Q \|\Lambda\| Q^T$    | Theoretical analysis         |
| **Fisher Approx.**   | $H \approx \mathbb{E}[\nabla \mathcal{L} \nabla \mathcal{L}^T]$ | GLMs, XGBoost               |
| **BHHH Approx.**     | $H \approx J^T J$                | M-estimation                 |
| **Diagonal Shift**   | $H + \lambda I$                  | Levenberg-Marquardt          |

## 4. XGBoost-Specific Justification

For the expected exposure loss:
```math
\mathcal{L} = \frac{1}{2} (\sigma(z_1) z_2 a - y)^2
```

The exact Hessian would be:
```math
\nabla^2 \mathcal{L} = \begin{bmatrix}
a^2 z_2^2 \sigma'(z_1)^2 + r \cdot a z_2 \sigma''(z_1) & a^2 z_2 \sigma(z_1) \sigma'(z_1) + r \cdot a \sigma'(z_1) \\
a^2 z_2 \sigma(z_1) \sigma'(z_1) + r \cdot a \sigma'(z_1) & a^2 \sigma(z_1)^2
\end{bmatrix}
```

The **approximation** drops the residual-dependent terms:
```math
\nabla^2 \mathcal{L} \approx \begin{bmatrix}
a^2 z_2^2 \sigma'(z_1)^2 & a^2 z_2 \sigma(z_1) \sigma'(z_1) \\
a^2 z_2 \sigma(z_1) \sigma'(z_1) & a^2 \sigma(z_1)^2
\end{bmatrix}
```

## 5. Practical Implications

1. **Numerical Stability**:
   - Ensures invertibility of the Hessian
   - Prevents explosive updates from negative curvature

2. **Convergence Guarantees**:
   - Maintains convergence to local minima (avoiding saddle points)
   - Compatible with line search/trust-region methods

3. **Computational Efficiency**:
   - PD matrices enable faster linear solvers (Cholesky vs. LU)
   - Simplifies parallelization in distributed XGBoost
  

## 6. When Exact Hessians Matter

### Cases Requiring Exact Hessians

1. **Cubic Regularization Methods**  
   When using methods like:
```math
\Delta z = \arg\min \left( \nabla\mathcal{L}^T d + \frac{1}{2}d^T H d + \frac{\sigma}{3}\|d\|^3 \right)
```
   where exact curvature is crucial.

2. **Constrained Optimization**  
   For problems with:  
```math
\min_z \mathcal{L}(z) \text{ s.t. } g(z) \leq 0
```
   where constraint curvature must be preserved.

3. **Small-Residual Problems**  
   When $\|r(z)\| \approx 0$ and:  
```math
\nabla^2\mathcal{L} \approx J^T J
```
   becomes naturally PD anyway.

4. **Dominant Residual Terms**  
   When the second-order term:  
```math
\sum r_i(z)\nabla^2 r_i(z)
```  
   dominates the Fisher information term.

### Practical Implications
| Case | Exact Hessian Needed? | Reason |
|------|-----------------------|--------|
| Standard XGBoost | ❌ No | Uses Fisher approximation |
| High-precision physics | ✅ Yes | Residual curvature matters |
| Small datasets | ✅ Yes | Fisher approximation unreliable |
| Near convergence | ❌ No | Residuals $\approx$ 0 |


## 7. Key References

1. Boyd & Vandenberghe, *Convex Optimization* (2004) - Chapter 9  
2. Nocedal & Wright, *Numerical Optimization* (2006) - Chapter 6  
3. XGBoost Paper (*Chen & Guestrin, 2016*) - Fisher approximation discussion