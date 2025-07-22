# Positive-Definite Hessian Approximations in Newton-Like Methods

## Why Approximate the Hessian as Positive-Definite?

### 1. Core Theoretical Justification

For an optimization problem with loss function $\mathcal{L}(z)$, Newton's method uses the update:

$$
z_{k+1} = z_k - \eta [\nabla^2 \mathcal{L}(z_k)]^{-1} \nabla \mathcal{L}(z_k)
$$

**Critical requirements**:
1. The Hessian $\nabla^2 \mathcal{L}$ must be **invertible**
2. The update direction must be a **descent direction** ($\nabla \mathcal{L}^T \Delta z < 0$)

### 2. Problems with Indefinite Hessians

When $\nabla^2 \mathcal{L}$ is not positive-definite (PD):
- **Non-convex regions**: Negative eigenvalues → ascent directions
- **Saddle points**: May converge to unstable points
- **Numerical instability**: Matrix inversion fails

### 3. Standard PD Approximation Techniques

For a general Hessian $H$, common PD approximations:

| Method                | Formula                          | Application Context          |
|-----------------------|----------------------------------|------------------------------|
| Fisher Information    | $H \approx \mathbb{E}[\nabla \mathcal{L} \nabla \mathcal{L}^T]$ | Generalized Linear Models    |
| BHHH Approximation    | $H \approx J^T J$                | M-estimation problems        |
| Eigenvalue Correction | $H \approx Q \max(\Lambda,0) Q^T$ | Theoretical analysis         |

### 4. Mathematical Formulation

For a loss $\mathcal{L} = \frac{1}{2} r(z)^2$ with residual $r(z)$:

**Exact Hessian**:
$$
\nabla^2 \mathcal{L} = \nabla r(z) \nabla r(z)^T + r(z) \nabla^2 r(z)
$$

**PD Approximation** (Drops second term):
$$
H_{PD} = \nabla r(z) \nabla r(z)^T
$$

*Key properties*:
1. Always positive semi-definite (PSD)
2. Maintains curvature information along gradient directions
3. $H_{PD} \succeq 0$ by construction ($v^T H_{PD} v = \|v^T \nabla r\|^2 \geq 0$)

### 5. Convergence Guarantees

With PD approximation:
- **Global convergence**: To stationary points
- **Local convergence**: Quadratic rate near minima if $H_{PD} \approx \nabla^2 \mathcal{L}$
- **Stability**: Bounded step sizes via trust-region methods

### 6. Practical Implementation

```python
def pd_hessian(grad_r):
    """Compute PD Hessian approximation"""
    return np.outer(grad_r, grad_r)  # grad_r * grad_r^T


## References

1. **Boyd & Vandenberghe** (2004). *Convex Optimization*.  
   [https://web.stanford.edu/~boyd/cvxbook/](https://web.stanford.edu/~boyd/cvxbook/)  
   *Chapter 9: Newton Methods*

2. **Nocedal & Wright** (2006). *Numerical Optimization, 2nd ed.*  
   [https://link.springer.com/book/10.1007/978-0-387-40065-5](https://link.springer.com/book/10.1007/978-0-387-40065-5)  
   *Chapter 6: Trust-Region Methods*

3. **Efron & Hastie** (2016). *Computer Age Statistical Inference*.  
   [https://web.stanford.edu/~hastie/CASI/](https://web.stanford.edu/~hastie/CASI/)  
   *Section 4.3: Fisher Information*