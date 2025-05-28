#%%
import numpy as np
import pandas as pd

def zero_one_inflated_beta_sample(n, alpha, beta, p_zero, p_one,to_print=True):
    """
    Generate samples from a zero-one inflated beta distribution.
    
    Parameters:
    - n: number of samples
    - alpha: beta distribution shape parameter (α > 0)
    - beta: beta distribution shape parameter (β > 0)
    - p_zero: probability of zero inflation (0 ≤ p_zero ≤ 1)
    - p_one: probability of one inflation (0 ≤ p_one ≤ 1, p_zero + p_one ≤ 1)
    
    Returns:
    - Array of n samples from the zero-one inflated beta distribution
    """
    # Validate inputs
    if not (0 <= p_zero <= 1 and 0 <= p_one <= 1 and (p_zero + p_one) <= 1):
        raise ValueError("Probabilities must be between 0 and 1 and sum to ≤ 1")
    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be positive")
    
    # Generate components
    components = np.random.choice(
        [0, 1, 2], 
        size=n, 
        p=[p_zero, p_one, 1 - p_zero - p_one]
    )
    
    # Generate beta samples
    beta_samples = np.random.beta(alpha, beta, size=n)
    
    # Combine components
    samples = np.where(components == 0, 0, np.where(components == 1, 1, beta_samples))
    # Print some statistics
    if to_print:
        print(f"Generated {n} samples:")
        print(f"- Zeros: {np.sum(samples == 0)/n:.1%} (expected {p_zero:.1%})")
        print(f"- Ones: {np.sum(samples == 1)/n:.1%} (expected {p_one:.1%})")
        print(f"- Beta distributed: {np.sum((samples > 0) & (samples < 1))/n:.1%}")
        print(f"Mean: {np.mean(samples):.4f}")
    
    return samples

# Parameters
n_sample = 100000
alpha = 2.0
beta = 8.0
p_zero = 0.1  # 10% chance of getting exactly 0
p_one = 0.05  # 5% chance of getting exactly 1

# Generate samples
samples = zero_one_inflated_beta_sample(n_sample, alpha, beta, p_zero, p_one)


pd.Series(samples).hist(bins=100)
# %%
