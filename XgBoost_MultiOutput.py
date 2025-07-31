#%%
"""
This script demonstrates a custom multi-output objective for XGBoost to jointly predict:
1. The probability of an event (as a sigmoid/logit output).
2. The impact of the event (as a regression output).

The custom loss function (`expected_exposure_loss`) is designed to optimize the expected exposure, defined as:
    expected_exposure = probability * (rate + impact) * amount

Key components:
- Custom objective function for multi-output regression using XGBoost's `multi_strategy="multi_output_tree"`.
- Synthetic data generation for demonstration, with features, rates, and amounts.
- Evaluation metrics:
    - Probability prediction is evaluated using mean squared error (MSE).
    - Impact prediction is evaluated using mean squared error (MSE).
    - Expected exposure is also evaluated using MSE.
"""
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.metrics import brier_score_loss
from math import sqrt
from tabulate import tabulate

def expected_exposure_loss(preds, dtrain):
    # Extract true values
    y_true = dtrain.get_label().reshape(-1, 2)  # [prob_true, impact_true]
    y_prob_true = y_true[:, 0]
    y_impact_true = y_true[:, 1]
    
    # Get amount from features (assuming it's the last column)
    # X = np.array(dtrain.get_data().todense()) if hasattr(dtrain.get_data(), 'todense') else dtrain.get_data()
    amounts = dtrain.get_data()[:, -1].toarray().flatten()  # Last column is amount
    rates = dtrain.get_data()[:, -2].toarray().flatten() # Second last column is rate

    
    # Reshape predictions
    z1 = preds[:, 0]  # Logit for probability
    z2 = preds[:, 1]  # Raw impact prediction
    
    # Compute probability and expected exposure
    probability_pred = 1 / (1 + np.exp(-z1))
    expected_exposure_pred = probability_pred * (rates + z2) * amounts
    expected_exposure_true = y_prob_true * (rates + y_impact_true) * amounts
    
    # Gradient calculations
    dL_dOutput = (expected_exposure_pred - expected_exposure_true)
    dProbability_dz1 = probability_pred * (1 - probability_pred)
    
    grad_z1 = dL_dOutput * dProbability_dz1 * (rates + z2) * amounts
    grad_z2 = dL_dOutput * probability_pred * amounts
    
    # Hessians
    hess_z1 = (dProbability_dz1 * (rates + z2) * amounts)**2
    hess_z2 = (probability_pred * amounts)**2
    
    # Interleave gradients/hessians
    grad = np.column_stack([grad_z1, grad_z2])
    hess = np.column_stack([hess_z1, hess_z2])
    
    return grad, hess

# Sample data
np.random.seed(42)
n_samples = 20000
n_features = 5
amounts = np.random.uniform(low=10000, high=20000, size=n_samples)
rates = np.random.uniform(low=0, high=0.08, size=n_samples)
X = np.column_stack([np.random.rand(n_samples, n_features), rates, amounts])
# Create labels as sigmoid of a linear combination of X_train plus noise
# Introduce correlation between y_prob and y_impact via a shared latent variable
shared_latent = X[:, 0] * 0.7 + X[:, 1] * -0.4  # Common influence

linear_comb = X[:, 0] * 0.5 + X[:, 1] * -0.3 + X[:, 2] * 0.8 + 0.3 * shared_latent
sigmoid = 1 / (1 + np.exp(-linear_comb))
noise = 1 + np.random.uniform(-0.1, 0.1, size=n_samples)
y_prob = sigmoid * noise

impact_linear_comb = X[:, 3] * 2.0 + X[:, 4] * -1.5 + rates + 0.5 * shared_latent
impact_noise = 1 + np.random.uniform(-0.1, 0.1, size=n_samples)
y_impact = (impact_linear_comb * impact_noise).clip(min=0.1)  # Ensure positive impact


y = np.column_stack([y_prob, y_impact])
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Re-create DMatrix for training with new split
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
# Define parameter setups for comparison
param_setups = [
    {
        "name": "Baseline (no tree_method, no multi_strategy)",
        "params": {
            "disable_default_eval_metric": 1,
            "eta": 0.1,
            "max_depth": 3
        }
    },
    {
        "name": "With tree_method & multi_strategy",
        "params": {
            "disable_default_eval_metric": 1,
            "eta": 0.1,
            "max_depth": 3,
            "tree_method": "hist",
            "multi_strategy": "multi_output_tree"
        }
    }
]

results = []

# Define true values for test set
probability_true = y_test[:, 0]
impact_true = y_test[:, 1]
amounts_test = X_test[:, -1]
rates_test = X_test[:, -2]
expected_exposure_true = probability_true * (rates_test + impact_true) * amounts_test

for setup in param_setups:
    model = xgb.train(
        setup["params"],
        dtrain,
        num_boost_round=100,
        obj=expected_exposure_loss,
    )
    raw_preds = model.predict(dtest).reshape(-1, 2)
    probability_pred = 1 / (1 + np.exp(-raw_preds[:, 0]))
    impact_pred = raw_preds[:, 1]

    prob_loss = mean_squared_error(probability_true, probability_pred)
    impact_mse = mean_squared_error(impact_true, impact_pred)
    expected_exposure_pred = probability_pred * (rates_test + impact_pred) * amounts_test
    expected_exposure_mse = mean_squared_error(expected_exposure_true, expected_exposure_pred)
    relative_rmse = sqrt(expected_exposure_mse) / np.mean(expected_exposure_true)

    results.append({
        "name": setup["name"],
        "prob_loss": prob_loss,
        "impact_mse": impact_mse,
        "expected_exposure_rmse": sqrt(expected_exposure_mse),
        "relative_rmse": relative_rmse
    })
# After the loop, print the results only once
results_df = pd.DataFrame(results)
print(tabulate(results_df, headers='keys', tablefmt='fancy_grid', floatfmt=".6f", showindex=False))
# Discussion:
# When "multi_strategy": "multi_output_tree" is not set, XGBoost uses one tree per target (i.e., independent trees for each output).
# In this mode, your custom loss function still computes gradients and hessians using both targets, but XGBoost will split the gradients/hessians per output and fit each tree independently.
# This means the trees do not "see" the joint structure in the splits, even though your gradients are coupled.
# With "multi_strategy": "multi_output_tree", XGBoost builds a single tree that jointly splits on all outputs, allowing splits to be chosen based on the combined gradient/hessian information.
# For custom objectives that couple outputs (like yours), "multi_output_tree" is generally more appropriate, as it allows the model to exploit the joint structure in the loss.
# In summary: single-tree (multi_output_tree) is better for coupled objectives; per-target trees may not fully leverage your custom loss.

# %%
