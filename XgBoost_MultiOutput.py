#%%
import numpy as np
import xgboost as xgb

def expected_exposure_loss(preds, dtrain):
    # Extract true values
    y_true = dtrain.get_label().reshape(-1, 2)  # [prob_true, impact_true]
    y_prob_true = y_true[:, 0]
    y_impact_true = y_true[:, 1]
    
    # Get amount from features (assuming it's the last column)
    # X = np.array(dtrain.get_data().todense()) if hasattr(dtrain.get_data(), 'todense') else dtrain.get_data()
    amounts = dtrain.get_float_info('feature_weights')  # Last column is amount
    
    # Reshape predictions
    preds = preds.reshape(-1, 2, order='F')
    z1 = preds[:, 0]  # Logit for probability
    z2 = preds[:, 1]  # Raw impact prediction
    
    # Compute probability and expected exposure
    probability_pred = 1 / (1 + np.exp(-z1))
    expected_exposure_pred = probability_pred * z2 * amounts
    expected_exposure_true = y_prob_true * y_impact_true * amounts
    
    # Gradient calculations
    dL_dOutput = (expected_exposure_pred - expected_exposure_true)
    dProbability_dz1 = probability_pred * (1 - probability_pred)
    
    grad_z1 = dL_dOutput * dProbability_dz1 * z2 * amounts
    grad_z2 = dL_dOutput * probability_pred * amounts
    
    # Hessians
    hess_z1 = (dProbability_dz1 * z2 * amounts)**2
    hess_z2 = (probability_pred * amounts)**2
    
    # Interleave gradients/hessians
    grad = np.column_stack([grad_z1, grad_z2]).flatten(order='F')
    hess = np.column_stack([hess_z1, hess_z2]).flatten(order='F')
    
    return grad, hess

# Sample data
np.random.seed(42)
n_samples = 100
n_features = 5
X_features = np.random.rand(n_samples, n_features)

amounts = np.random.uniform(1, 100, n_samples)
X_train = np.column_stack([X_features, amounts])  # Add amount as last feature

y_prob_train = np.random.randint(0, 2, n_samples)
y_impact_train = np.random.rand(n_samples) * 10
y_multi_train = np.column_stack([y_prob_train, y_impact_train])

# Train
dtrain = xgb.DMatrix(X_train, label=y_multi_train)
dtrain.set_float_info('feature_weights',X_features[:,-1])
params = {
    "disable_default_eval_metric": 1,
    "eta": 0.1,
    "max_depth": 3,
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    obj=expected_exposure_loss,
)

# Predict (must include amount in features!)
X_test = np.column_stack([X_features[:5], amounts[:5]])
dtest = xgb.DMatrix(X_test)
raw_preds = model.predict(dtest).reshape(-1, 2, order='F')
probability_pred = 1 / (1 + np.exp(-raw_preds[:, 0]))
impact_pred = raw_preds[:, 1]

print("Probability:", probability_pred)
print("Impact:", impact_pred)
print("Amount:", amounts[:5])
print("Expected Exposure:", probability_pred * impact_pred * amounts[:5])
# %%
