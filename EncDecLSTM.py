#%%
"""
Encoder–Decoder LSTM with Autoregressive Decoder for Multi-Horizon Binary Forecasting
------------------------------------------------------------------------------------
What this script does:
1) Generate synthetic weekly data for N series of length T with 3 covariates X.
2) Construct a binary target y via a logistic model depending on X and lagged y.
3) Split each series into:
   - history (encoder) length E = T - H
   - future (decoder) horizon H
   - actual future covariates (for training/eval) and separate scenario covariates (for stress tests).
4) Define an encoder–decoder LSTM:
   - Encoder: LSTM over history covariates.
   - Decoder: LSTMCell that runs AUTOREGRESSIVELY for H steps.
     At each step, input = [scenario/actual covariates at step h, previous y (true during training, predicted at test)].
5) Train with teacher forcing (BCEWithLogitsLoss over all H steps).
6) Evaluate on held-out set using actual future covariates (so we can compute metrics).
7) Also run a stress-test inference using scenario covariates.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm  # notebook-friendly progress bar
# ------------------------------
# 0. Repro & Device
# ------------------------------
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 1. Synthetic Data Generation
# ------------------------------
N = 1000   # number of series (entities)
T = 20     # total observed length per series
H = 5      # forecast horizon (future steps to predict)
E = T - H  # encoder (history) length

num_cov = 3  # number of covariates per timestep

# Covariates X ~ Normal(0,1), shape (N, T, num_cov)
X = np.random.randn(N, T, num_cov)

# Binary target y_t from a logistic model with lagged dependence
# y_t ~ Bernoulli( sigmoid( 0.8*x1 - 0.5*x2 + 0.3*x3 + 0.5*y_{t-1} ) )
y = np.zeros((N, T), dtype=np.float32)
for i in range(N):
    prev = 0.0
    for t in range(T):
        x1, x2, x3 = X[i, t]
        logit = 0.8 * x1 - 0.5 * x2 + 0.3 * x3 + 0.5 * prev
        p = 1 / (1 + np.exp(-logit))
        y[i, t] = np.random.binomial(1, p)
        prev = y[i, t]

# Actual future covariates (the last H steps of X) used for TRAINING & EVALUATION
X_future_actual = X[:, E:E+H, :]            # (N, H, num_cov)
y_future_actual = y[:, E:E+H]               # (N, H)
X_history = X[:, :E, :]                     # (N, E, num_cov)
y_last_obs = y[:, E-1:E]                    # (N, 1) last observed y at time t=E-1

# Scenario covariates for STRESS TEST (drawn from same distribution for demo)
X_future_scenario = np.random.randn(N, H, num_cov)

# ------------------------------
# 2. Train / Test Split
# ------------------------------
split = int(N * 0.8)

X_hist_train, X_hist_test = X_history[:split], X_history[split:]
Xf_act_train, Xf_act_test = X_future_actual[:split], X_future_actual[split:]
y_last_train, y_last_test = y_last_obs[:split], y_last_obs[split:]
y_fut_train, y_fut_test = y_future_actual[:split], y_future_actual[split:]
Xf_scn_test = X_future_scenario[split:]  # scenario only used for demo inference

# ------------------------------
# 3. PyTorch Tensors
# ------------------------------
X_hist_train_t = torch.tensor(X_hist_train, dtype=torch.float32).to(device)   # (B, E, num_cov)
X_hist_test_t  = torch.tensor(X_hist_test,  dtype=torch.float32).to(device)

Xf_act_train_t = torch.tensor(Xf_act_train, dtype=torch.float32).to(device)   # (B, H, num_cov)
Xf_act_test_t  = torch.tensor(Xf_act_test,  dtype=torch.float32).to(device)

Xf_scn_test_t  = torch.tensor(Xf_scn_test,  dtype=torch.float32).to(device)   # (B, H, num_cov), stress test

y_last_train_t = torch.tensor(y_last_train, dtype=torch.float32).to(device)   # (B, 1)
y_last_test_t  = torch.tensor(y_last_test,  dtype=torch.float32).to(device)

y_fut_train_t  = torch.tensor(y_fut_train, dtype=torch.float32).to(device)    # (B, H)
y_fut_test_t   = torch.tensor(y_fut_test,  dtype=torch.float32).to(device)

# ------------------------------
# 4. Model Definition (Encoder–Decoder with AR decoder)
# ------------------------------
class Encoder(nn.Module):
    """
    LSTM encoder over history.
    Input:  (batch, E, num_cov)
    Output: final hidden (num_layers, batch, hidden), final cell (num_layers, batch, hidden)
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, x):
        _, (h, c) = self.lstm(x)
        return h, c

class DecoderAR(nn.Module):
    """
    Autoregressive decoder using LSTMCell.
    At each step h = 1..H:
      input = [future_covariates_at_h, y_prev]
      where y_prev = true y (teacher forcing) during training, or predicted prob at inference.
    """
    def __init__(self, num_cov_dec, hidden_size, output_size=1):
        super().__init__()
        self.input_size = num_cov_dec + 1   # +1 to include previous y
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(self.input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, future_x, h_enc, c_enc, y0,
                teacher_forcing_targets=None, teacher_forcing=True):
        """
        future_x: (batch, H, num_cov_dec)  -> future known covariates (actual for training/eval, scenario for stress)
        h_enc: (num_layers, batch, hidden) -> encoder final hidden states
        c_enc: (num_layers, batch, hidden) -> encoder final cell states
        y0:    (batch, 1)                  -> last observed y at t=E-1
        teacher_forcing_targets: (batch, H) -> ground truth future y (only in training/eval)
        teacher_forcing: bool
        Returns:
            logits: (batch, H, 1)
        """
        batch, H, num_cov = future_x.size()
        assert num_cov + 1 == self.input_size, "Decoder input_size mismatch."

        # Initialize decoder hidden state with top encoder layer state
        if h_enc.dim() == 3:   # (num_layers, batch, hidden)
            h_t = h_enc[-1]
            c_t = c_enc[-1]
        else:                  # already (batch, hidden)
            h_t, c_t = h_enc, c_enc

        y_prev = y0  # (batch, 1), keep 2D to concatenate easily
        outputs = []

        for t in range(H):
            # Concatenate future covariates at step t with previous y
            step_x = future_x[:, t, :]                            # (batch, num_cov)
            inp = torch.cat([step_x, y_prev], dim=1)              # (batch, num_cov + 1)
            h_t, c_t = self.cell(inp, (h_t, c_t))                 # update state
            logit = self.fc(h_t)                                  # (batch, 1)
            outputs.append(logit.unsqueeze(1))                    # collect logits

            # Next-step previous y:
            if teacher_forcing and (teacher_forcing_targets is not None):
                y_prev = teacher_forcing_targets[:, t:t+1]        # (batch, 1), true y
            else:
                # Use predicted probability (softer than hard rounding)
                y_prev = torch.sigmoid(logit)                     # (batch, 1)

        return torch.cat(outputs, dim=1)  # (batch, H, 1)

class Seq2SeqAR(nn.Module):
    def __init__(self, input_size_enc, num_cov_dec, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.encoder = Encoder(input_size_enc, hidden_size, num_layers=num_layers, dropout=dropout)
        self.decoder = DecoderAR(num_cov_dec, hidden_size, output_size=1)

    def forward(self, enc_x, dec_x, y_last, y_future=None, teacher_forcing=True):
        """
        enc_x:    (batch, E, input_size_enc)
        dec_x:    (batch, H, num_cov_dec)
        y_last:   (batch, 1)
        y_future: (batch, H)  ground-truth future targets (only used if teacher_forcing=True)
        """
        h, c = self.encoder(enc_x)
        logits = self.decoder(
            future_x=dec_x,
            h_enc=h,
            c_enc=c,
            y0=y_last,
            teacher_forcing_targets=y_future,
            teacher_forcing=teacher_forcing
        )
        return logits  # (batch, H, 1)

# ------------------------------
# 5. Training Setup
# ------------------------------
hidden_size = 32
num_layers = 2
dropout = 0.0
lr = 0.01
epochs = 100
batch_size = 64

model = Seq2SeqAR(
    input_size_enc=num_cov,
    num_cov_dec=num_cov,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ------------------------------
# 6. Training Loop (Teacher Forcing)
# ------------------------------
def iterate_minibatches(B):
    """Yield mini-batch indices."""
    perm = torch.randperm(B, device=device)
    for i in range(0, B, batch_size):
        yield perm[i:i+batch_size]

B_train = X_hist_train_t.size(0)


# total number of iterations = epochs * number of batches per epoch
total_iters = epochs * (B_train // batch_size + int(B_train % batch_size > 0))

pbar = tqdm(total=total_iters, desc="Training", dynamic_ncols=True)
init_loss = 'N/A'

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for idx in iterate_minibatches(B_train):
        enc_x   = X_hist_train_t[idx]
        dec_x   = Xf_act_train_t[idx]
        y_last  = y_last_train_t[idx]
        target  = y_fut_train_t[idx]

        optimizer.zero_grad()
        logits = model(enc_x, dec_x, y_last, y_future=target, teacher_forcing=True)
        loss = criterion(logits.squeeze(-1), target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(epoch=epoch+1, init_loss=init_loss, loss=loss.item())
        pbar.update(1)
    init_loss = loss.item() if epoch == 0 else init_loss

pbar.close()

# ------------------------------
# 7. Evaluation on Test (Actual future covariates)
# ------------------------------
model.eval()
with torch.no_grad():
    logits_test = model(
        X_hist_test_t,        # history
        Xf_act_test_t,        # ACTUAL future covariates (so we can evaluate)
        y_last_test_t,
        teacher_forcing=False # AUTOREGRESSIVE inference
    )  # (B, H, 1)

    probs_test = torch.sigmoid(logits_test).cpu().numpy().squeeze(-1)  # (B, H)
    preds_test = (probs_test > 0.5).astype(int)

y_true = y_fut_test.reshape(-1)     # flatten over horizon
y_prob = probs_test.reshape(-1)
y_pred = preds_test.reshape(-1)

acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)
print(f"\n[Evaluation w/ actual future covariates]  Accuracy: {acc:.3f} | ROC-AUC: {auc:.3f}")

print("\nSample test case (actual future):")
print("Probs:", np.round(probs_test[0], 3))
print("Pred :", preds_test[0])
print("True :", y_fut_test[0])

# ------------------------------
# 8. Stress-Test Inference (Scenario covariates)
# ------------------------------
with torch.no_grad():
    logits_scn = model(
        X_hist_test_t,        # same history
        Xf_scn_test_t,        # SCENARIO future covariates (stress path)
        y_last_test_t,
        teacher_forcing=False
    )
    probs_scn = torch.sigmoid(logits_scn).cpu().numpy().squeeze(-1)

print("\nSample stress-test scenario forecast (probabilities):")
print("Scenario probs:", np.round(probs_scn[0], 3))
print("(No ground truth under scenario; shown for decisioning/sensitivity.)")

# %%
