# Multi-Horizon Time Series Forecasting: Encoder-Decoder LSTM and TFT

## 1. Encoder-Decoder LSTM for Time Series

### 1.1 Overview
- Encoder-decoder LSTM extends the vanilla LSTM to handle **sequence-to-sequence (seq2seq) tasks**.
- Encoder summarizes past history into hidden states (context vector).
- Decoder produces future representations and ultimately forecasts multiple time steps.

### 1.2 Handling Variable-Length Sequences
- Use **padding and masking** for different IDs with varying sequence lengths.
- Packing sequences (`pack_padded_sequence` in PyTorch) ensures LSTM ignores padded steps.
- Vanilla LSTM can process variable lengths but requires manual handling and is limited in multi-horizon forecasting.

### 1.3 Forecasting Multiple Periods
- Vanilla LSTM produces outputs aligned with input timestamps → not directly forecasting the future.
- Encoder-decoder LSTM separates history (encoder) and future (decoder) sequences.
- **Autoregressive decoding:** decoder can feed previous outputs into the next step.
- Loss functions can be defined over the full forecast horizon to train all weights together.
- Weights are optimized based on multi-step output rather than one-step rolling forecasts, mitigating train–inference mismatch.

### 1.4 Context Vector and Decoder
- Context vector = final or sequence of hidden states from encoder.
- Decoder LSTM processes **known future inputs** (scenario variables) step-by-step.
- Forecast at t+h uses:
  - Encoder hidden states (history up to t)
  - Known future scenario at t+h
  - Static context
- Prevents information leakage: decoder does not attend to future known inputs for earlier forecasts.

### 1.5 Multi-Horizon Training and Loss
- Loss function typically averages over horizon and sums over batch:
  - **Mean over horizon:** ensures each forecast step contributes equally.
  - **Sum over batch:** aggregates across series/entities.
- Encoder-decoder weights are updated using the full-horizon loss, allowing horizon-aware training.

### 1.6 Practical Considerations
- Historical window `n` should capture relevant patterns, but not overly long.
- Short horizons (e.g., <50 steps) rarely benefit from attention.
- Use exogenous scenario variables as **known-future inputs**.
- Rarely-changing features that are unknown at inference should be treated as static.

---

## 2. Embeddings and Static Features

### 2.1 Embeddings
- Transform categorical/static variables into dense continuous vectors.
- Allow the model to learn similarity and relationships across IDs/entities.
- Improves generalization and efficiency compared to one-hot encoding.

### 2.2 Static Enrichment
- Static features (e.g., product type, geography) are processed via GRN (Gated Residual Network) to produce a **static context vector**.
- Injected into encoder, decoder, and attention layers.
- Rarely-changing variables unknown at inference should be treated as static.

---

## 3. Temporal Fusion Transformer (TFT)

### 3.1 Overview
- TFT is an advanced architecture for **multi-horizon forecasting with known future covariates**.
- Components:
  1. **Embedding layers:** categorical and continuous features.
  2. **Variable Selection Networks (VSN):** learn feature importance per timestep.
  3. **Encoder LSTM:** processes historical inputs.
  4. **Decoder LSTM:** processes known-future inputs.
  5. **Static Enrichment:** conditions temporal components on entity identity.
  6. **Temporal Attention:** attends over encoder outputs only (past history).
  7. **Post-attention GRNs and gating:** non-linear fusion and stabilization.
  8. **Output head:** multi-horizon predictions (quantile or point forecasts).

### 3.2 Encoder and Decoder in TFT
- Encoder LSTM outputs **sequence of hidden states** (one per history timestep) for attention.
- Decoder LSTM outputs **sequence of hidden states** aligned with future known inputs.
- Forecasts are produced after combining decoder hidden states with attention + static enrichment → **not autoregressive on targets**.

### 3.3 Attention Mechanism
- **Temporal attention** is restricted to historical encoder outputs.
- Decoder hidden states **do not attend to future steps** beyond the current forecast to prevent leakage.
- Future known inputs are incorporated sequentially via the decoder LSTM.

### 3.4 Gated Residuals and GRNs
- **Gated Residual Networks (GRNs):** allow flexible information flow and stabilize training.
- **Post-attention GRNs:** process attention outputs before prediction layers.

### 3.5 Advantages of TFT
- Handles multi-horizon forecasts **in one shot**.
- Incorporates known-future scenario variables safely.
- Provides interpretability: variable selection weights, attention maps.
- Optimizes full-horizon loss directly, reducing train–inference mismatch.

### 3.6 Practical Considerations
- TFT is heavier than encoder-decoder LSTM:
  - ~2–5× parameters and 2–4× training time.
- Worth it when:
  - History is long (>50–100 steps)
  - Many covariates
  - Multi-horizon optimization is critical
  - Explainability is required (banking/stress testing)
- For short-history problems (<20–30 steps), TFT may not outperform a simple LSTM and adds unnecessary complexity.

---

## 4. Scenario-Based Forecasting

### 4.1 Feeding Scenarios
- Known future covariates (macro variables, rates) are fed as **decoder inputs** step-by-step.
- Avoid attention over future known inputs to prevent leakage.
- Forecasts at t+h only use:
  - History (encoder states)
  - Scenario at t+h (and earlier implicitly through decoder hidden states)
  - Static features

### 4.2 Autoregressive vs. Multi-Horizon
- Encoder-decoder LSTM can be made autoregressive (seq2scalar) or multi-output.
- Multi-output approach: produces all horizons in one step but cannot safely use scenario paths stepwise without leakage.
- TFT naturally produces **multi-horizon forecasts** using known-future inputs sequentially in decoder, combining attention over history.

### 4.3 Short-History Example: Mortgage Drawdowns
- Weekly commitment life ≤20 weeks; horizon ≤20 weeks.
- History length is short → attention layer in TFT likely unnecessary.
- Encoder–decoder LSTM or discrete-time hazard model is sufficient.
- Feed scenario variables week-by-week as known-future inputs in decoder.
- Keep model small (hidden size 32–64, 1–2 layers, modest embeddings).

---

## 5. Summary Recommendations

| Aspect | Encoder–Decoder LSTM | TFT |
|--------|-------------------|-----|
| History length | Any, but short histories suffice | Benefits long histories (>50–100 steps) |
| Scenario handling | Autoregressive decoder feeds scenario stepwise | Decoder LSTM + VSN incorporates known-future inputs safely |
| Multi-horizon | Autoregressive or multi-output | Multi-horizon one-shot, full-horizon loss |
| Interpretability | Limited | High: VSN + attention maps + static enrichment |
| Training resources | Light | 2–5× parameters and 2–4× time |
| Use case | Short/mid horizon, simple covariates | Complex scenario-based, long horizon, many covariates, regulated domains |
| Industry adoption | Common for challengers | Banking: challenger model, stress testing scenarios, or when interpretability is required |

---

**References / Further Reading**
- Bryan Lim & Sercan Ö. Arik, *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*, 2019.  
- PyTorch Forecasting library: [https://pytorch-forecasting.readthedocs.io](https://pytorch-forecasting.readthedocs.io)  
- Deep learning for time series: sequence models, attention, and scenario forecasting.  

