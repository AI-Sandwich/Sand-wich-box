
#%%
import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE

# 1. Create or load time series dataframe
# Example structure: columns group_id, time_idx, target variable, covariates...
df = pd.DataFrame({
    "group_id": ["A"] * 400 + ["B"] * 400,
    "time_idx": list(range(400)) * 2,
    "value": np.sin(np.linspace(0, 20, 800)) + np.random.normal(scale=0.1, size=800),
})  # your data here

# 2. Prepare dataset parameters
max_encoder_length = 30
max_prediction_length = 10
training_cutoff = df["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="value",                # specify numeric or real target
    group_ids=["group_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],
    static_reals=[],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["value"],
    target_normalizer=GroupNormalizer(groups=["group_id"]),
    add_relative_time_idx=True,
    add_encoder_length=True,
    add_target_scales=True,
)

validation = TimeSeriesDataSet.from_dataset(
    training, df, predict=True, stop_randomization=True
)

train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
val_loader = validation.to_dataloader(train=False, batch_size=128, num_workers=0)

# 3. Initialize model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    loss=SMAPE(),
    log_interval=10,
    reduce_on_plateau_patience=3,
)

# 4. Setup PyTorch Lightning Trainer with callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
lr_logger = LearningRateMonitor()

trainer = pl.Trainer(
    max_epochs=20,
    accelerator="auto",
    devices="auto",
    gradient_clip_val=0.1,
    callbacks=[early_stop, lr_logger],
)

# 5. Train
trainer.fit(
    tft,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)

# 6. Predict
tft.predict(val_loader, mode="raw", return_x=True).output

# %%
