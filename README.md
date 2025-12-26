# CT-VAE with Macro-Conditional Prior

This repository implements a Conditional Time-VAE (CT-VAE) for probabilistic time-series forecasting, where latent dynamics are conditioned not only on observed history but also on macroeconomic states learned via a pretrained macro encoder.

The model is designed to capture cyclical structure, uncertainty, and tail risk, and supports scenario-based forecasting and rolling evaluation.

## Model Overview

- Macro Encoder:
Learns latent representations of macroeconomic cycles (TCN-based, pretrained separately).

- Conditional Latent Prior:
Latent prior conditioned on both scenario variables and macro latent states.

- Temporal Encoder:
FiLM-modulated TCN for condition-aware sequence encoding.

- Decoder:
Structured decomposition into trend, seasonality, and residual dynamics, with a Student-t likelihood for heavy-tailed uncertainty.

## Features

- Uncertainty-aware forecasting

- Scenario simulation & stress testing

- Rolling backtest / forward evaluation

- Probabilistic & risk metrics (NLL, CRPS, Coverage, VaR, ES)

- Baselines: ARIMA, LSTM, vanilla cVAE


## Usage
python macro_pretrain.py

python main.py

## Applications

Macroeconomic-driven forecasting

Cycle & risk analysis

Scenario-based projections
