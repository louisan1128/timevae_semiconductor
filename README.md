# Semiconductor Cycles: Exploring Possible Futures

**Generative Scenario Modeling for Semiconductor Industry Cycles**  
An undergraduate team project using a Conditional Time-VAE to generate semiconductor export scenarios.

Semiconductor cycles are affected by exchange rates, interest rates, and manufacturing activity, as well as past export and price movements. We started this project with a question: could we use this information together to generate several possible paths for the industry?

We tried scenarios with a weaker Korean won, lower interest rates, and improving economic indicators, then compared how the forecast distributions and downside risks changed.

[Presentation slides](Generative%20Scenario%20Modeling%20for%20Semiconductor%20Industry%20Cycles.pdf) · [Experiment notebook](Timevae_final.ipynb)

## How we built it

We used monthly semiconductor data from January 2010 to August 2025, including exports, DRAM, exchange rates, CAPEX, PMI, CLI, and ISM. The model takes **36 months of observations and generates the following 12 months**.

There was not much data to train on at this frequency. To make use of a longer history, we first pretrained an encoder on FRED macroeconomic data starting in 1970, then connected its representations to the semiconductor model's conditional prior.

The model was implemented in PyTorch. This is the overall architecture used in our presentation:

![CT-VAE model architecture](assets/model-architecture.png)

The temporal encoder uses TCNs with FiLM conditioning. The decoder combines **trend, seasonality, and residual components**, with a Student-t output distribution to represent large fluctuations. In the scenario experiments, we sample around the temporal encoder's latent distribution and change the economic conditions passed to the decoder.

## What we found

This is the forecast under the baseline conditions. The orange dashed line shows the median, and the shaded areas show quantile ranges across the generated paths. The vertical axis uses standardized values. This figure is from the original presentation.

![Semiconductor export forecast under baseline conditions](assets/baseline-forecast.png)

We compared the model with ARIMA, LSTM, and vanilla cVAE. Along with prediction error, we evaluated the forecast distributions using NLL, CRPS, and prediction interval coverage.

Here are the main comparison results reported in the presentation. Lower MSE and CRPS are better.

| Model | MSE ↓ | CRPS ↓ | 80% interval coverage |
|---|---:|---:|---:|
| CT-VAE | 0.4449 | **0.1322** | 86.19% |
| Vanilla cVAE | 0.4476 | 0.1493 | 81.58% |
| LSTM | **0.3114** | 0.1714 | 82.44% |
| ARIMA | 0.9491 | 0.7676 | 27.20% |

LSTM had the lowest point prediction error, while CT-VAE had the lowest CRPS in this comparison. Our main interest was generating possible future paths and examining their distributions. For an 80% prediction interval, coverage should be assessed against the 80% target together with the interval width.

We also removed individual components to see how the results changed. The table below keeps three measures from the ablation results: prediction error, interval coverage, and the proportion of paths with a decline greater than 10%.

| Configuration | MSE ↓ | 80% interval coverage | Paths with >10% decline |
|---|---:|---:|---:|
| Full CT-VAE | 0.1989 | 51.00% | 14% |
| Without Student-t | 0.1765 | 58.33% | 18% |
| Without Macro Prior | 0.1920 | 33.33% | 10% |
| Without FiLM | 0.1892 | 41.67% | 9% |
| Without trend/seasonality decomposition | 0.2033 | 51.00% | 22% |

Removing some components reduced the error. However, removing the macro prior or FiLM reduced interval coverage, while removing the decomposition increased the proportion of large-decline paths. The full model's 80% interval coverage was only 51% in this experiment, so uncertainty calibration still needs work. These results come from a separate experiment from the model comparison above. A smaller proportion of declining paths alone does not establish better model accuracy.

We then changed the economic conditions. The following values come from the **final scenario summary table in the presentation**, using baseline conditions from August 2025.

| Scenario | Paths with rising exports | Paths with >10% decline | Mean change in the bottom 10% (ES) |
|---|---:|---:|---:|
| Baseline | 98% | 2% | +1.1% |
| KRW/USD exchange rate +10% | 88% | 8% | -18.5% |
| U.S. 10-year Treasury yield (GS10) −15% | 80% | 14% | -18.0% |
| Economic recovery | 96% | 2% | +3.6% |

The recovery scenario combines GS10 −15%, PMI +7%, CLI +2%, ISM +5%, and KRW/USD −6%. A higher KRW/USD exchange rate worsened the downside outcomes. Lowering the Treasury yield alone did not improve the outlook either. Changing the economic indicators together improved the outcomes in the lower tail.

![Forecast export distributions under different economic conditions](assets/scenario-distributions.png)

The red recovery curve shifts to the right of the baseline distribution, showing that the model generates paths with higher export levels under those conditions.

The rising-path proportion counts generated paths that end above the reference export level. It is not a prediction accuracy score or a probability of stock prices rising. These experiments show the model's response to changed conditions and do not establish causal effects.

<sub>The scenario values above follow the final summary slide. Some values on the individual scenario slides differ, including the rate-decrease scenario's tail probability, VaR, and ES, and some recovery scenario values. These need to be reconciled against the experiment logs.</sub>

## Code and running the experiments

If you are looking through the code for the first time, `model.py` and `main.py` are good starting points.

| File | Contents |
|---|---|
| `macro_pretrain.py` | Macroeconomic encoder pretraining |
| `model.py` | Encoder, Decoder, Conditional Prior, and TimeVAE |
| `data_train.py` | Preprocessing, sliding windows, training, and evaluation functions |
| `main.py` | Model training and scenario experiments |
| `scenario_eval.py` | Path generation, plots, and probabilistic and risk metrics |
| `baseline_*.py` | ARIMA, LSTM, and vanilla cVAE baselines |
| `train_ablation_models.py`, `ablation_experiment.py` | Training and evaluation for ablation variants |
| `data.csv`, `macro.csv` | Monthly semiconductor and macroeconomic data |

The main dependencies are PyTorch, NumPy, pandas, scikit-learn, SciPy, statsmodels, matplotlib, and joblib. Run these commands from the project directory containing the CSV files:

```bash
python macro_pretrain.py
python main.py
```

This is experimental code, with paths and settings defined inside the scripts. Check the epochs, scenario conditions, and device setting in `main.py` before running it. Currently, `macro_pretrain.py` also runs its top-level code when imported, so the execution flow needs some cleanup.

## What is still left to do

The dataset is short, and the training windows overlap substantially. We would like to add data from more countries and industries, and revisit the evaluation with stricter time-based separation between training and testing. The forecast origin and consistency of preprocessing for the macro encoder also need attention in the current implementation.

We have not directly compared this model with GANs or diffusion models. Other directions we would like to explore are handling changes in economic regimes and generating the scenario conditions themselves.
