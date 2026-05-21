# Car Price Prediction — Neural Network

Predicts automobile prices from the [UCI Automobile dataset](https://archive.ics.uci.edu/dataset/10/automobile) using a deep neural network built with Keras / TensorFlow.

## Project structure

```
car-price-prediction/
├── car_price_prediction.py   # Full pipeline: preprocessing → training → evaluation
├── requirements.txt
├── data/
│   └── Automobile_data.csv   # Add dataset here (not committed)
└── plots/                    # Generated at runtime
    ├── training_loss.png
    ├── actual_vs_predicted.png
    ├── horsepower_vs_engine_size.png
    └── model_architecture.png
```

## Setup

```bash
pip install -r requirements.txt
```

Place `Automobile_data.csv` in the `data/` folder, then run:

```bash
python car_price_prediction.py
```

## Model architecture

Seven Dense layers (2048 → 1024 → 512 → 256 → 128 → 64 → 32 → 1) with:
- L2 regularisation (λ = 0.005) on each Dense layer
- Batch normalisation + LeakyReLU after each layer
- 50 % Dropout for regularisation
- Adam optimiser with exponential learning-rate decay
- Early stopping on validation loss (patience = 10)

## Results

| Split | MAE | MSE | R² |
|-------|-----|-----|----|
| Train | $1,587.90 | 3,523,282 | 0.911 |
| Test  | $2,569.52 | 15,073,126 | 0.887 |

## Observations

- Test R² of **0.887** means the model explains ~89% of the variance in car prices on unseen data — a strong result for this dataset.
- The train/test gap (MAE +62%, MSE ×4) indicates **moderate overfitting**, which is expected: the dataset has only ~190 rows after preprocessing, while the network has millions of parameters. Regularisation (L2 + Dropout + BatchNorm) limits but cannot eliminate this on such a small dataset.
- A shallower architecture (e.g. 3 layers: 256 → 128 → 64) would likely generalise better. The deep network is intentional here to explore the effect of depth and regularisation on a small dataset.
- Early stopping triggered well before 200 epochs in all runs, confirming the model converges quickly relative to its capacity.
