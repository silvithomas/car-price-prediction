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
| Train | — | — | — |
| Test  | — | — | — |

*(Fill in after running)*
