import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from keras.utils import plot_model

# ── Load ──────────────────────────────────────────────────────────────────────
data = pd.read_csv('data/Automobile_data.csv')
df = pd.DataFrame(data)

# ── Preprocessing ─────────────────────────────────────────────────────────────
df.replace('?', np.nan, inplace=True)

# Convert columns that should be numeric
numeric_cols = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm', 'price']
for col in numeric_cols:
    df[col] = df[col].astype(float)

# Fill missing num-of-doors using body-style context
df["num-of-doors"].replace(np.nan, "four", inplace=True)

# Fill missing horsepower using engine-size correlation
df["horsepower"].replace(np.nan, 112, inplace=True)

# Fill missing peak-rpm using feature-based mean
df["peak-rpm"].replace(np.nan, 5100, inplace=True)

# Drop duplicate rows before filling bore/stroke
df.drop([55, 56], axis=0, inplace=True)

# Fill missing bore and stroke using make-based mode
df["bore"].replace(np.nan, 3.39, inplace=True)
df["stroke"].replace(np.nan, 3.39, inplace=True)

# Fill missing normalized-losses with median
df["normalized-losses"].replace(np.nan, df['normalized-losses'].median(), inplace=True)

# Drop rows where price is unknown (target variable — cannot impute)
df.dropna(subset=["price"], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)

print("Dataset shape after preprocessing:", df.shape)
df.info()

# ── EDA Plots ─────────────────────────────────────────────────────────────────
plt.figure(figsize=(5, 5))
sns.scatterplot(x='horsepower', y='engine-size', data=df, color="red")
plt.title("Horsepower vs Engine Size")
plt.tight_layout()
plt.savefig("plots/horsepower_vs_engine_size.png")
plt.show()

# ── Feature Engineering ───────────────────────────────────────────────────────
cat_features = [
    'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
    'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders',
    'fuel-system'
]

encoded_data = pd.get_dummies(df, columns=cat_features)

X = encoded_data.drop('price', axis=1)
y = encoded_data['price']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(input_dim):
    model = Sequential()
    for units in [2048, 1024, 512, 256, 128, 64, 32]:
        model.add(Dense(units, kernel_regularizer=l2(0.005)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

model = build_model(X_train.shape[1])

lr_schedule = ExponentialDecay(initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9)
optimizer = Adam(learning_rate=lr_schedule)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model.compile(loss='mean_squared_error', optimizer=optimizer)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=200,
    batch_size=16,
    callbacks=[early_stopping]
)

# ── Evaluation ────────────────────────────────────────────────────────────────
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Train Loss (MSE): {train_loss:.4f}")
print(f"Test  Loss (MSE): {test_loss:.4f}")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

for split, y_true, y_pred in [("Train", y_train, y_train_pred), ("Test", y_test, y_test_pred)]:
    print(f"\n{split} Metrics")
    print(f"  MAE : {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"  MSE : {mean_squared_error(y_true, y_pred):.2f}")
    print(f"  R²  : {r2_score(y_true, y_pred):.4f}")

# ── Visualisation ─────────────────────────────────────────────────────────────
import os
os.makedirs("plots", exist_ok=True)

# Training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("plots/training_loss.png")
plt.show()

# Actual vs Predicted
comparison = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_test_pred.flatten()})
print(comparison.head(20))

comparison.head(40).plot(kind='bar', figsize=(14, 6))
plt.title("Actual vs Predicted Car Prices")
plt.grid(which='major', linestyle='-', linewidth=0.5, color='green')
plt.grid(which='minor', linestyle=':', linewidth=0.5, color='black')
plt.tight_layout()
plt.savefig("plots/actual_vs_predicted.png")
plt.show()

# Model architecture diagram
plot_model(model, to_file='plots/model_architecture.png', show_shapes=True, show_layer_names=True)
