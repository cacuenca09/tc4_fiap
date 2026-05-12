"""
LSTM Stock Price Predictor - Model Training
Empresa: Apple Inc. (AAPL)
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ─── Configurações ────────────────────────────────────────────────────────────
SYMBOL      = 'AAPL'
START_DATE  = '2018-01-01'
END_DATE    = '2024-07-20'
SEQ_LEN     = 60
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
BATCH_SIZE  = 32
EPOCHS      = 100
PATIENCE    = 15

PATHS = {
    'model':   'model/lstm_model.keras',
    'scaler':  'model/scaler.pkl',
    'meta':    'model/model_meta.json',
    'metrics': 'model/metrics.json',
    'plot':    'model/training_plot.png',
    'pred':    'model/predictions_plot.png',
}


def fetch_data(symbol, start, end):
    print(f"[1/6] Baixando dados de {symbol}…")
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    
    # Achatar colunas MultiIndex do yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.dropna(inplace=True)

    close  = df['Close'].values.flatten()
    high   = df['High'].values.flatten()
    low    = df['Low'].values.flatten()
    open_  = df['Open'].values.flatten()

    df['MA_7']     = pd.Series(close, index=df.index).rolling(7).mean()
    df['MA_21']    = pd.Series(close, index=df.index).rolling(21).mean()
    df['MA_50']    = pd.Series(close, index=df.index).rolling(50).mean()
    df['STD_21']   = pd.Series(close, index=df.index).rolling(21).std()
    df['Return']   = pd.Series(close, index=df.index).pct_change()
    df['HL_ratio'] = (high - low) / (close + 1e-9)
    df['OC_ratio'] = (close - open_) / (open_ + 1e-9)

    delta = pd.Series(close, index=df.index).diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    df.dropna(inplace=True)
    print(f"      {len(df)} registros | {df.shape[1]} colunas")
    return df


def preprocess(df, seq_len, train_r, val_r):
    print("[2/6] Pré-processando…")

    features = ['Close', 'Volume', 'MA_7', 'MA_21', 'MA_50',
                 'STD_21', 'Return', 'HL_ratio', 'OC_ratio', 'RSI']

    # Flatten para garantir array 2D limpo
    data = np.column_stack([df[f].values.flatten() for f in features]).astype(float)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    n       = len(scaled)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)

    def make_sequences(arr, seq):
        X, y = [], []
        for i in range(seq, len(arr)):
            X.append(arr[i - seq:i])
            y.append(arr[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = make_sequences(scaled[:n_train], seq_len)
    X_val,   y_val   = make_sequences(scaled[n_train:n_train + n_val], seq_len)
    X_test,  y_test  = make_sequences(scaled[n_train + n_val:], seq_len)

    test_dates = df.index[n_train + n_val + seq_len:]

    print(f"      Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, test_dates


def build_model(seq_len, n_features):
    print("[3/6] Construindo modelo LSTM…")
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(0.3),
        BatchNormalization(),

        LSTM(64, return_sequences=False),
        Dropout(0.3),

        Dense(32, activation='relu'),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='huber',
        metrics=['mae'],
    )
    model.summary()
    return model


def train(model, X_train, y_train, X_val, y_val):
    print("[4/6] Treinando…")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=7, min_lr=1e-6, verbose=1),
        ModelCheckpoint(PATHS['model'], save_best_only=True, verbose=0),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Treinamento LSTM — {SYMBOL}', fontsize=14, fontweight='bold')
    axes[0].plot(history.history['loss'],     label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss (Huber)')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history.history['mae'],     label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_title('MAE')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PATHS['plot'], dpi=150, bbox_inches='tight')
    plt.close()
    return history


def inv_close(scaler, arr, n_features):
    dummy = np.zeros((len(arr), n_features))
    dummy[:, 0] = arr.flatten()
    return scaler.inverse_transform(dummy)[:, 0].reshape(-1, 1)


def evaluate(model, X_test, y_test, scaler, test_dates):
    print("[5/6] Avaliando…")
    n_features = scaler.n_features_in_

    y_pred_s = model.predict(X_test, verbose=0)
    y_pred   = inv_close(scaler, y_pred_s, n_features)
    y_true   = inv_close(scaler, y_test,   n_features)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2   = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

    metrics = {'MAE': round(float(mae), 4), 'RMSE': round(float(rmse), 4),
               'MAPE': round(float(mape), 4), 'R2': round(float(r2), 4)}

    print(f"      MAE : {mae:.4f}")
    print(f"      RMSE: {rmse:.4f}")
    print(f"      MAPE: {mape:.4f}%")
    print(f"      R²  : {r2:.4f}")

    with open(PATHS['metrics'], 'w') as f:
        json.dump(metrics, f, indent=2)

    # Ajuste de tamanho para datas vs previsões
    min_len = min(len(test_dates), len(y_pred))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test_dates[:min_len], y_true[:min_len], label='Preço Real',    color='#1f77b4', linewidth=1.5)
    ax.plot(test_dates[:min_len], y_pred[:min_len], label='Previsão LSTM', color='#ff7f0e', linewidth=1.5, linestyle='--')
    ax.set_title(f'{SYMBOL} — Previsão de Fechamento\nMAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%  R²={r2:.4f}')
    ax.set_xlabel('Data'); ax.set_ylabel('Preço (USD)')
    ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PATHS['pred'], dpi=150, bbox_inches='tight')
    plt.close()
    return metrics


def save_artifacts(scaler, metrics):
    print("[6/6] Salvando artefatos…")
    os.makedirs('model', exist_ok=True)
    joblib.dump(scaler, PATHS['scaler'])
    meta = {
        'symbol': SYMBOL, 'start_date': START_DATE, 'end_date': END_DATE,
        'seq_len': SEQ_LEN, 'n_features': 10,
        'trained_at': datetime.utcnow().isoformat(),
        'metrics': metrics, 'paths': PATHS,
    }
    with open(PATHS['meta'], 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"      Modelo : {PATHS['model']}")
    print(f"      Scaler : {PATHS['scaler']}")


def main():
    print("=" * 60)
    print(" LSTM Stock Price Predictor — Apple Inc. (AAPL)")
    print("=" * 60)

    df = fetch_data(SYMBOL, START_DATE, END_DATE)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, test_dates = \
        preprocess(df, SEQ_LEN, TRAIN_RATIO, VAL_RATIO)

    model   = build_model(SEQ_LEN, n_features=X_train.shape[2])
    history = train(model, X_train, y_train, X_val, y_val)
    metrics = evaluate(model, X_test, y_test, scaler, test_dates)
    save_artifacts(scaler, metrics)

    print("\n✅ Pipeline concluído com sucesso!")
    print(f"   Métricas finais: {metrics}")


if __name__ == '__main__':
    main()
