import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def calculate_indicators(df):
    # Moving Averages relative to price
    df['SMA_7_Rel'] = df['Close'] / df['Close'].rolling(window=7).mean()
    df['SMA_20_Rel'] = df['Close'] / df['Close'].rolling(window=20).mean()
    df['SMA_50_Rel'] = df['Close'] / df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility (standardized)
    df['Volatility'] = df['Close'].pct_change().rolling(window=7).std()
    
    # MACD (standardized)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Rel'] = (exp1 - exp2) / df['Close']
    
    # Price Returns (History)
    for i in range(1, 6):
        df[f'Return_Lag_{i}'] = df['Close'].pct_change(periods=i)
        
    return df

def train_gold_model():
    print("Loading and preprocessing data...")
    df = pd.read_csv("data/gold_prices.csv", parse_dates=True, index_col=0)
    df.sort_index(inplace=True)
    
    df = calculate_indicators(df)
    
    # Target: Predict Next Day Return
    df['Target_Return'] = df['Close'].pct_change().shift(-1)
    
    df.dropna(inplace=True)
    
    # Split
    train_size = int(len(df) * 0.9)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    features = [
        'SMA_7_Rel', 'SMA_20_Rel', 'SMA_50_Rel', 'RSI', 'Volatility', 'MACD_Rel'
    ] + [f'Return_Lag_{i}' for i in range(1, 6)]
    
    X_train, y_train = train_df[features], train_df['Target_Return']
    X_test, y_test = test_df[features], test_df['Target_Return']
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
    
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50
    )
    
    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)], 
              verbose=False)
    
    # Evaluation
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Evaluation (Next Day Return):")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Accuracy check: Did it at least get the direction right?
    dir_true = np.sign(y_test)
    dir_pred = np.sign(predictions)
    accuracy = (dir_true == dir_pred).mean()
    print(f"Directional Accuracy: {accuracy:.2%}")
    
    # Save
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(model, "models/gold_model.pkl")
    joblib.dump(features, "models/features.pkl")
    print("\nModel saved successfully.")

if __name__ == "__main__":
    train_gold_model()
